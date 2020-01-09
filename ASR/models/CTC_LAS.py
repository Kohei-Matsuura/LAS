"""
The basical LAS model translated by Mr. Ueno
"""

import torch
import sys
import time
import copy
import numpy as np
from math import pow
import utils.tools as tools
import utils.BeamSearchTools as BS

from encoders.CTCEncoder import Encoder
from decoders.SimpleDecoder import Decoder
from attentions.LabAttention import Attention

class Model(torch.nn.Module):
    def __init__(self, hd):
        super(Model, self).__init__()
        self.encoder = Encoder(hd)
        self.attention = Attention(hd)
        self.decoder = Decoder(hd)

    def forward(self, px, bt, xl, tl, hd):
        """
        Tensor -> hyp-labels (one-hots)
        B*T*F -> B*L*W
        (F: Feature Size, W: Class Size)
        !!! bt & tl is used to train decoder.
        (with causing training bias)
        """
        onehot_targets = tools.t_batch_to_onehot(bt, tl, hd['CLASS_SIZE_ATTN'])

        CTC_dist, enc_hidden, enc_hl = self.encoder(px, xl)
        NUM_INPUT_CHANNEL = 1
        alpha = torch.zeros((hd['BATCH_SIZE'],
                                  NUM_INPUT_CHANNEL,
                                  max(xl))).to(hd['DEVICE'])
        dec_hidden = torch.zeros((hd['BATCH_SIZE'],
                                   hd['DEC_HID_SIZE'])).to(hd['DEVICE'])
        dec_memory = torch.zeros((hd['BATCH_SIZE'],
                                   hd['DEC_HID_SIZE'])).to(hd['DEVICE'])
        hyps = torch.zeros((max(tl - 1), hd['BATCH_SIZE'], hd['CLASS_SIZE_ATTN'])).to(hd['DEVICE'])

        # (T-1)*B*W
        input = torch.zeros((hd['BATCH_SIZE'], hd['CLASS_SIZE_ATTN']))

        mask = torch.ones((hd['BATCH_SIZE'], max(xl))).to(hd['DEVICE'])

        for i, tmp in enumerate(xl):
            if tmp < max(xl):
                mask.data[i, tmp:] = 0.0

        for i, t in enumerate(bt):
            input[i, t[0]] = 1.0

        st1 = time.time()
        for i in range(max(tl) - 1):
            # <EOS> won't be input into decoder, so 'len - 1'
            alpha = self.attention(enc_hidden, dec_hidden, alpha, mask)
            alpha = alpha.transpose(1, 2)
            # alpha: B*T*1

            context_vector = (alpha * enc_hidden).sum(dim=1)

            alpha = alpha.transpose(1, 2)
            # alpha: B*1*T

            input = onehot_targets[:, i, :]

            st = time.time()
            #print('START DECODE')
            dec_hidden, dec_memory, hyps[i] = self.decoder(dec_hidden,
                                                            context_vector,
                                                            dec_memory,
                                                            input)
            # hyps[i + 1]: B*W
        # hyps: T*B*W
        attn_hyps = hyps.transpose(0, 1)
        # hyps: B*T*W
        return CTC_dist, attn_hyps, onehot_targets, bt, xl, tl


    def decode(self, px, xl, beam_width, output_source, hd):
        """
        px, xl -> dist
        """
        ctc_dist, enc_hidden, _ = self.encoder(px, xl)

        if output_source == 'CTC' or output_source == 'ctc':
            result_ids = []
            # plus, not multiple because in log space

            for j in ctc_dist[0]:
                result_ids.append(np.argmax(j.detach().cpu()))
            best_hyps = tools.contractor(result_ids, hd['CLASS_SIZE'])
            best_hyps = [best_hyps]
        else:
            beam_list = []
            NUM_INPUT_CHANNEL = 1
            for i in range(beam_width):
                # Define tensors each BeamCells, since tensor is mutable ;(
                alpha = torch.zeros((hd['BATCH_SIZE'],
                                    NUM_INPUT_CHANNEL,
                                    max(xl))).to(hd['DEVICE'])
                dec_hidden = torch.zeros((hd['BATCH_SIZE'],
                                        hd['DEC_HID_SIZE'])).to(hd['DEVICE'])
                dec_memory = torch.zeros((hd['BATCH_SIZE'],
                                         hd['DEC_HID_SIZE'])).to(hd['DEVICE'])
                init_params = {'hid': dec_hidden, 'alpha': alpha, 'mem': dec_memory}
                init_hyp = [hd['SOS_ID']]
                #init_hyp = [sos_id]
                b = BS.BeamCell(i, init_hyp, init_params, alpha)
                beam_list.append(b)

            mask = torch.ones((hd['BATCH_SIZE'], max(xl))).to(hd['DEVICE'])

            for i, tmp in enumerate(xl):
                if tmp < max(xl):
                    mask.data[i, tmp:] = 0.0
            #for i in range(4):
            #    #print(id(beam_list[i].params['hid']))
            #exit()
            iter = 0

            while(not(BS.is_all_end(beam_list, hd)) and iter < 100):
                iter += 1
                dists = []
                #c1 = time.time()
                for b in beam_list:
                    if not(b.hyp[-1] == hd['EOS_ID']):
                        b.params['alpha'] = self.attention(enc_hidden,
                                                b.params['hid'],
                                                b.params['alpha'],
                                                mask)
                        b.params['alpha'] = b.params['alpha'].transpose(1, 2)
                        context_vector = (b.params['alpha'] * enc_hidden).sum(dim=1)
                        b.params['alpha'] = b.params['alpha'].transpose(1, 2)
                        # alpha: B*1*T
                        onehot_input = tools.int_to_onehot(b.hyp[-1], hd['CLASS_SIZE_ATTN'])
                        onehot_input = torch.FloatTensor(onehot_input).to(hd['DEVICE'])
                        # W
                        onehot_input = onehot_input.unsqueeze(0)
                        # 1*W
                        b.params['hid'], b.params['mem'], dist = self.decoder.decode(onehot_input,
                                                                    b.params['hid'],
                                                                    context_vector,
                                                                    b.params['mem'])
                        # Be careful, here must be 'len + 1' not only 'len'
                        dists.append([b, dist])
                    else:
                        # BeamCells already done
                        dists.append([b, torch.tensor([])])
                best_beams = BS.get_N_best(dists, beam_width, hd)
                beam_list = BS.make_next_beam_list(beam_list, best_beams, hd)
            best_hyps = BS.get_hyps(beam_list, beam_width, hd)
        return best_hyps

    def make_trans_martrix(self, output_ids, hd):
        other_ids = []
        for i in range(hd['CLASS_SIZE_ATTN']):
            if not(i in output_ids):
                other_ids.append(i)
        trans_matrix = np.zeros((len(other_ids), len(output_ids)))
        for i, out in enumerate(output_ids):
            for j, other in enumerate(other_ids):
                trans_matrix[j][i] = self.decoder.MSE_from_two_class(out, other, hd)
        return trans_matrix
