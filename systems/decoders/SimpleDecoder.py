"""
The implementation of a simple decoder.
This decoder has only 1 layer.
"""

import torch
import torch.nn as nn
import utils.tools as tools
import numpy as np

class Decoder(nn.Module):
    def __init__(self, hd):
        super(Decoder, self).__init__()
        concat_size = (hd['ENC_HID_SIZE'] * 2) + hd['EMB_SIZE']
        self.decoder_LSTM = nn.LSTMCell(concat_size, hd['DEC_HID_SIZE'])
        self.embed = nn.Linear(hd['CLASS_SIZE_ATTN'], hd['EMB_SIZE'], bias=False)
        #self.concat_Lin = nn.Linear(concat_size, hd['DEC_HID_SIZE'], bias=False)
        self.dec_hid_to_y = nn.Linear(hd['DEC_HID_SIZE'], hd['DEC_BOTTLE_NECK_SIZE'], bias=False)
        self.attn_hid_to_y = nn.Linear(hd['ENC_HID_SIZE'] * 2, hd['DEC_BOTTLE_NECK_SIZE'])
        self.y_to_dist = nn.Linear(hd['DEC_BOTTLE_NECK_SIZE'], hd['CLASS_SIZE_ATTN'])
        self.tanh = nn.Tanh()
        #self.affine = nn.Linear(hd['DEC_HID_SIZE'], hd['CLASS_SIZE'])


    def forward(self, prev_dec_hidden, attn_enc_hidden, prev_dec_memory, targets):
        # onehot_targets: B*W (W: Word Size)
        emb_targets = self.embed(targets)
        # B*E (E: emb_size)
        concat_inputs = torch.cat((emb_targets, attn_enc_hidden), dim=1)
        # B*(E + H_mid)
        next_hid, next_mem = self.decoder_LSTM(concat_inputs, (prev_dec_hidden, prev_dec_memory))
        # Both: B*H_dec
        bottle_neck = self.dec_hid_to_y(next_hid) + self.attn_hid_to_y(attn_enc_hidden)
        tanh_bottle_neck = self.tanh(bottle_neck)
        hyp_dist = self.y_to_dist(tanh_bottle_neck)
        hyp_dist = nn.functional.softmax(hyp_dist, dim=1)
        # distribution of hyp: B*W
        return next_hid, next_mem, hyp_dist


    def decode(self, input, dec_hidden, context_vector, memory):
        """
        On evaluation, decoding must be done without targets.
        """
        emb_input = self.embed(input)
        concat_input = torch.cat((emb_input, context_vector), dim=1)
        #prev_mix_hidden = nn.Tanh(self.concat_Lin(concat_hiddens))
        next_hid, next_mem = self.decoder_LSTM(concat_input, (dec_hidden, memory))
        #y = self.affine(next_hid)
        bottel_neck = self.dec_hid_to_y(next_hid) + self.attn_hid_to_y(context_vector)
        tanh_bottle_neck = self.tanh(bottel_neck)
        hyp_dist = self.y_to_dist(tanh_bottle_neck)
        hyp_dist = nn.functional.softmax(hyp_dist, dim=1)
        # dist: B*W
        hyp_dist = hyp_dist.squeeze(0)
        # hyp_dist: W

        return next_hid, next_mem, hyp_dist


class Aux_Decoder(Decoder):
    # for multi-lingual learning
    def __init__(self, hd):
        super(Aux_Decoder, self).__init__(hd)
        self.embed = nn.Linear(hd['CLASS_SIZE_AUX'], hd['EMB_SIZE'], bias=False)
        self.y_to_dist = nn.Linear(hd['DEC_BOTTLE_NECK_SIZE'], hd['CLASS_SIZE_AUX'])

class Aux2_Decoder(Decoder):
    # for multi-lingual learning
    def __init__(self, hd):
        super(Aux2_Decoder, self).__init__(hd)
        self.embed = nn.Linear(hd['CLASS_SIZE_AUX2'], hd['EMB_SIZE'], bias=False)
        self.y_to_dist = nn.Linear(hd['DEC_BOTTLE_NECK_SIZE'], hd['CLASS_SIZE_AUX2'])
