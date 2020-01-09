"""
The calculation of the attention.
[alpha_(l-1), decoder_hidden_(l-1), encoder_output]
->[alpha(l)]
"""

import time
import numpy as np
import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hd):
        super(Attention, self).__init__()

        self.hd = hd

        self.num_kernel = 10
        self.padding_size = 50
        self.kernel_width = self.padding_size * 2
        # tensors for 'elements to score'
        self.ConvAttn = nn.Conv1d(1, self.num_kernel, self.kernel_width, stride=1, bias=False, padding=self.padding_size)
        self.ConvToScore = nn.Linear(self.num_kernel, hd['ATTN_HID_SIZE'], bias=False)
        self.HiddenEncToScore = nn.Linear(hd['ENC_HID_SIZE'] * 2, hd['ATTN_HID_SIZE'])
        self.HiddenDecToScore = nn.Linear(hd['DEC_HID_SIZE'], hd['ATTN_HID_SIZE'], bias=False)
        self.CalcScore = nn.Linear(hd['ATTN_HID_SIZE'], 1, bias=False)


    def forward(self, enc_output, prev_dec_hidden, prev_alpha, mask):
        start = time.time()
        T = enc_output.size(1) #max(xl)
        start = time.time()
        #alpha = torch.zeros((self.hd['BATCH_SIZE'], 1, T)).cuda()

        #mask = torch.ones((bs, T)).cuda()
        #
        #for i, tmp in enumerate(xl):
        #    if tmp < T:
        #        mask.data[i, tmp:] = 0.0
        # mask: B*T
        #print('ATTN_START: ' + str(time.time() - start))
        # prev_alpha is B*1*T
        #print('CHECK7 ATTENTION: ' + str(time.time() - st))
        tmpconv = self.ConvAttn(prev_alpha)
        # B*10*(T+1)
        #print('CHECK6 ATTENTION: ' + str(time.time() - st))
        tmpconv = tmpconv.transpose(1, 2)[:, :T, :]
        # B*T*10
        #print('CHECK5 ATTENTION: ' + str(time.time() - st))

        prev_attn_e = self.ConvToScore(tmpconv)
        # B*T*mid_hidden_size

        prev_dec_e = self.HiddenDecToScore(prev_dec_hidden)
        # B*dec_hidden_size

        prev_dec_e = prev_dec_e.unsqueeze(1)
        # B*1*mid_hidden_size

        enc_e = self.HiddenEncToScore(enc_output)
        # B*T*mid_hidden_size
        #print('CHECK1 ATTENTION: ' + str(time.time() - st))

        e = torch.tanh(enc_e + prev_dec_e + prev_attn_e)
        # B*T*mid_hidden_size (Broadcasted)

        e = self.CalcScore(e)
        # B*T*1

        e = e.squeeze(2)
        # B*T

        e_exp = (e - e.max(1)[0].unsqueeze(1)).exp()

        e_exp = e_exp * mask
        #print('CHECK3 ATTENTION: ' + str(time.time() - st))
        # B*T

        alpha = e_exp / e_exp.sum(dim=1, keepdim=True)
        #print('CHECK2 ATTENTION: ' + str(time.time() - st))
        # B*T

        alpha = alpha.unsqueeze(1)
        # B*1*T: to return the same shape
        #print('CHECK1 ATTENTION: ' + str(time.time() - st))

        #print('ATTN_END: ' + str(time.time() - start))
        return alpha
