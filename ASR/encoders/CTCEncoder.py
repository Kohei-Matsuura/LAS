"""
This Encoder return two hiddens.
That is, N + M layers of BiLSTMs.
"""

import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, hd):
        """
        fs: the size of lmfb (maybe 120)
        hs: the num of LSTM cell
        nl: the number of BiLSTM (default: 5)
        dp: dropout (default: 0.2)
        """
        super(Encoder, self).__init__()
        self.feature_size = hd['FEATURE_SIZE']
        self.hidden_size = hd['ENC_HID_SIZE']

        self.affine = nn.Linear(hd['ENC_HID_SIZE'] * 2, hd['CLASS_SIZE'] + 1)

        self.firstBiLSTMs = nn.LSTM(input_size=hd['FEATURE_SIZE'],
                               hidden_size=hd['ENC_HID_SIZE'],
                               num_layers=hd['NUM_CTC_LAYERS'],
                               batch_first=True,
                               dropout=hd['DROPOUT'],
                               bidirectional=True)
        self.post_layer_size = hd['NUM_ENC_LAYERS'] - hd['NUM_CTC_LAYERS']
        if self.post_layer_size > 0:
            self.secondBiLSTMs = nn.LSTM(input_size=hd['ENC_HID_SIZE'] * 2,
                                   hidden_size=hd['ENC_HID_SIZE'],
                                   num_layers=self.post_layer_size,
                                   batch_first=True,
                                   dropout=hd['DROPOUT'],
                                   bidirectional=True)


    def forward(self, px, xl):
        """
        px: pps of x / simgle x
        xl: x's lengths
        t: target tensor
        """
        self.firstBiLSTMs.flatten_parameters()

        ph1, _ = self.firstBiLSTMs(px)
        bh1, hl = nn.utils.rnn.pad_packed_sequence(ph1, batch_first=True, total_length=max(xl))

        pre_dist = self.affine(bh1)
        CTC_dist = torch.nn.functional.log_softmax(pre_dist, dim=2)

        if self.post_layer_size > 0:
            self.secondBiLSTMs.flatten_parameters()
            ph2, _ = self.secondBiLSTMs(ph1)
            bh2, hl = nn.utils.rnn.pad_packed_sequence(ph2, batch_first=True, total_length=max(xl))
        else:
            bh2 = bh1

        return CTC_dist, bh2, hl
