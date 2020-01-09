"""
You can train any models in this script (maybe).
"""

# system libs
import sys
import os

# standard
import time
import random
import numpy as np
import copy

# PyTorch libs
import torch

# other miscellaneous
import utils.tools as tools
import hparams
from models.CTC_LAS import Model

ATTN_SCRIPT = sys.argv[1]
CTC_SCRIPT = sys.argv[2]
CTC_CLASS_SIZE = int(sys.argv[3])
ATTN_CLASS_SIZE = int(sys.argv[4])
SAVE_DIR = sys.argv[5]

print('ATTN_SCRIPT: ' + ATTN_SCRIPT)
print('CTC_SCRIPT: ' + CTC_SCRIPT)
print('CTC_CLASS_SIZE: ' + str(CTC_CLASS_SIZE))
print('ATTN_CLASS_SIZE: ' + str(ATTN_CLASS_SIZE))
print('SAVE_DIR: ' + SAVE_DIR)
print()

hd = hparams.hd
tools.print_dict(hd)

hd['CLASS_SIZE'] = CTC_CLASS_SIZE
hd['CLASS_SIZE_ATTN'] = ATTN_CLASS_SIZE
DEVICE = torch.device('cuda' if torch.cuda.is_available else 'cpu')
hd['DEVICE'] = DEVICE

# Seeding
random.seed(hd['SEED'])
np.random.seed(hd['SEED'])
torch.manual_seed(hd['SEED'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Setting the model trainable
my_model = Model(hd)
my_model.apply(tools.init_weight)
my_model.train()

# Sending the model to GPU
my_model.to(hd['DEVICE'])

optimizer = torch.optim.Adam(my_model.parameters(),
                              lr=hd['LEARNING_RATE'],
                              weight_decay=hd['WEIGHT_DECAY'])
ctc_loss = torch.nn.CTCLoss(blank=hd['CLASS_SIZE'], reduction='sum')

# Hashing the script file
attn_htk_label_list = []
with open(ATTN_SCRIPT) as f:
   for l in f:
        attn_htk_label_list.append(l.split(' ', 1))
FILE_LINE_NUM_ATTN = len(attn_htk_label_list)
print('FILE_LINE_NUM(ATTN): ' + str(FILE_LINE_NUM_ATTN))


ctc_htk_label_list = []
with open(CTC_SCRIPT) as f:
   for l in f:
        ctc_htk_label_list.append(l.split(' ', 1))
FILE_LINE_NUM_CTC = len(ctc_htk_label_list)
print('FILE_LINE_NUM(CTC): ' + str(FILE_LINE_NUM_CTC))


if FILE_LINE_NUM_CTC == FILE_LINE_NUM_ATTN:
    FILE_LINE_NUM = FILE_LINE_NUM_CTC
else:
    print('ERROR: text lengths mismatch')
    exit()

# Now, train is beginning...
start = int(round(time.time()))
for e in range(0, hd['EPOCH_NUM']):
#for e in range(1):
    num_iter = FILE_LINE_NUM // hd['BATCH_SIZE']
    if e == 30 or e == 35:
        hd['LEARNING_RATE'] = hd['LEARNING_RATE'] * 0.1
        optimizer = torch.optim.Adam(my_model.parameters(),
                                     lr=hd['LEARNING_RATE'],
                                     weight_decay=hd['WEIGHT_DECAY'])
        print('LEARNING_RATE: ', hd['LEARNING_RATE'])

    for i in range(0, num_iter):
        st = time.time()
        loss = 0.0
        loss_attn = 0.0
        loss_ctc = 0.0
        ctc_list_x, ctc_list_t = tools.list_to_batches(ctc_htk_label_list, hd['BATCH_SIZE'], FILE_LINE_NUM, i, False)
        attn_list_x, attn_list_t = tools.list_to_batches(attn_htk_label_list, hd['BATCH_SIZE'], FILE_LINE_NUM, i, False)
        #print(ctc_list_t)
        #print(attn_list_t)
        #print()
        assert ctc_list_x == attn_list_x, 'ASERTION ERROR: list_x mismatch!'

        # x must be the same, but t is not so
        px, ctc_bt, xl, ctc_tl = tools.batch_to_pps(ctc_list_x, ctc_list_t)
        px, attn_bt, xl, attn_tl = tools.batch_to_pps(attn_list_x, attn_list_t)

        # Going through the model
        # attn_list_t is used at the teacher forcing.
        ctc_dists, hyp_dists, onehot_targets, attn_bt, xl, attn_tl = my_model(px, attn_bt, xl, attn_tl, hd)

        # CTC
        ctc_dists = ctc_dists.transpose(0, 1)
        loss_ctc = ctc_loss(ctc_dists, ctc_bt, xl, ctc_tl) / hd['BATCH_SIZE']

        # ATTN
        LS_targets = onehot_targets * 0.9 + 0.1 / hd['CLASS_SIZE_ATTN']
        for b in range(hd['BATCH_SIZE']):
            loss_attn = loss_attn + tools.cross_entropy(hyp_dists[b], LS_targets[b][1:]) / attn_tl[b]

        loss = hd['lambda'] * loss_attn + ( 1.0 - hd['lambda']) * loss_ctc

        if not torch.isnan(loss):
            prev_model = copy.deepcopy(my_model)
            #del prev_optimizer
            prev_optimizer = copy.deepcopy(optimizer)
            print('iter: ' + str(e + 1) + '-' + str(i + 1))
            print('loss_attn: ' + str(loss_attn.item()))
            print('loss_ctc: ' + str(loss_ctc.item()))
            print('loss: ' + str(loss.item()))
            now = int(round(time.time()))
            print(str(now - start) + ' seconds have passed.')
            print()
            sys.stdout.flush()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(my_model.parameters(), hd['CLIPPING'])
            optimizer.step()
        else:
            print('loss_ctc is nan. skipping...')
            #import pdb; pdb.set_trace()
            my_model = prev_model
            optimizer = prev_optimizer

            optimizer = torch.optim.Adam(my_model.parameters(),
                              lr=hd['LEARNING_RATE'],
                              weight_decay=hd['WEIGHT_DECAY'])
            optimizer.load_state_dict(prev_optimizer.state_dict())

        loss.detach()
        torch.cuda.empty_cache()

    torch.save(my_model.state_dict(), SAVE_DIR + '/params/epoch' + str(e + 1) + '.net')
    torch.save(optimizer.state_dict(), SAVE_DIR + '/params/epoch' + str(e + 1) + '.opt')
