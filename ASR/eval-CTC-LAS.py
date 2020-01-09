# system libs
import sys
import os

# standard
import time
import random
import numpy as np

# PyTorch libs
import torch

# other miscellaneous
import utils.tools as tools
import utils.BeamSearchTools as BS
from models.CTC_LAS import Model
import tmp.eval_hparams as hp

# Since with which save to decode is not identical,
# you have to choose and tell it here.
MODEL_DIR = sys.argv[1]
SCRIPT_DIR = sys.argv[2]
BEAM_WIDTH = int(sys.argv[3])
#sos_id = int(sys.argv[4])
src = sys.argv[4]
CTC_CLASS_SIZE = int(sys.argv[5])
ATTN_CLASS_SIZE = int(sys.argv[6])

hd = hp.hd
hd['BATCH_SIZE'] = 1
hd['SPEC_AUG'] = False
hd['CLASS_SIZE'] = CTC_CLASS_SIZE
hd['CLASS_SIZE_ATTN'] = ATTN_CLASS_SIZE
# In decode state, BATCH SIZE must be 1

# Calculate the time from making dir to the end of training
start = int(round(time.time()))

# load ids to mask languages
#output_ids = np.load("ASR/tmp/id_list.npy")
# WARNING: this relative path is from WHERE YOU EXEC THIS SCRIPT, not from this script.
#print(output_ids)
#sys.exit(0)

# loading the model
trained_params = torch.load(MODEL_DIR)
trained_model = Model(hd)
trained_model.load_state_dict(trained_params)
trained_model.eval()

# Sending the model to GPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    trained_model.to(DEVICE)

#x = trained_model.make_trans_martrix(output_ids, hd)
#print(np.argmax(x, axis=1))
#print(x)
#sys.exit(0)


htk_files = tools.file_to_list(SCRIPT_DIR)

for htk in htk_files:
    x = tools.load_dat(htk, hd['SPEC_AUG'])
    x = torch.FloatTensor(x).unsqueeze(0).cuda()
    # B*T*120

    xl = [len(x[0])]
    # B*T (B = 1)

    px = torch.nn.utils.rnn.pack_padded_sequence(x, xl, batch_first=True)

    beam_hyps = trained_model.decode(px, xl, BEAM_WIDTH, src, hd)

    #BS.print_hyps(beam_hyps, htk, 4, hd)
    sys.stdout.flush()
    BS.print_hyps(beam_hyps, htk, 1, hd)
    #print()
    #print(str(int(round(time.time())) - start) + ' seconds have passed.')
    #exit()
