"""
This is hyper-parameters of models.
When used, it have to be imported.
"""

hd = {}

hd['align1'] = '{:-^30}'.format('')

hd['SOS_ID'] = 2
hd['EOS_ID'] = 1
hd['UNK_ID'] = 0

# Settings
hd['align2'] = '{:-^25}'.format('Setting')

#hd['CTC_INPUT_FEATURE_SIZE'] = 120

hd['EPOCH_NUM'] = 200
hd['BATCH_SIZE'] = 3

# Rarely changed
hd['align3'] = '{:-^25}'.format('')
hd['NUM_ENC_LAYERS'] = 5
hd['NUM_CTC_LAYERS'] = 5
hd['NUM_DEC_LAYERS'] = 1

hd['align4'] = '{:-^25}'.format('')
hd['ENC_HID_SIZE'] = 320
hd['ATTN_HID_SIZE'] = 640
hd['DEC_HID_SIZE'] = 320

hd['DEC_BOTTLE_NECK_SIZE'] = 320

hd['align5'] = '{:-^25}'.format('')
hd['EMB_SIZE'] = 320
hd['lambda'] = 0.8

hd['align6'] = '{:-^25}'.format('')
# Never changed (maybe)
hd['SEED'] = 100
hd['SPEC_AUG'] = False
hd['FEATURE_SIZE'] = 120
hd['LEARNING_RATE'] = 0.001
hd['WEIGHT_DECAY'] = 1e-5
hd['CLIPPING'] = 5.0
hd['DROPOUT'] = 0.2

hd['align7'] = '{:-^30}'.format('')
