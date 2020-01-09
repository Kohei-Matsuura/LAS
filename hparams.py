"""
This is hyper-parameters of models.
When used, it have to be imported.
"""

hd = {}

"""
# Training / Decoding script.
hd['align1'] = '{:-^30}'.format('I/O')
hd['TRAIN_SCRIPT'] = ''
hd['DECODE_SCRIPT'] = ''

# Directory to save model-params / results
hd['align2'] = '{:-^30}'.format('Save')
hd['SAVE_DIR'] = '20190419_1652'
"""

# Settings
hd['align3'] = '{:-^30}'.format('Setting')

#hd['CLASS_SIZE'] = 25
#hd['CLASS_SIZE_ATTN'] = 494 # 256 # 494
#hd['CLASS_SIZE_CTC'] = 25
#hd['CLASS_SIZE_AUX'] = 44
#hd['CLASS_SIZE_AUX2'] = 75
#hd['MAIN_TAG'] = 'ainu'
#hd['AUX_TAG'] = 'jnas'
#hd['AUX2_TAG'] = 'wsj'

#hd['SPEAKER_NUM'] = 7
#hd['SD_BOTTLE_NECK'] = 1024 #int(hd['SPEAKER_NUM'] / 2)
#hd['SPEAKER_LAMBDA'] = -3.0

hd['CTC_INPUT_FEATURE_SIZE'] = 120

hd['EPOCH_NUM'] = 40
hd['BATCH_SIZE'] = 30
#hd['BEAM_WIDTH'] = 4

# Rarely changed
hd['align4'] = '{:-^25}'.format('')
hd['NUM_ENC_LAYERS'] = 5
hd['NUM_CTC_LAYERS'] = 5
hd['NUM_DEC_LAYERS'] = 1

hd['align5'] = '{:-^25}'.format('')
hd['ENC_HID_SIZE'] = 320
hd['ATTN_HID_SIZE'] = 640
hd['DEC_HID_SIZE'] = 320
#hd['CONCAT_SIZE'] = hd['ENC_HID_SIZE'] * 2 + hd['DEC_HID_SIZE']

hd['DEC_BOTTLE_NECK_SIZE'] = 320

hd['align6'] = '{:-^25}'.format('')
hd['EMB_SIZE'] = 320
hd['lambda'] = 0.8
#hd['coverage_lambda'] = 0.2
#hd['coverage_tau'] = 0.1
#hd['LS_lambda'] = 0.9

hd['align7'] = '{:-^25}'.format('')
# Never changed (maybe)
hd['SEED'] = 100
hd['SPEC_AUG'] = False
hd['FEATURE_SIZE'] = 120
hd['LEARNING_RATE'] = 0.001
hd['WEIGHT_DECAY'] = 1e-5
hd['CLIPPING'] = 5.0
hd['DROPOUT'] = 0.2

hd['align8'] = '{:-^25}'.format('')
hd['SOS_ID'] = 2
hd['EOS_ID'] = 1
hd['UNK_ID'] = 0

hd['align9'] = '{:-^30}'.format('')
