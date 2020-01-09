# Bash script for training

# Argments
SAVE_DIR=${1}
SCRIPT_DIR=${2}
CLASS_SIZE=${3}

# Setting which GPU to use
gpu=0

mkdir ${SAVE_DIR}/params

CUDA_VISIBLE_DEVICES=${gpu} python systems/train-CTC-LAS.py ${SCRIPT_DIR} ${SCRIPT_DIR} ${CLASS_SIZE} ${CLASS_SIZE} ${SAVE_DIR} | tee ${SAVE_DIR}/loss.log

