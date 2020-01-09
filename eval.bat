# Bash script for evaluating models

# Getting Args
PARAMS=${1}
SCRIPT=${2}
CLASS_SIZE=${3}

# Setting GPU
gpu=0

# Using GPU
CUDA_VISIBLE_DEVICE=${gpu} python systems/eval-CTC-LAS.py ${PARAMS} ${SCRIPT} 4 attn ${CLASS_SIZE} ${CLASS_SIZE}

