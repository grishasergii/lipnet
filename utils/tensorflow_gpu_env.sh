#!/bin/bash

# To run it from the terminal:
# chmod +x ./tensorflow_gpu_env.sh
# source ./tensorflow_gpu_env.sh
# prompt will change to: (tensorflow_gpu) sergii@sergii:

# Set path to CUDA
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"
export CUDA_HOME=/usr/local/cuda

# Activate Tensorflow GPU virtual environment
source ~/tensorflow_gpu/bin/activate

