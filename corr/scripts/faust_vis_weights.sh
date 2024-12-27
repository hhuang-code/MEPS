#!/usr/bin/env bash

source ~/.bashrc
conda activate meps

export LD_LIBRARY_PATH=/share/apps/NYUAD/cuda/11.8.0/lib64:$LD_LIBRARY_PATH

python faust_vis_weights.py faust --model_name faust_best.pth.tar