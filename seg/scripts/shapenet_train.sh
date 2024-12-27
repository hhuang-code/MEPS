#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --gres=gpu:a100:1
##SBATCH --gres=gpu:rtx8000:2
#SBATCH --mem=80GB
##SBATCH --exclusive
#SBATCH --time=96:00:00
#SBATCH --job-name='shapenet'
#SBATCH -p nvidia
##SBATCH --reservation=gpu
#SBATCH -q cair
#SBATCH -w cn010

#SBATCH --mail-type=END
#SBATCH --mail-user=hh1811@nyu.edu

source ~/.bashrc
conda activate meps

export LD_LIBRARY_PATH=/share/apps/NYUAD/cuda/11.8.0/lib64:$LD_LIBRARY_PATH

python shapenet_train.py shapenet --lr_decay_step 500
#python shapenet_train.py shapenet --resume_model_name best_shapenet.pth.tar