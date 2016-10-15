#!/bin/bash

#SBATCH --time=0:20:00   # walltime
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:4 # number of GPUs
#SBATCH --mem-per-cpu=10240M   # memory per CPU core
#SBATCH -J "tf_gan"   # job name
#SBATCH --qos=test # tell them it's a test job to get it running asap

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
python gan_lab7.py
