#!/usr/bin/env bash
module load cuda/10.0 cudnn/7.4
source ../venv/bin/activate
export PYTHONUNBUFFERED=1
srun -p gpi.compute -t 1-00 -c 4 --mem 4GB --gres gpu:turing:1 python 03_sequential_mnist.py --log_dir ../logs