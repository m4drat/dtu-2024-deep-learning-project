#!/bin/sh

# A100 GPU queue, there is also gpua40 and gpua10
#BSUB -q gpua100

# job name
#BSUB -J faces-train

# 4 cpus, 1 machine, 1 gpu, 24 hours (the max)
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00

# at least 8 GB RAM
#BSUB -R "rusage[mem=8GB]"

# stdout/stderr files for debugging (%J is substituted for job ID)
#BSUB -o /dtu/blackhole/14/204020/deep-learning/job_out/faces_gan_%J.out
#BSUB -e /dtu/blackhole/14/204020/deep-learning/job_out/faces_gan_%J.err

# your training script here, e.g.
# activate environment ...
module load python3/3.9.18
module load cuda/12.4

cd /dtu/blackhole/14/204020/deep-learning
source /dtu/blackhole/14/204020/deep-learning/.venv/bin/activate

python /dtu/blackhole/14/204020/deep-learning/faces-gan.py


