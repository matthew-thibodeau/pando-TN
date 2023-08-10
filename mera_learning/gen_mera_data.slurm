#!/bin/bash

#SBATCH -o /nfs_home/npsawaya/pando-TN/mera_learning/outs/job.%j.out
#SBATCH -D /nfs_home/npsawaya/pando-TN/mera_learning
#SBATCH -J mera_trunc 
#SBATCH --get-user-env
#SBATCH --partition=clxtrb
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=4


srun /nfs_home/npsawaya/anaconda3/bin/python mera_sample_generator.py 32 

