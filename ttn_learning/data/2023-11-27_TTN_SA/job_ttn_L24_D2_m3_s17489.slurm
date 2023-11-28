#!/bin/bash

#SBATCH -o data/2023-11-27_TTN_SA/job_ttn_L24_D2_m3_s17489.o%j
#SBATCH -e data/2023-11-27_TTN_SA/job_ttn_L24_D2_m3_s17489.err%j
#SBATCH -D ./
#SBATCH --get-user-env
#SBATCH --partition=spr
#SBATCH --time=23:59:00
#SBATCH --ntasks=28
#SBATCH --cpus-per-task=8
#SBATCH -J job_ttn_L24_D2_m3_s17489
#SBATCH --mail-user=gian.giacomo.guerreschi@intel.com
#SBATCH --mail-type=begin --mail-type=end

source ~/anaconda3/bin/activate pando-env
export I_MPI_HYDRA_BOOTSTRAP=slurm
mpiexec.hydra -genv I_MPI_DEBUG 4 -genv OMP_NUM_THREADS 8 -genv KMP_AFFINITY compact,granularity=thread,verbose python3 ./ttn_perform_SA_with_MPI.py -L 24 -m 3 -s 17489 --runstart 28 --runend 56 -i 1000 --today 2023-11-27
