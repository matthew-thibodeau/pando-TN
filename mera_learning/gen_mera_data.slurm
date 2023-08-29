#!/bin/bash

#SBATCH -o /nfs_home/npsawaya/proj/pando-TN/mera_learning/outs/job.%j.out
#SBATCH -D /nfs_home/npsawaya/proj/pando-TN/mera_learning
#SBATCH -J mera_trunc 
#SBATCH --get-user-env
#SBATCH --partition=clxtrb
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1

source /swtools/intel/compilers_and_libraries_2020.4.304/linux/bin/compilervars.sh intel64
source /swtools/intel/compilers_and_libraries_2020.4.304/linux/mpi/bin64/mpivars.sh
source /swtools/intel/compilers_and_libraries_2020.4.304/linux/tbb/bin/tbbvars.sh intel64
export INTEL_LICENSE_FILE=/swtools/intel/licenses/
export I_MPI_HYDRA_BOOTSTRAP=slurm

conda activate pando
srun python mera_sample_generator.py 32 




