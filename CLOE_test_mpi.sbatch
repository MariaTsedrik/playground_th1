#!/bin/bash
#SBATCH --job-name=cloempi
#SBATCH --constraint=100GbE
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH -N 1               # Total number of nodes 
#SBATCH -n 32            # Total number of mpi tasks
#SBATCH --output="aa-cloe-paral-musigma-lmax500-emupriors-mpi.out"
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH --mail-user=mtsedrik@ed.ac.uk



#module load anaconda/3.9
. /usr/local/anaconda/3.9/etc/profile.d/conda.sh
conda activate cloe_env


# If bash is your shell
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
cd CLOE/playground/scripts/sampling

mpiexec -n 32 python -m mpi4py.futures nautilus_example_update_mpi.py
