#!/bin/bash

# Slurm sbatch options
#SBATCH -o mpi/mpi_%j.log
#SBATCH -n 17
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu:volta:1
#SBATCH --time=4-00:00:00

# Initialize the module command first
source /etc/profile

# Load MPI module
module load mpi/openmpi-4.0

# Load Anaconda module
module load anaconda/2020a

# Call your script as you would from the command line
echo "mpirun python -B train_agent.py $env_name $model --nsteps $nsteps"
mpirun python -B train_agent.py $env_name $model --nsteps $nsteps