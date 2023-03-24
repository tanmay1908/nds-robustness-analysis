#!/bin/bash
#
#SBATCH --job-name=vt
#SBATCH --account=viscam
#SBATCH --partition=viscam --qos=normal
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=10
#SBATCH --time 192:00:00
#SBATCH --output=/viscam/u/tanmayx/projects/neural-deferred-shading/env_create_%A.out
#SBATCH --error=/viscam/u/tanmayx/projects/neural-deferred-shading/env_create_%A.err
#SBATCH --mail-user=tanmayx@stanford.edu
#SBATCH --mail-type=ALL
######################
# Begin work section #
######################
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

##########################################
# Setting up virtualenv / conda / docker #
##########################################
# example here if using virtualenv
source /viscam/u/tanmayx/.bashrc
##############################################################
# Setting up LD_LIBRARY_PATH or other env variable if needed #
##############################################################
# export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64:/usr/lib/x86_64-linux-gnu 
echo "Working with the LD_LIBRARY_PATH: "$LD_LIBRARY_PATH
###################
# Run your script #
###################
cd /viscam/u/tanmayx/projects/neural-deferred-shading
conda env create -f environment.yml
