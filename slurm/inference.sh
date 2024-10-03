#!/bin/bash
#SBATCH --gres=gpu:1 # number of gpus per node
#SBATCH --mem=128GB # memory per node 
#SBATCH --cpus-per-task=32
#SBATCH --time=0-11:59 # time (DD-HH:MM) 
#SBATCH --output=log_jobid_%A_arr_%a.out # %A for main jobID, %a for array id.
#SBATCH --partition=3dlg-hcvc-lab-short
#SBATCH -J job_name

REPOSITORY_NAME="3d-grand"
SINGULARITY_IMAGE="3dgrand.sif"

# cleanup
ulimit -Su unlimited
ulimit -Sv unlimited
cd "/localscratch/atw7/${REPOSITORY_NAME}"

# export variables
# export SINGULARITYENV_CUDA_VISIBLE_DEVICES="$SLURM_JOB_GPUS"

singularity exec --nv --bind "$(pwd):/mnt" --pwd /mnt --env-file .env "${SINGULARITY_IMAGE}" python3.11 inference.py --prompts data/scannetpp/test_prompts_debug.csv
