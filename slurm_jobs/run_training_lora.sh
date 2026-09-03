#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE               # Do not propagate login shell environment

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=medprm_training_lora
#SBATCH --time=96:00:00             # gpu partition MaxTime; ~63h needed to finish from checkpoint-950 at observed pace
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                   # 8B model in bf16 + activations + LoRA; adjust if you hit OOM
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1           # 1x A100; swap to a40:1 if a100 queue is busy
#SBATCH --output=logs/train/slurm_%x_%j.out
#SBATCH --error=logs/train/slurm_%x_%j.err

##OPTIONAL JOB SPECIFICATIONS
##SBATCH --account=YOUR_ACCOUNT_ID  # uncomment and fill in if your allocation requires it
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=your_email@tamu.edu

# --- Modules to load each session (match what you used to build the env) ---
module purge
module load Anaconda3/2025.12-2
module load CUDA/12.4.1   # matches torch's cu124 build; sets CUDA_HOME so deepspeed's
                           # op-builder import (pulled in by accelerate's unwrap_model,
                           # even though we don't use deepspeed for training) doesn't crash

# --- Conda environment ---
source "$(conda info --base)/etc/profile.d/conda.sh"   # required so 'conda activate' works in a non-interactive batch shell
conda activate PRM

# --- Move into the repo ---
cd $SCRATCH/Med-PRM    # adjust path if your clone lives elsewhere

# --- Make sure log dir exists (script also does this, but just in case) ---
mkdir -p logs/train

# --- Run the LoRA training script ---
bash scripts/2_training_lora.sh
