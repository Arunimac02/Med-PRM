#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE               # Do not propagate login shell environment

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=medprm_scoring
#SBATCH --time=48:00:00             # adjust based on how long scoring takes; bump if it times out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                   # 8B model in bf16 + activations; adjust if you hit OOM
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2           # 2x A100; input dataset is sharded across both in scripts/4_scoring_lora_PRM.sh
#SBATCH --output=logs/inference/slurm_%x_%j.out
#SBATCH --error=logs/inference/slurm_%x_%j.err

##OPTIONAL JOB SPECIFICATIONS
##SBATCH --account=YOUR_ACCOUNT_ID  # uncomment and fill in if your allocation requires it
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=your_email@tamu.edu

# --- Modules to load each session (match what you used to build the env) ---
module purge
module load Anaconda3/2025.12-2

# --- Conda environment ---
source "$(conda info --base)/etc/profile.d/conda.sh"   # required so 'conda activate' works in a non-interactive batch shell
conda activate PRM

# --- Move into the repo ---
cd $SCRATCH/Med-PRM    # adjust path if your clone lives elsewhere

# --- Make sure log dir exists (script also does this, but just in case) ---
mkdir -p logs/inference

# --- Run the scoring script ---
bash scripts/4_scoring_lora_PRM.sh