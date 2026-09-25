#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE               # Do not propagate login shell environment

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=medprm_mm_smoketest
#SBATCH --time=01:00:00             # smoke test only: ~12 questions x 4 samples x 2 models
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1           # 1x A100; models run sequentially, not concurrently
#SBATCH --output=logs_mm/inference/slurm_%x_%j.out
#SBATCH --error=logs_mm/inference/slurm_%x_%j.err

##OPTIONAL JOB SPECIFICATIONS
##SBATCH --account=YOUR_ACCOUNT_ID  # uncomment and fill in if your allocation requires it
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=your_email@tamu.edu

# --- Modules to load each session (match what you used to build the env) ---
module purge
module load Anaconda3/2025.12-2
module load CUDA/13.1.0             # provides nvcc; PRM_mm's torch is built for CUDA 13.0,
                                     # needed for flashinfer's sampler to JIT-compile its kernel

# --- Conda environment ---
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate PRM_mm

# --- Compute nodes on this cluster have no internet access; both models are
# --- already fully downloaded to the local HF cache, so force offline mode
# --- rather than letting huggingface_hub/vllm try (and fail) to reach the Hub.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# --- Move into the repo ---
cd $SCRATCH/Med-PRM

# --- Make sure log/output dirs exist (script also does this, but just in case) ---
mkdir -p logs_mm/inference dataset_mm/smoketest_traces

# --- Run the smoke test ---
bash scripts_mm/3_generate_traces_smoketest.sh
