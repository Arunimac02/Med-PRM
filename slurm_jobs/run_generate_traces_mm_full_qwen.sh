#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE               # Do not propagate login shell environment

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=medprm_mm_full_qwen
#SBATCH --time=96:00:00             # 4 days -- this is the GPU partition's hard max (checked via
                                     # `sinfo -p gpu`); zero margin past this, but batched/resumable
                                     # output (see python_mm/3_generate_traces.py --batch_size) means
                                     # a kill right at this wall just needs a resubmit, not a restart.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G                   # sized from an exact measurement of all ~12,139 dataset images:
                                     # largest per-data_source chunk (SLAKE) = 9.58GB decoded images
                                     # + ~3.5GB vllm engine overhead -> ~13-14GB realistic peak, 24G leaves
                                     # ~10GB margin. See prm_docs/smoketest_analysis.txt for the full derivation.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1           # own GPU, runs in parallel with the Huatuo job (separate script)
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
mkdir -p logs_mm/inference dataset_mm/generated_traces

# --- Run the full-scale generation for Qwen only ---
# --- If this job is killed/times out, just resubmit (sbatch) this same
# --- script again -- completed batches are skipped automatically.
bash scripts_mm/3_generate_traces_full_qwen.sh
