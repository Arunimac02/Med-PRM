#!/bin/bash
# Full-scale multimodal trace generation, Qwen3-VL-8B-Instruct only, across
# all 11,286 questions (vqa_rad + slake + medxpertqa_mm). Runs as its own
# SLURM job on its own GPU, in parallel with the Huatuo run
# (3_generate_traces_full_huatuo.sh) instead of both models sharing one GPU
# sequentially -- see prm_docs/smoketest_analysis.txt for the throughput
# data and time-limit reasoning behind this split.
set -e

INPUT_FILE="dataset_mm/normalized/mm_benchmark.json"
OUTPUT_DIR="dataset_mm/generated_traces"

# Matches the text pipeline's own default (scripts/3_test_dataset_sampling.sh)
# exactly, per the paper -- ~1.4M generations across both models at this
# dataset's full scale (11,286 questions x 64 samples x 2 models).
REPEAT_COUNT=64
TEMPERATURE=0.7
TOP_K=50
TOP_P=0.9
MAX_TOKENS=4096

# max_model_len: real dataset max prompt length after --max_pixels capping
# is 7628 tokens (measured across all 11,286 questions via
# python_mm/measure_prompt_lengths.py, same for both models) + 4096
# generation budget = 11,724 minimum; 12000 leaves a small buffer.
# max_pixels: caps this model's own dynamic-resolution image tiling to a
# ~1280-tokens-per-image budget (1280*1024, since Qwen3-VL uses 1024
# pixels/token -- 16px patch x 2x merge). Without this, a single
# high-resolution image could cost 9000+ tokens on its own.
# max_images_per_prompt: 102 medxpertqa_mm questions have 5-6 images;
# vllm's default assumption of 4 would reject those prompts outright.
MAX_MODEL_LEN=12000
MAX_PIXELS=1310720
MAX_IMAGES_PER_PROMPT=6

GPU_ID=0

MODEL_PATH="Qwen/Qwen3-VL-8B-Instruct"
MODEL_NAME=$(basename "$MODEL_PATH")

log_dir="logs_mm/inference"
mkdir -p "$log_dir"

echo "=== Full run: $MODEL_PATH ==="
python python_mm/3_generate_traces.py \
  --model_path "$MODEL_PATH" \
  --gpu_id "$GPU_ID" \
  --input_file "$INPUT_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --repeat_count "$REPEAT_COUNT" \
  --temperature "$TEMPERATURE" \
  --top_k "$TOP_K" \
  --top_p "$TOP_P" \
  --max_tokens "$MAX_TOKENS" \
  --max_model_len "$MAX_MODEL_LEN" \
  --max_pixels "$MAX_PIXELS" \
  --max_images_per_prompt "$MAX_IMAGES_PER_PROMPT" \
  > "${log_dir}/full_${MODEL_NAME}.log" 2>&1
echo "=== Done: $MODEL_PATH -> ${log_dir}/full_${MODEL_NAME}.log ==="
