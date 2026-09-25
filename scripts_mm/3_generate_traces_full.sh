#!/bin/bash
# Full-scale multimodal trace generation across all 11,286 questions
# (vqa_rad + slake + medxpertqa_mm) x both VLMs. Validated end-to-end via
# 4 smoke tests first (see prm_docs/smoketest_analysis.txt) -- image
# attachment, prompt/extraction robustness, and per-data_source memory
# chunking are all confirmed working before running this at full scale.
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

GPU_ID=0

MODELS=(
  "Qwen/Qwen3-VL-8B-Instruct"
  "FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL"
)

log_dir="logs_mm/inference"
mkdir -p "$log_dir"

for MODEL_PATH in "${MODELS[@]}"; do
  MODEL_NAME=$(basename "$MODEL_PATH")
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
    > "${log_dir}/full_${MODEL_NAME}.log" 2>&1
  echo "=== Done: $MODEL_PATH -> ${log_dir}/full_${MODEL_NAME}.log ==="
done
