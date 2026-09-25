#!/bin/bash
# Small end-to-end smoke test of python_mm/3_generate_traces.py: a handful
# of questions per data source, few samples, both VLMs -- validates
# image attachment, prompt format, and scoring before a full-scale run.
set -e

INPUT_FILE="dataset_mm/normalized/mm_benchmark.json"
OUTPUT_DIR="dataset_mm/smoketest_traces"

REPEAT_COUNT=4
TEMPERATURE=0.7
TOP_K=50
TOP_P=0.9
MAX_TOKENS=4096        # matches the text pipeline's default; 2048 truncated long medxpertqa_mm vignettes mid-answer in the first smoke test
LIMIT_PER_SOURCE=4

# See scripts_mm/3_generate_traces_full_qwen.sh / _huatuo.sh for how these
# values are derived (python_mm/measure_prompt_lengths.py). max_pixels is
# model-specific (each model has its own pixels-per-image-token ratio), so
# it's a parallel array matching MODELS' order.
MAX_MODEL_LEN=12000
MAX_IMAGES_PER_PROMPT=6

GPU_ID=0

MODELS=(
  "Qwen/Qwen3-VL-8B-Instruct"
  "FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL"
)
MAX_PIXELS_LIST=(
  1310720
  1003520
)

log_dir="logs_mm/inference"
mkdir -p "$log_dir"

for i in "${!MODELS[@]}"; do
  MODEL_PATH="${MODELS[$i]}"
  MAX_PIXELS="${MAX_PIXELS_LIST[$i]}"
  MODEL_NAME=$(basename "$MODEL_PATH")
  echo "=== Smoke test: $MODEL_PATH ==="
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
    --limit_per_source "$LIMIT_PER_SOURCE" \
    > "${log_dir}/smoketest_${MODEL_NAME}.log" 2>&1
  echo "=== Done: $MODEL_PATH -> ${log_dir}/smoketest_${MODEL_NAME}.log ==="
done
