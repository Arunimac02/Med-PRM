#!/bin/bash

############################################
# Load Environment Variables
############################################
# Load environment variables from .env file
if [ -f ".env" ]; then
  source ".env"
fi

# Exit if environment variables are not set
if [ -z "$HF_TOKEN" ]; then
  echo "Error: HF_TOKEN is not set. Please check .env file."
  exit 1
fi

############################################
# User-defined Parameters
############################################
# Default settings

USE_RAG="yes"
USE_ORM="no"

# DATA_SOURCE_LIST='["med_qa"]' Process all if empty
PROCESS_SOLUTION_NUM=64

MODEL_PATH="model_train/meta-llama/Llama-3.1-8B-Instruct-gemini_label-filter_yes-ep3-20260731_191535-RAG_yes"
INPUT_JSON="dataset/dataset_3_sampled_dataset/llama-3.1-medprm-reward-test-set/2_test_dataset.json"
# GPUs to shard the input dataset across. Same model, same script, same
# get_prob()/process_json_with_prm() logic as a single-GPU run -- just split
# by question so each GPU scores a disjoint subset in parallel, then merge.
GPUS=(0 1)

# Set output directory
OUTPUT_DIR="dataset/dataset_4_scored_dataset"

# Set maximum token length (1024 for other PRMs, 4096 for RAG-PRM)
MAX_TOKEN_LEN=4096

# Set option inclusion (yes/no)
INCLUDE_OPTIONS="no"

# Create log directory
LOG_DIR="logs/inference"
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

BASE_INPUT_NAME="$(basename "$INPUT_JSON" .json)"
# Extract first data source element
FIRST_DATA_SOURCE=$(echo $DATA_SOURCE_LIST | sed -E 's/\[\"([^\"]+)\".*/\1/')

MODEL_BASENAME="${MODEL_PATH##*/}"
FINAL_OUTPUT_JSON="${OUTPUT_DIR}/${MODEL_BASENAME}_${FIRST_DATA_SOURCE}_sol${PROCESS_SOLUTION_NUM}_${BASE_INPUT_NAME}.json"
NUM_SHARDS=${#GPUS[@]}
SHARD_DIR="${OUTPUT_DIR}/shards_${MODEL_BASENAME}"
mkdir -p "$SHARD_DIR"

# --------------------------------------------------------------
# Split the input JSON (a top-level list of questions) into
# NUM_SHARDS contiguous chunks, one per GPU.
# --------------------------------------------------------------
python -c "
import json
with open('$INPUT_JSON', encoding='utf-8') as f:
    data = json.load(f)
n = $NUM_SHARDS
chunk = -(-len(data) // n)  # ceil division
for i in range(n):
    shard = data[i * chunk : (i + 1) * chunk]
    with open(f'$SHARD_DIR/shard{i}.json', 'w', encoding='utf-8') as f:
        json.dump(shard, f, ensure_ascii=False)
    print(f'Shard {i}: {len(shard)} questions')
"

# Parallel execution: one shard per GPU, same model/script/args each time
for i in "${!GPUS[@]}"; do
    GPU="${GPUS[$i]}"
    SHARD_INPUT="${SHARD_DIR}/shard${i}.json"
    SHARD_OUTPUT="${SHARD_DIR}/shard${i}_out.json"
    LOG_FILE="${LOG_DIR}/TEST_$(date +'%Y%m%d_%H%M%S')_${MODEL_BASENAME}_shard${i}.log"

    echo "====== Evaluation Settings (Model: ${MODEL_BASENAME}, Shard: ${i}, GPU: ${GPU}) ======" | tee -a "$LOG_FILE"
    echo "Model Path: $MODEL_PATH" | tee -a "$LOG_FILE"
    echo "Shard Input File: $SHARD_INPUT" | tee -a "$LOG_FILE"
    echo "Shard Output File: $SHARD_OUTPUT" | tee -a "$LOG_FILE"
    echo "GPU: $GPU" | tee -a "$LOG_FILE"
    echo "RAG Usage: $USE_RAG" | tee -a "$LOG_FILE"
    echo "Max Token Length: $MAX_TOKEN_LEN" | tee -a "$LOG_FILE"
    echo "Include Options: $INCLUDE_OPTIONS" | tee -a "$LOG_FILE"
    echo "ORM Usage: $USE_ORM" | tee -a "$LOG_FILE"
    echo "Number of Solutions to Process: $PROCESS_SOLUTION_NUM" | tee -a "$LOG_FILE"
    echo "====================" | tee -a "$LOG_FILE"

#        --data_source_list "$DATA_SOURCE_LIST" \
    python python/4_scoring_PRM.py \
        --model_save_path "$MODEL_PATH" \
        --input_json_file "$SHARD_INPUT" \
        --output_json_file "$SHARD_OUTPUT" \
        --device "$GPU" \
        --hf_token "$HF_TOKEN" \
        --use_rag "$USE_RAG" \
        --max_token_len "$MAX_TOKEN_LEN" \
        --include_options "$INCLUDE_OPTIONS" \
        --use_orm "$USE_ORM" \
        --process_solution_num "$PROCESS_SOLUTION_NUM" 2>&1 | tee -a "$LOG_FILE" &
done

wait
echo "All shards completed. Merging outputs..."

python -c "
import json, glob
shards = sorted(glob.glob('$SHARD_DIR/shard*_out.json'),
                 key=lambda p: int(p.split('shard')[-1].split('_')[0]))
merged = []
for p in shards:
    with open(p, encoding='utf-8') as f:
        merged.extend(json.load(f))
with open('$FINAL_OUTPUT_JSON', 'w', encoding='utf-8') as f:
    json.dump(merged, f, indent=4, ensure_ascii=False)
print(f'Merged {len(merged)} questions from {len(shards)} shards into $FINAL_OUTPUT_JSON')
"

echo "All model evaluations completed. Final output: $FINAL_OUTPUT_JSON"
