#!/usr/bin/env python
# coding: utf-8
"""
One-off measurement: find the true maximum prompt length (question text +
image vision-tokens, as each model's own processor would tokenize it) across
the whole dataset, for both models. Used to size --max_model_len correctly
in python_mm/3_generate_traces.py instead of guessing/padding a number --
the earlier max_model_len=4096 (mistakenly reused from --max_tokens) crashed
on a real Huatuo prompt at 4210 tokens.

CPU-only (transformers AutoProcessor, no vllm/GPU needed) -- this only
tokenizes, it never runs a forward pass.
"""
import json
import os

from PIL import Image
from transformers import AutoProcessor

DATA_PATH = "/scratch/user/arunimac02/Med-PRM/dataset_mm/normalized/mm_benchmark.json"
REPO_ROOT = "/scratch/user/arunimac02/Med-PRM"

MODELS = {
    "Qwen3-VL-8B-Instruct": "Qwen/Qwen3-VL-8B-Instruct",
    "HuatuoGPT-Vision-7B-Qwen2.5VL": "FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL",
}

# Cap each model's own dynamic-resolution image tiling to a shared ~1280
# image-tokens-per-image budget (1280 is the Qwen-VL family's own commonly
# used example default, not an arbitrary number), computed from each
# model's own patch/merge geometry: Huatuo (Qwen2.5-VL) = 784 pixels per
# token, Qwen3-VL = 1024 pixels per token.
MAX_PIXELS = {
    "Qwen3-VL-8B-Instruct": 1280 * 1024,
    "HuatuoGPT-Vision-7B-Qwen2.5VL": 1280 * 784,
}

SYSTEM_PROMPT_MC = (
    "You are a medical expert answering a question about the attached "
    "image(s). Solve the question step-by-step. Do not analyze individual "
    "options in a single step. Each step of your explanation must start "
    "with '## Step {number}: ' format. You must provide the answer using "
    "the phrase 'the answer is (option alphabet)' at the end of your step."
)
SYSTEM_PROMPT_QA = (
    "You are a medical expert answering a question about the attached "
    "image(s). Solve the question step-by-step. Each step of your "
    "explanation must start with '## Step {number}: ' format. The final "
    "answer must output a concise and clearly defined diagnostic term, or "
    "simply 'Yes' or 'No' if the question is a yes/no question. You must "
    "provide the final answer using the phrase '## Final Answer: {answer}' "
    "at the end of your final step. Please refer to the following "
    "examples: '## Final Answer: Yes' or '## Final Answer: Right kidney'."
)


def build_conversation(q):
    sys_prompt = SYSTEM_PROMPT_MC if q["answer_type"] == "MC" else SYSTEM_PROMPT_QA
    content = [{"type": "image"} for _ in q["images"]]
    content.append({"type": "text", "text": q["question"]})
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": content},
    ]


def main():
    data = json.load(open(DATA_PATH, encoding="utf-8"))
    print(f"Loaded {len(data)} questions")

    for label, repo_id in MODELS.items():
        print(f"\n=== {label} ===")
        processor = AutoProcessor.from_pretrained(
            repo_id, trust_remote_code=True, max_pixels=MAX_PIXELS[label]
        )

        max_len = 0
        max_qid = None
        lengths = []
        for i, q in enumerate(data):
            conv = build_conversation(q)
            text = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            images = [Image.open(os.path.join(REPO_ROOT, p)).convert("RGB") for p in q["images"]]
            inputs = processor(text=[text], images=images if images else None, return_tensors="pt")
            n_tokens = inputs["input_ids"].shape[1]
            lengths.append(n_tokens)
            if n_tokens > max_len:
                max_len = n_tokens
                max_qid = q["question_id"]
            if (i + 1) % 1000 == 0:
                print(f"  ...{i + 1}/{len(data)} processed, running max so far: {max_len}")

        lengths.sort()
        n = len(lengths)
        print(f"  max prompt length: {max_len} tokens (question_id={max_qid})")
        print(f"  p50={lengths[n // 2]}  p95={lengths[int(n * 0.95)]}  p99={lengths[int(n * 0.99)]}")


if __name__ == "__main__":
    main()
