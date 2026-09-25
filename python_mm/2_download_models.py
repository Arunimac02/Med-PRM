#!/usr/bin/env python
# coding: utf-8
"""
Pre-download the two vision-language models used for multimodal trace
generation, via huggingface_hub.snapshot_download rather than letting
vllm.LLM(...) lazy-download them on first use, so the download can be
verified (size, file completeness) before any GPU job depends on it.

Downloads land in the default HF cache (~/.cache/huggingface), which on
this system is symlinked to /scratch/user/arunimac02/.cache/huggingface
-- confirmed via `readlink -f ~/.cache/huggingface` before writing this
script, so there was no need to override HF_HOME to a custom folder.

Models:
- Qwen/Qwen3-VL-8B-Instruct               (~17.5GB)
- FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL (~15GB)
"""
from huggingface_hub import snapshot_download

MODELS = [
    "Qwen/Qwen3-VL-8B-Instruct",
    "FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL",
]

if __name__ == "__main__":
    for repo_id in MODELS:
        print(f"=== Downloading {repo_id} ===")
        path = snapshot_download(repo_id=repo_id)
        print(f"=== Done: {repo_id} -> {path} ===")
