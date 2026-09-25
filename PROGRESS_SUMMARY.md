# Multimodal Med-PRM — Progress Summary

Prepared 2026-09-25 from the repository's code, configs, logs, notes, and git history. All numbers are quoted directly from the files cited; anything not directly verifiable in the repo is marked **[UNVERIFIED]**.

---

## 1. Project Overview

**Base project (inherited, not this student's original work):** *Med-PRM: Medical Reasoning Models with Step-wise Guideline-verified Process Rewards* — a text-only medical Process Reward Model (PRM), accepted as an **Oral Presentation at EMNLP 2025** (arXiv:2506.11474). It uses retrieval-augmented generation (RAG) to verify each step of a model's chain-of-thought reasoning against medical knowledge, and trains a Llama-3.1-8B step-level reward model on that supervision (`README.md`).

**This student's work extends it in two phases:**

**Phase A — Reproduce the text-only PRM with feasible compute.** The original codebase's training script (`python/2_training.py`) does full fine-tuning of the 8B model, which doesn't fit in memory on a single 40GB A100 without DeepSpeed. A parallel **LoRA fine-tuning path** was added (`python/2_training_lora.py`) so the whole pipeline could actually be run and validated on the available cluster hardware, then the result was checked against the paper's published numbers.

**Phase B — Multimodal Med-PRM (the current focus).** Extends the text pipeline to medical **visual** question answering: generating chain-of-thought reasoning traces (grounded in medical images) from two vision-language models, with correctness labels, as training data for a future multimodal PRM.

**What is actually implemented today:**
- Text pipeline: data prep → RAG/Gemini judge labeling → LoRA training → PRM scoring → benchmark reporting. Fully working, reproduction verified against the paper (Section 5).
- Multimodal pipeline: dataset download/normalization (3 VQA benchmarks unified into one schema) → chain-of-thought trace generation from two frozen 7-8B vision-language models (Qwen3-VL-8B-Instruct, HuatuoGPT-Vision-7B-Qwen2.5VL) at full scale (11,286 questions × 64 samples × 2 models). **Not yet implemented:** any multimodal PRM/ORM training script, and the OPEN-answer correctness-judging step (needed before the generated traces can be used as clean training labels).

**Architecture (text PRM, `model_train/llama-3.1-medprm-reward-v1.0/config.json`):** Llama-3.1-8B-Instruct backbone (32 layers, hidden size 4096, 32 attention heads / 8 KV heads, vocab 128,257 — vocab was extended for a new step-boundary token). Step-level reward is read out via next-token probability of `+`/`-` immediately after a special `" ки"` marker inserted at each reasoning step (`model_train/llama-3.1-medprm-reward-v1.0/README.md`).

**No multimodal PRM architecture exists yet** — the current multimodal artifacts are raw CoT generations from off-the-shelf VLMs, not a trained reward model.

---

## 2. Timeline

Git history alone understates the work: most of the multimodal effort (`dataset_mm/`, `python_mm/`, `scripts_mm/`, `slurm_jobs/*_mm_*`, `logs_mm/`, `prm_docs/`) is **currently untracked in git** (confirmed via `git status`), and the LoRA-related commits were made well after the runs they describe actually happened. The timeline below is reconstructed from git log, SLURM job logs, and file modification times.

| Date | Event | Source |
|---|---|---|
| 2025-05-27 | Original Med-PRM repo initialized | `git log` |
| 2025-06-15–18 | Initial code + README commits (pre-existing authors) | `git log` |
| 2025-06-15/17 | Med-PRM arXiv preprint announced | `README.md` News |
| 2025-08-20 | Med-PRM accepted to EMNLP 2025 | `README.md` News |
| 2025-09-15 | Med-PRM selected as EMNLP 2025 Oral Presentation | `README.md` News |
| 2026-07-25 | Conda environment (`PRM`) debugged and fixed for this cluster (pip resolver conflict between `vllm`, `trl`, `transformers`; flash-attn wheel install) | `ENVIRONMENT_SETUP_NOTES.txt` (mtime 2026-07-25) |
| 2026-07-29 | LoRA training run_1 fails (no internet on compute node); run_2 fails (DeepSpeed/CUDA_HOME crash on checkpoint save); run_3 starts | `LORA_TRAINING_AND_SCORING_CHANGES.txt`, `logs/train/run_1..3` |
| 2026-07-29 → 07-31 | run_3 runs ~46.5h, reaches step 1402/2850 (49% through 3 epochs), killed by SLURM time limit; checkpoint-950 saved | same |
| 2026-07-31 | run_4 fails resuming from checkpoint (PyTorch ≥2.6 `torch.load(weights_only=True)` rejects `rng_state.pth`); fixed; run_5 launched (19:15) | same |
| 2026-08-03 | run_5 completes all 3 epochs (train_runtime 226,242s ≈ 62.8h); LoRA adapter merged and saved | `logs/train/run_5/TRAIN_..._20260731_191535...log` |
| 2026-08-03 | PRM scoring of the LoRA model on the 5,469-question test set (2-GPU sharded run, job 19292651) | `logs/inference/run_5/TEST_...log` |
| 2026-08-04 | Merged scored output (5,469 records) written | `dataset/dataset_4_scored_dataset/...json` mtime |
| 2026-08-08 | `LORA_TRAINING_AND_SCORING_CHANGES.txt` written documenting the LoRA path and run history | file mtime |
| 2026-08-30 | LoRA reproduction verified against the paper's published numbers, benchmark-by-benchmark (`REPRODUCTION_VERIFICATION.txt`) | file mtime |
| 2026-09-02 | LoRA training/scoring code and SLURM scripts committed to git (4 commits) — this is when the already-completed July/August work was actually checked in | `git log` |
| 2026-09-03 | 7-benchmark reproduction report generated (`dataset_5_benchmark_report/`) | file mtime |
| 2026-09-04 | Multimodal phase begins: VQA-RAD, SLAKE, MedXpertQA-MM downloaded and normalized into one 11,286-row schema; new `PRM_mm` conda env created; both VLM weights downloaded | `prm_docs/setup.txt`, file mtimes |
| 2026-09-05–06 | Smoke tests 1–4 on ~12 questions: environment debugging (vllm API changes, offline-mode, CUDA/nvcc), extraction-logic fixes, memory-scaling fix (per-`data_source` chunking) | `prm_docs/smoketest_analysis.txt` |
| 2026-09-07 (early) | Smoke test 5 passes cleanly with all fixes combined | `prm_docs/setup.txt`, dir mtime 02:09 |
| 2026-09-07 02:04 → 11:54 | Full-scale trace generation, HuatuoGPT-Vision-7B: **completed**, 9h49m41s | `logs_mm/inference/slurm_medprm_mm_full_huatuo_19673588.out`/log |
| 2026-09-07 02:04 → 22:55 | Full-scale trace generation, Qwen3-VL-8B: **completed**, 20h51m25s | `logs_mm/inference/slurm_medprm_mm_full_qwen_19673587.out`/log |
| 2026-09-07 23:06–23:14 | Per-model/per-data_source outputs reorganized and merged into consolidated files | `prm_docs/setup.txt`, output file mtimes |

---

## 3. Data

### Text pipeline
- **Training set:** `dataset/dataset_1_train_dataset/llama-3.1-medprm-reward-training-set/1_train_dataset.json` — used by training; `LORA_TRAINING_AND_SCORING_CHANGES.txt` reports **11,700 examples** used in run_5 (1 skipped for a data error).
- **Test set:** `dataset/dataset_3_sampled_dataset/llama-3.1-medprm-reward-test-set/2_test_dataset.json` — **5,469 questions**, up to 64 sampled solutions each, spanning 13 `data_source` subsets (MedQA-4: 1,273; MedQA-5: 1,273; PubMedQA: 500; MedMCQA: 500; DDXPlus: 500; MMLU subsets totaling 1,089; OSCE: 214; NEJM: 120), per `REPRODUCTION_VERIFICATION.txt`.
- **Step labels:** generated by `python/1_train_dataset_RAG_judge_labeling.py`, an LLM-as-judge pipeline using Google's Gemini API to label per-step reasoning validity against retrieved documents (RAG), following the original Med-PRM method.

### Multimodal pipeline (`dataset_mm/`)
Three medical VQA benchmarks normalized into one common schema (`python_mm/1_normalize_data.py` → `dataset_mm/normalized/mm_benchmark.json`):

| Source | Rows | Splits | Answer type |
|---|---|---|---|
| VQA-RAD | 2,248 | train 1,796 / test 451 / unmatched 1 | CLOSED/OPEN |
| SLAKE | 7,033 | train 4,919 / val 1,053 / test 1,061 (English-only) | CLOSED/OPEN |
| MedXpertQA-MM | 2,005 | dev 5 / test 2,000 | MC |
| **Total** | **11,286** | — | MC 2,005 / CLOSED 4,080 / OPEN 5,201 |

(Row counts and answer-type breakdown independently re-verified directly against `dataset_mm/normalized/mm_benchmark.json` in this session, matching `prm_docs/setup.txt` exactly.)

- VQA-RAD downloaded from the official OSF release (not the HF mirror) for richer metadata, reconciled to the standard train/test split via text+image-hash matching.
- SLAKE downloaded from the original bilingual repo, filtered to English only.
- MedXpertQA-MM downloaded via `snapshot_download` reading raw JSONL (the HF `datasets` loader couldn't parse the newer schema under the pinned `datasets==3.6.0`).
- All ~12,139 image references verified to decode correctly; 0 missing files.

**How synthetic step labels are generated for the multimodal data (current state):** `python_mm/3_generate_traces.py` prompts each VLM to produce step-by-step reasoning (`## Step N:` markers) plus a final answer. `prm_processed_solution`/`orm_processed_solution` fields are then produced by **pure text post-processing** (not by a model) — stripping markdown, appending the literal `" ки"` boundary token per step or once per solution — mirroring exactly how the text pipeline's training script consumes these fields (later replacing `" ки"` with `"+"`/`"-"` by correctness). Auto-scoring: MC = exact letter match, CLOSED = normalized exact-text match, OPEN = left unscored (`"score": "None"`), pending a not-yet-built LLM-judge step. **This means multimodal step labels are not yet real correctness labels for OPEN questions (46% of the dataset) — see Section 7.**

**Full-run scale:** 11,286 questions × 64 samples/question × 2 models ≈ 1.44M generated solutions. Output sizes: Qwen3-VL-8B outputs total **6.1 GB**, HuatuoGPT-Vision-7B outputs total **3.1 GB** (`du -sh dataset_mm/generated_traces/*`).

---

## 4. Models & Training

### Text PRM (trained, completed)
- **Base model:** `meta-llama/Llama-3.1-8B-Instruct` (8,597,553,152 total parameters, per training log).
- **Method:** LoRA (via `peft`), rank 16, alpha 32, dropout 0.05, target modules = all attention + MLP projections (q/k/v/o, gate/up/down), plus a fully-trainable `embed_tokens` (for the new `" ки"` token). **567,283,712 / 8,597,553,152 parameters trainable (6.60%).**
- **Hyperparameters:** learning_rate 2e-6 (inherited from the full-fine-tune config — noted as unusually low for LoRA), per_device_train_batch_size 1, gradient_accumulation_steps 64 (effective batch 64), 3 epochs, cosine LR schedule, bf16, max_token_len 4096, risk_param (mu) 5.0, save_steps 200.
- **Training run (run_5, the one used downstream):** train_runtime **226,242s (~62.8h)**, train_samples_per_second 0.807, train_steps_per_second 0.013, final reported train_loss **15.51** (HF Trainer running average, pulled down by high early-epoch losses). Loss trajectory: ~72.8 (initial) → 44.1→25.3 (epoch 0.1–0.9) → 24.4→22.7 (epoch 1–2) → plateaus at 23.6→23.0 (epoch 2–3). **Getting a completed run took 5 SLURM attempts** (2 environment failures, 1 time-limit kill, 1 checkpoint-resume crash, then success).
- **Saved model:** `model_train/meta-llama/Llama-3.1-8B-Instruct-gemini_label-filter_yes-ep3-20260731_191535-RAG_yes/` (LoRA adapter merged into base weights).

### Multimodal trace generation (completed; not model training)
- **Models:** `Qwen/Qwen3-VL-8B-Instruct` (17GB weights) and `FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL` (16GB weights), both **frozen** — used only for inference/sampling, not fine-tuned.
- **Sampling config (both models, `scripts_mm/3_generate_traces_full_{qwen,huatuo}.sh`):** repeat_count 64, temperature 0.7, top_k 50, top_p 0.9, max_tokens 4096, max_model_len 12,000, max_images_per_prompt 6, max_pixels 1,310,720 (Qwen) / 1,003,520 (Huatuo) — the per-model pixel caps bound each model's own vision-tokenization to a shared ~1,280-tokens-per-image budget.
- **No multimodal PRM/ORM has been trained.** `python_mm/` contains only `0_preparing.py`, `1_normalize_data.py`, `2_download_models.py`, `3_generate_traces.py`, `measure_prompt_lengths.py` — no training script analogous to `python/2_training_lora.py` exists yet.

---

## 5. Results

All numbers below are quoted directly from the cited files; none are estimated or invented.

### Text PRM — raw scoring run (before benchmark correction)
From `logs/inference/run_5/TEST_..._shard{0,1}.log` (`LORA_TRAINING_AND_SCORING_CHANGES.txt` §7):

| | N | PRM Accuracy | Maj-Vote Accuracy |
|---|---|---|---|
| Shard 0 | 2,735 | 1,907/2,735 (**69.73%**) | 1,856/2,735 (67.86%) |
| Shard 1 | 2,734 | 2,136/2,734 (**78.13%**) | 2,098/2,734 (76.74%) |
| Combined | 5,469 | 4,043/5,469 (**73.94%**) | 3,954/5,469 (72.30%) |

Note: this combined figure blends 13 benchmarks of very different difficulty and is **not** directly comparable to the paper's headline MedQA-4 number (`REPRODUCTION_VERIFICATION.txt` explains the shard 0 vs. shard 1 gap is a difficulty-mix artifact of the file being grouped by `data_source`, not a scoring issue).

### Text PRM — corrected, protocol-matched comparison vs. the published paper
From `REPRODUCTION_VERIFICATION.txt`, isolating `data_source == med_qa_4op` (N=1,273):

| Strategy | This repro (LoRA) | Paper (Llama-3.1-8B-Instruct) | Diff |
|---|---|---|---|
| Self-Consistency (SC) | 75.18% | 74.86% | +0.32pp |
| Best-of-N (PRM Acc) | **76.98%** | **76.98%** | **0.00pp — exact match** |
| SC+RM hybrid | 77.69–77.93% | 78.24% | -0.3 to -0.6pp |

MedMCQA (N=500): SC 63.40% (exact match to paper), Best-of-N 65.60% vs. paper's 63.40% (+2.20pp), SC+RM hybrid 65.80% vs. paper's 66.40% (-0.60pp).

**Conclusion recorded in the repo:** the LoRA-trained reward model (training only 6.6% of parameters) reproduces the paper's Best-of-N number on MedQA-4 to two decimal places.

### Text PRM — full 7-benchmark report
From `dataset/dataset_5_benchmark_report/benchmark_report.md` (generated by `python/5_benchmark_report.py`):

| Benchmark | N | Best-of-N | SC | SC+RM (min_score) | SC+RM (final_score) |
|---|---|---|---|---|---|
| MedQA-4 | 1273 | 76.98% | 75.18% | 77.69% | 77.93% |
| MedQA-5 | 1273 | 73.37% | 70.7% | 72.82% | 72.82% |
| MedMCQA | 500 | 65.6% | 63.4% | 65.8% | 65.8% |
| MMLU-Med | 1089 | 82.55% | 81.63% | 82.74% | 82.64% |
| DDXPlus | 500 | 78.4% | 75.2% | 76.4% | 76.6% |
| AgentClinic (NEJM*) | 120 | 55.83% | 51.67% | 53.33% | 53.33% |
| AgentClinic (MedQA*) | 214 | 74.3% | 74.77% | 75.7% | 75.7% |
| **Average** | — | **72.43%** | **70.36%** | **72.07%** | **72.12%** |

Caveat noted directly in that file: "AgentClinic" columns use `nejm`/`osce` as stand-ins — there is no true `agent_clinic` data_source in this scored dataset, so treat those two rows as approximate.

### Multimodal pipeline — no accuracy/quality metrics yet
No PRM has been trained on the multimodal data, so there are **no reward-model accuracy numbers** for the multimodal side. What exists are pipeline-health measurements from `prm_docs/smoketest_analysis.txt`:
- Auto-scored MC/CLOSED accuracy exists per-solution in the raw output files (exact/normalized string match), but has not been aggregated into a benchmark table — **[UNVERIFIED]** whether anyone has computed a per-model accuracy roll-up from `dataset_mm/generated_traces/`.
- Known data-quality artifact: in one full-run file (`mm_benchmark_medxpertqa_mm_Qwen3-VL-8B-Instruct_64_batch0.json`), **217/16,000 (~1.4%) solutions have `answer: None`** (generation got stuck in a repetition loop, hit the 4,096-token cap) and are auto-scored as incorrect.
- Throughput: Qwen ~6.25 sec/question on VQA-RAD vs. Huatuo ~2.6 sec/question — Huatuo finished the full run **2.5–3x faster** despite needing ~30% more prompt/vision tokens per image, because Qwen generates roughly 2x more output text per response.

---

## 6. Infrastructure

- **Cluster:** Texas A&M HPRC (SLURM `gpu` partition, A100 GPUs, node examples g084/g086; compute nodes have **no internet access**, only login nodes do).
- **Conda environments** (`/scratch/user/arunimac02/.conda/envs/`, redirected from the default home-directory location by this system's `.condarc`):
  - `PRM` — text pipeline (training, RAG/Gemini labeling, scoring). `torch 2.6.0+cu124`, `transformers 4.46.3`, `trl 0.12.2`, `vllm 0.8.4`. Reconstructed as a snapshot-restore (`pip install --no-deps`) rather than a fresh dependency resolve, because `trl` and `vllm`'s pinned `transformers` ranges are mutually incompatible when resolved from scratch (documented in `ENVIRONMENT_SETUP_NOTES.txt`). Flash-attn 2.7.4 installed separately from a prebuilt wheel; noted as **non-functional on this system's T4 dev node** (Turing, compute capability 7.5 — FlashAttention 2 needs Ampere+), so training-related flash-attention code paths require an A100 SLURM job, not the interactive node.
  - `PRM_mm` — new, separate environment for multimodal inference only (unpinned `vllm==0.28.0`, `torch` built for CUDA 13.0). Kept separate because Qwen3-VL/Huatuo need newer `transformers`/`vllm` than the pinned `PRM` env supports. Training-only packages (`peft`, `bitsandbytes`, `deepspeed`, `wandb`, `trl`, `pymilvus`, `google-generativeai`) deliberately left out for now.
- **Pipelines built:**
  - Text: `0_preparing.py` → `1_train_dataset_RAG_judge_labeling.py` → `2_training.py` / `2_training_lora.py` → `3_test_dataset_sampling.py` → `4_scoring_PRM.py` / `4_scoring_lora_PRM.py` → `5_benchmark_report.py` (new), each with a matching `scripts/*.sh` launcher and `slurm_jobs/*.sh` SLURM wrapper.
  - Multimodal: `0_preparing.py` → `1_normalize_data.py` → `2_download_models.py` → `3_generate_traces.py`, plus a standalone diagnostic (`measure_prompt_lengths.py`) written specifically to measure true tokenized prompt lengths (via each model's own `AutoProcessor`, CPU-only) after a context-length crash — used to set `max_model_len`/`max_pixels` from real data instead of guesses.
- **Scoring parallelism:** `scripts/4_scoring_lora_PRM.sh` shards one model's test set across multiple GPUs (vs. the original one-model-per-GPU design) and merges results back in original order.
- **Checkpoint resume:** added to `2_training_lora.py` (`--resume_from_checkpoint`, step-based `save_steps` instead of epoch-based), which is what allowed run_5 to recover from run_3's time-limit kill instead of restarting from scratch.
- **Multimodal generation resumability:** `3_generate_traces.py` batches by `data_source` (`--batch_size`, default 250) and skips any batch whose output file already exists — a killed/pre-empted job resumes via plain resubmission, no separate state file. This is what let the Qwen job be cancelled and resubmitted mid-run without losing the 15 already-completed batches.

---

## 7. Open Issues

**Text pipeline:**
- LoRA learning_rate (2e-6) was inherited from the full-fine-tune config rather than retuned for LoRA (LoRA papers typically use 1e-4–3e-4); loss plateaus after epoch 1, an open question flagged in `LORA_TRAINING_AND_SCORING_CHANGES.txt` about whether 3 epochs / this LR is the right stopping point.
- Environment reproducibility is fragile: `environment.yml` is a frozen `conda env export` snapshot with hard-conflicting pins (`vllm` needs `transformers>=4.51.1`, `trl` needs `transformers<4.47.0`, file pins `4.46.3`) — a fresh `conda env create` will fail; the working install requires the two-stage snapshot-restore procedure documented in `ENVIRONMENT_SETUP_NOTES.txt`.

**Multimodal pipeline:**
- **Full-scale run required 3 rounds of bug-driven fixes**, discovered only at scale despite 5 smoke tests: (1) `max_model_len` wrongly conflated with `max_tokens`, causing Huatuo's first full-run attempt (job 19658750) to crash 33m49s in on a 4,210-token prompt exceeding the 4,096 cap; (2) a hardcoded 4-image-per-prompt limit that would have rejected 102 real questions (up to 6 images each); (3) unbounded per-image tokenization (a single high-resolution image could cost 9,000–11,000+ tokens) — fixed via `max_pixels` capping, not just raising `max_model_len`.
- Qwen's first full-run attempt (job 19658748) was **cancelled deliberately** after the Huatuo crash revealed the shared design flaw, rather than let it run to a near-certain future failure; 15/47 batches (~3,748 questions) it had already completed were generated **without** the `max_pixels` cap and were kept as-is (accepted inconsistency vs. re-running ~6h of compute).
- **~1.4% of Qwen's MedXpertQA-MM solutions have `answer: None`** (repetition-loop, hit token cap) — a known, quantified artifact, not fixed (would need `repetition_penalty` added to `SamplingParams`), left in the dataset as auto-scored-incorrect.
- Model output-format brittleness: both VLMs sometimes don't follow the requested `## Step N:` / `## Final Answer:` structure exactly (Huatuo more often than Qwen) — required two rounds of extraction-logic fallbacks (`extract_qa_answer`, `extract_mc_answer`) to avoid mislabeling genuinely-correct terse answers as unscorable.
- Early full-dataset memory sizing was wrong twice before landing on the right number: a smoke-test-scale extrapolation (~4.2GB) and a 30-image sample estimate (~11GB for SLAKE) were both superseded by an exact full-dataset measurement (~18.6GB unchunked), which is what drove the per-`data_source` chunking design rather than just requesting more `--mem`.
- An early throughput estimate for the full run was off by ~5x (extrapolated ~96h vs. actual ~20h for Qwen) because it used small-batch per-request latency without accounting for vLLM's continuous batching at scale — flagged in the notes as a methodology lesson.

---

## 8. What's Not Done Yet

1. **OPEN-answer correctness labeling.** 5,201 OPEN questions (46% of the dataset) × 64 samples × 2 models = **665,728 solutions** currently have `"score": "None"` and cannot be used as PRM/ORM training labels until judged. Explicitly not started — deferred pending a decision on Gemini API key/quota, whether to judge every solution vs. dedupe identical (question_id, extracted-answer) pairs, and whether compute-node offline-access constraints require running this step from a login node or with a local judge model instead (`prm_docs/setup.txt`, "NEXT STEP SCOPE NOTE").
2. **Multimodal PRM/ORM training itself.** No script exists yet (`python_mm/` has no `4_training_*.py` analog). Per the setup notes, this will require adding `peft`, `bitsandbytes`, `deepspeed`, `wandb`, `trl` into (or alongside) the `PRM_mm` environment and re-verifying version compatibility, since the current `PRM_mm` env was deliberately built inference-only.
3. **Multimodal scoring/benchmark pipeline.** No equivalent of `4_scoring_PRM.py` / `5_benchmark_report.py` exists for the multimodal outputs yet — no aggregated multimodal PRM Accuracy / Maj-Vote Accuracy numbers exist because there is no multimodal PRM.
4. **Consolidation of the 15 pre-`max_pixels`-fix Qwen batches** — an accepted but unresolved inconsistency in the current dataset (Section 7) that would need addressing before treating the full trace set as uniformly generated.
5. **AgentClinic benchmark gap** in the text-pipeline benchmark report — the scored dataset has no true `agent_clinic` data_source; the current report substitutes `nejm`/`osce` as approximate stand-ins, flagged with `*` (`python/5_benchmark_report.py`).
6. Full fine-tuning (non-LoRA) of the text PRM (`python/2_training.py`) has not been run to completion in this environment — only the LoRA path has produced a trained, scored model.

---

*Sources cited throughout: `README.md`, `ENVIRONMENT_SETUP_NOTES.txt`, `LORA_TRAINING_AND_SCORING_CHANGES.txt`, `REPRODUCTION_VERIFICATION.txt`, `prm_docs/setup.txt`, `prm_docs/smoketest_analysis.txt`, `dataset/dataset_5_benchmark_report/benchmark_report.md`, training/inference logs under `logs/` and `logs_mm/`, and direct inspection of `dataset_mm/normalized/mm_benchmark.json`, `dataset/dataset_4_scored_dataset/*.json`, and `model_train/*/config.json`.*
