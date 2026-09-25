#!/usr/bin/env python
# coding: utf-8
"""
vLLM-based multimodal chain-of-thought trace generation.

Multimodal analog of python/3_test_dataset_sampling.py: reads
dataset_mm/normalized/mm_benchmark.json, attaches each question's image(s),
and samples step-by-step reasoning traces from one vision-language model
per run (invoke once per model, same pattern as the text pipeline).

Local images are passed to vllm's LLM.chat() as PIL images via the
"image_pil" content-part shorthand (vllm.entrypoints.chat_utils
.CustomChatCompletionContentPILImageParam) rather than base64 data URIs --
confirmed by inspecting that class before writing this script.

Answer-type handling:
  - MC     (medxpertqa_mm): choices are already embedded in the question
    text, so the prompt asks for "the answer is (letter)"; auto-scored by
    exact letter match against `correct_answer`.
  - CLOSED (vqa_rad/slake short factual answers): prompt asks for
    "Final Answer: ..."; auto-scored by normalized-text exact match.
  - OPEN   (vqa_rad/slake free-form answers): same prompt as CLOSED but
    left unscored (score="None") -- correctness needs an LLM judge, added
    in a later step.
"""
import argparse
import gc
import json
import math
import os
import re
from collections import defaultdict

from PIL import Image
from vllm import LLM, SamplingParams

# ──────────────────────────────────────────────────────────────
# 0. Prompts
# ──────────────────────────────────────────────────────────────
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

STEP_PATTERN = r"(?:## )?Step \d+:"


# ──────────────────────────────────────────────────────────────
# 1. Argument parser
# ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="vLLM multimodal CoT trace generation")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--gpu_id", type=str, required=True)
    p.add_argument("--input_file", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--repeat_count", type=int, required=True)
    p.add_argument("--temperature", type=float, required=True)
    p.add_argument("--top_k", type=int, required=True)
    p.add_argument("--top_p", type=float, required=True)
    p.add_argument("--max_tokens", type=int, required=True)
    p.add_argument("--max_model_len", type=int, required=True,
                    help="Total context window (prompt text + image tokens + "
                         "generation), NOT the same as --max_tokens (generation-only "
                         "budget). Measured via python_mm/measure_prompt_lengths.py: "
                         "with --max_pixels capping applied, both models' true max "
                         "prompt length across the full dataset is 7628 tokens, so "
                         "this must be >= 7628 + max_tokens.")
    p.add_argument("--max_pixels", type=int, required=True,
                    help="Caps each image's resolution before it's tiled into "
                         "vision-tokens (vllm mm_processor_kwargs), so a single "
                         "unusually high-resolution image can't blow up the prompt "
                         "length on its own. Value is model-specific (depends on "
                         "that model's own patch/merge geometry) -- see "
                         "python_mm/measure_prompt_lengths.py for how it's derived.")
    p.add_argument("--max_images_per_prompt", type=int, default=6,
                    help="102 medxpertqa_mm questions have 5-6 images; default matches "
                         "the true max found in the dataset (checked directly, not "
                         "assumed) so none get rejected by vllm.")
    p.add_argument("--data_source_list", type=str, default=None,
                    help="Comma-separated data sources to process; all if omitted")
    p.add_argument("--limit_per_source", type=int, default=None,
                    help="Cap questions per data_source (smoke testing)")
    p.add_argument("--batch_size", type=int, default=250,
                    help="Questions per batch within a data_source. Each batch is "
                         "saved to its own file and skipped on rerun if that file "
                         "already exists, so a crashed/killed job can be resumed "
                         "by resubmitting the same command.")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────
# 2. Preprocessing
# ──────────────────────────────────────────────────────────────
def format_question(qdata):
    answer_type = qdata["answer_type"]
    gt = qdata["correct_answer"].strip()
    if answer_type == "MC":
        gt = gt.upper()
    return {
        "question_id": qdata["question_id"],
        "data_source": qdata["data_source"],
        "answer_type": answer_type,
        "question": qdata["question"].strip(),
        "images": qdata["images"],
        "ground_truth_for_eval": gt,
    }


def load_images(paths):
    return [Image.open(p).convert("RGB") for p in paths]


# ──────────────────────────────────────────────────────────────
# 3. Response post-processing
# ──────────────────────────────────────────────────────────────
def extract_steps_from_text(txt: str):
    mts = list(re.finditer(STEP_PATTERN, txt))
    if not mts:
        return [txt.strip()] if txt.strip() else []
    steps = []
    for i, m in enumerate(mts):
        start = m.start()
        end = mts[i + 1].start() if i + 1 < len(mts) else len(txt)
        steps.append(txt[start:end].strip().replace("## ", ""))
    return steps


def extract_mc_answer(txt: str):
    low = txt.lower()
    m = re.findall(r"\\boxed\{\(?([a-z])\)?\}", low)
    if m:
        return m[-1].upper()
    m_list = list(re.finditer(r"(?:answer is|the answer is|final answer is)\s*:?\s*\(?([a-z])\)?", low))
    if m_list:
        return m_list[-1].group(1).upper()
    # Fallback: some models state their choice without the exact "answer
    # is" phrasing (e.g. "...is an abdominal ultrasound (D)." or "Answer:
    # (D) Biopsy of mass"). Take the last standalone parenthesized letter
    # in the text -- models consistently state their actual final choice
    # last, after narrating through/eliminating other options earlier.
    paren_list = list(re.finditer(r"\(([a-z])\)", low))
    if paren_list:
        return paren_list[-1].group(1).upper()
    return None


def extract_qa_answer(txt: str):
    m = re.search(r"(?:## )?Final Answer:\s*(.*?)(?:\n|$)", txt, flags=re.I)
    if m:
        return m.group(1).strip()
    # Fallback: some models skip the requested step-by-step format entirely
    # and just answer directly (e.g. a bare "Yes"). Only treat the whole
    # response as the answer when it has no step structure and is short --
    # a long, unstructured response with no "Final Answer:" is more likely
    # a truncated multi-step generation than a terse direct answer, and
    # guessing an answer for that case would be worse than leaving it blank.
    stripped = txt.strip()
    if stripped and len(extract_steps_from_text(txt)) <= 1 and len(stripped) <= 100:
        return stripped
    return ""


def normalize_text(s):
    return re.sub(r"\s+", " ", str(s) if s is not None else "").strip().lower()


def prm_process_solution(txt: str):
    no_nl = txt.replace("\n", " ")
    mts = list(re.finditer(STEP_PATTERN, no_nl))
    if not mts:
        return no_nl.strip() + " ки" if no_nl.strip() else ""
    head = no_nl[:mts[0].start()].strip()
    steps = []
    for i, m in enumerate(mts):
        start = m.start()
        end = mts[i + 1].start() if i + 1 < len(mts) else len(no_nl)
        steps.append(no_nl[start:end].strip().replace("## ", "") + " ки")
    if head:
        steps.insert(0, head)
    return " ".join(steps)


def orm_process_solution(txt: str):
    return txt.replace("\n", " ") + " ки"


# ──────────────────────────────────────────────────────────────
# 4. JSON I/O
# ──────────────────────────────────────────────────────────────
def load_json(path):
    return json.load(open(path, encoding="utf-8"))


def save_json(obj, path):
    json.dump(obj, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────────────────────
# 5. Prompt collection + vLLM execution
# ──────────────────────────────────────────────────────────────
def build_conversation(q):
    sys_prompt = SYSTEM_PROMPT_MC if q["answer_type"] == "MC" else SYSTEM_PROMPT_QA
    content = [{"image_pil": im} for im in load_images(q["images"])]
    content.append({"type": "text", "text": q["question"]})
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": content},
    ]


def collect_prompts(qdatas, n_samples):
    convs, meta = [], []
    for q in qdatas:
        conv = build_conversation(q)
        for _ in range(n_samples):
            convs.append(conv)
            meta.append({
                "question_id": q["question_id"],
                "ground_truth": q["ground_truth_for_eval"],
                "answer_type": q["answer_type"],
            })
    return convs, meta


def llm_chat(llm, samp_params, convs, meta):
    outs = llm.chat(convs, samp_params)
    bucket = defaultdict(lambda: {"generated_texts": []})
    for i, o in enumerate(outs):
        qid = meta[i]["question_id"]
        bucket[qid]["generated_texts"].append(o.outputs[0].text)
    return bucket


# ──────────────────────────────────────────────────────────────
# 6. Main pipeline
# ──────────────────────────────────────────────────────────────
def generate_all(qdatas, repeat_cnt, llm, samp_params):
    first_cnt = math.ceil(repeat_cnt * 1.25)

    convs1, meta1 = collect_prompts(qdatas, first_cnt)
    res1 = llm_chat(llm, samp_params, convs1, meta1)

    valid = defaultdict(list)
    for q in qdatas:
        qid = q["question_id"]
        for txt in res1[qid]["generated_texts"]:
            if 0 < len(extract_steps_from_text(txt)) < 10:
                valid[qid].append(txt)

    need2 = [q for q in qdatas if len(valid[q["question_id"]]) < repeat_cnt]
    if need2:
        convs2, meta2 = [], []
        for q in need2:
            lack = (repeat_cnt - len(valid[q["question_id"]])) * 3
            conv = build_conversation(q)
            for _ in range(lack):
                convs2.append(conv)
                meta2.append({
                    "question_id": q["question_id"],
                    "ground_truth": q["ground_truth_for_eval"],
                    "answer_type": q["answer_type"],
                })
        res2 = llm_chat(llm, samp_params, convs2, meta2)
        for q in need2:
            qid = q["question_id"]
            for txt in res2[qid]["generated_texts"]:
                if 0 < len(extract_steps_from_text(txt)) < 10:
                    valid[qid].append(txt)

    outputs = []
    for q in qdatas:
        qid = q["question_id"]
        answer_type = q["answer_type"]
        gt = q["ground_truth_for_eval"]
        sols = []
        for txt in valid[qid][:repeat_cnt]:
            if answer_type == "MC":
                pred = extract_mc_answer(txt)
                score = int(pred == gt) if pred else 0
            elif answer_type == "CLOSED":
                pred = extract_qa_answer(txt)
                score = int(normalize_text(pred) == normalize_text(gt)) if pred else 0
            else:  # OPEN
                pred = extract_qa_answer(txt)
                score = "None"
            sols.append({
                "solution": txt,
                "prm_processed_solution": prm_process_solution(txt),
                "orm_processed_solution": orm_process_solution(txt),
                "answer": pred,
                "score": score,
            })
        outputs.append({
            "question_id": qid,
            "data_source": q["data_source"],
            "answer_type": answer_type,
            "question": q["question"],
            "images": q["images"],
            "correct_answer": gt,
            "solutions": sols,
        })
    return outputs


# ──────────────────────────────────────────────────────────────
# 7. Main
# ──────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    llm = LLM(
        model=args.model_path,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": args.max_images_per_prompt},
        mm_processor_kwargs={"max_pixels": args.max_pixels},
    )
    samp = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    raw = load_json(args.input_file)
    print(f"Loaded {len(raw)} questions")

    if args.data_source_list:
        keep = {ds.strip() for ds in args.data_source_list.split(",")}
        raw = [q for q in raw if q["data_source"] in keep]
        print(f"Filtered to {len(raw)} questions (data sources: {', '.join(keep)})")

    if args.limit_per_source:
        by_source = defaultdict(list)
        for q in raw:
            by_source[q["data_source"]].append(q)
        raw = [q for src in by_source.values() for q in src[: args.limit_per_source]]
        print(f"Capped to {len(raw)} questions ({args.limit_per_source} per data source)")

    # --- Original single-batch approach: builds every question's
    # --- conversations (and decodes every image via PIL) for the WHOLE
    # --- input in one collect_prompts() call before any generation starts.
    # --- Fine at smoke-test scale, but at full scale (~12,139 images
    # --- across the dataset, ~1.53MB each decoded) this held ~18.6GB of
    # --- decoded images in host RAM simultaneously. Replaced below with a
    # --- per-data_source chunked version that loads the model once but
    # --- processes/frees one data_source's images at a time, and writes
    # --- one output file per data_source instead of one combined file.
    #
    # transformed = [format_question(q) for q in raw]
    # dataset = generate_all(transformed, args.repeat_count, llm, samp)
    #
    # os.makedirs(args.output_dir, exist_ok=True)
    # model_name = os.path.basename(args.model_path.rstrip("/"))
    # base = os.path.splitext(os.path.basename(args.input_file))[0]
    # out_path = os.path.join(args.output_dir, f"{base}_{model_name}_{args.repeat_count}.json")
    # save_json(dataset, out_path)
    # print(f"Done -> {out_path}")

    os.makedirs(args.output_dir, exist_ok=True)
    model_name = os.path.basename(args.model_path.rstrip("/"))
    base = os.path.splitext(os.path.basename(args.input_file))[0]

    by_source = defaultdict(list)
    for q in raw:
        by_source[q["data_source"]].append(q)

    # Batched within each data_source (not just chunked by source) so a
    # crashed/killed job -- expected to be a real risk given full-scale runs
    # here take multiple days, right up against this cluster's 4-day GPU
    # partition time limit -- only loses at most one batch's worth of
    # progress, not a whole data_source (SLAKE alone is ~7,033 questions and
    # could itself run for many hours). Resuming is just resubmitting the
    # same command: each batch's output file existing on disk means it's
    # skipped, so already-completed work is never redone.
    for data_source, group in by_source.items():
        n_batches = math.ceil(len(group) / args.batch_size)
        print(f"--- Processing data_source={data_source} ({len(group)} questions, "
              f"{n_batches} batches of up to {args.batch_size}) ---")
        for batch_idx in range(n_batches):
            batch = group[batch_idx * args.batch_size : (batch_idx + 1) * args.batch_size]
            out_path = os.path.join(
                args.output_dir,
                f"{base}_{data_source}_{model_name}_{args.repeat_count}_batch{batch_idx}.json",
            )
            if os.path.exists(out_path):
                print(f"Skipping {data_source} batch {batch_idx}/{n_batches - 1} "
                      f"(already exists: {out_path})")
                continue

            print(f"Processing {data_source} batch {batch_idx}/{n_batches - 1} "
                  f"({len(batch)} questions)")
            transformed = [format_question(q) for q in batch]
            dataset = generate_all(transformed, args.repeat_count, llm, samp)
            save_json(dataset, out_path)
            print(f"Done: {data_source} batch {batch_idx}/{n_batches - 1} -> {out_path}")

            del transformed, dataset
            gc.collect()


if __name__ == "__main__":
    main()
