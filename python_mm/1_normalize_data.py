#!/usr/bin/env python
# coding: utf-8
"""
Normalize VQA-RAD, SLAKE, and MedXpertQA-MM (downloaded by 0_preparing.py)
into one common schema, so a later reasoning-trace-generation step can
consume all three uniformly -- mirroring how python/3_test_dataset_sampling.py
consumes a single JSON file with a `data_source` column across multiple
text-only benchmarks (med_qa, medmc_qa, pubmed_qa, ...).

Common schema per row:
    question_id      - unique across all three benchmarks, e.g. "vqa_rad_test_42"
    data_source       - "vqa_rad" | "slake" | "medxpertqa_mm"
    split             - the original split name (train/validation/dev/test/unmatched)
    question          - question text
    options           - {"A": "...", ...} for MedXpertQA, {} otherwise
    correct_answer    - answer text (VQA-RAD/SLAKE) or correct letter (MedXpertQA)
    answer_type       - unified 3-way tag: "MC" | "CLOSED" | "OPEN"
    images            - list of image file paths (always a list, even if length 1)
    modality          - imaging modality when the source states it directly, else None
    source_metadata   - original fields not already promoted to a top-level key,
                        kept for traceability back to the raw source

Images are referenced by path, not re-encoded or copied: all three raw
downloads already extracted real image files to disk (dataset_mm/*/imgs or
.../images), so the manifest just points at those existing files.
"""
import json
import os

from datasets import load_from_disk

BASE_DIR = "dataset_mm"
OUTPUT_DIR = os.path.join(BASE_DIR, "normalized")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "mm_benchmark.json")


def normalize_vqa_rad():
    ds = load_from_disk(os.path.join(BASE_DIR, "vqa_rad"))
    rows = []
    for split_name, split_data in ds.items():
        for i, r in enumerate(split_data):
            image_path = os.path.join(BASE_DIR, "vqa_rad", "imgs", r["image_name"])
            rows.append({
                "question_id": f"vqa_rad_{split_name}_{r['qid']}",
                "data_source": "vqa_rad",
                "split": split_name,
                "question": r["question"],
                "options": {},
                "correct_answer": r["answer"],
                "answer_type": r["answer_type"].strip().upper(),
                "images": [image_path],
                "modality": None,
                "source_metadata": {
                    "qid": r["qid"],
                    "phrase_type": r["phrase_type"],
                    "question_type": r["question_type"],
                    "image_organ": r["image_organ"],
                    "image_name": r["image_name"],
                    "image_case_url": r["image_case_url"],
                },
            })
    return rows


def normalize_slake():
    ds = load_from_disk(os.path.join(BASE_DIR, "slake"))
    rows = []
    for split_name, split_data in ds.items():
        for i, r in enumerate(split_data):
            image_path = os.path.join(BASE_DIR, "slake", "imgs", "imgs", r["img_name"])
            rows.append({
                "question_id": f"slake_{split_name}_{r['qid']}",
                "data_source": "slake",
                "split": split_name,
                "question": r["question"],
                "options": {},
                "correct_answer": r["answer"],
                "answer_type": r["answer_type"].strip().upper(),
                "images": [image_path],
                "modality": r["modality"],
                "source_metadata": {
                    "qid": r["qid"],
                    "img_id": r["img_id"],
                    "img_name": r["img_name"],
                    "location": r["location"],
                    "base_type": r["base_type"],
                    "content_type": r["content_type"],
                    "triple": r["triple"],
                },
            })
    return rows


def normalize_medxpertqa_mm():
    # Read the raw JSONL directly (same files 0_preparing.py extracted)
    # instead of going through the Arrow dataset: once `images` is cast
    # to a decoded PIL Image list, the original filenames are gone (PIL's
    # `.filename` doesn't survive the `.load()` call `datasets` applies
    # right after opening), so the raw pre-cast filenames are the only
    # reliable source of the image paths.
    raw_dir = os.path.join(BASE_DIR, "medxpertqa_mm", "_raw", "MM")
    imgs_dir = os.path.join(BASE_DIR, "medxpertqa_mm", "images", "images")
    split_files = {"dev": "dev.jsonl", "test": "test.jsonl"}

    rows = []
    for split_name, fname in split_files.items():
        with open(os.path.join(raw_dir, fname), encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                image_paths = [os.path.join(imgs_dir, fn) for fn in r["images"]]
                rows.append({
                    "question_id": f"medxpertqa_mm_{split_name}_{r['id']}",
                    "data_source": "medxpertqa_mm",
                    "split": split_name,
                    "question": r["question"],
                    "options": r["options"],
                    "correct_answer": r["label"],
                    "answer_type": "MC",
                    "images": image_paths,
                    "modality": None,
                    "source_metadata": {
                        "id": r["id"],
                        "medical_task": r["medical_task"],
                        "body_system": r["body_system"],
                        "question_type": r["question_type"],
                    },
                })
    return rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_rows = []
    all_rows += normalize_vqa_rad()
    print(f"VQA-RAD: {len([r for r in all_rows if r['data_source'] == 'vqa_rad'])} rows")

    all_rows += normalize_slake()
    n_slake = len([r for r in all_rows if r["data_source"] == "slake"])
    print(f"SLAKE: {n_slake} rows")

    all_rows += normalize_medxpertqa_mm()
    n_mx = len([r for r in all_rows if r["data_source"] == "medxpertqa_mm"])
    print(f"MedXpertQA-MM: {n_mx} rows")

    # Renumber question_id as a single global sequence (1..N), independent
    # of data_source -- traceability back to the source is preserved via
    # the `data_source`/`split`/`source_metadata` fields on each row.
    for i, r in enumerate(all_rows, start=1):
        r["question_id"] = i

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)

    print(f"\nTotal: {len(all_rows)} rows -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
