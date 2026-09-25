#!/usr/bin/env python
# coding: utf-8
"""
Download the three multimodal medical VQA benchmarks used for the
multimodal Med-PRM track: VQA-RAD, SLAKE (official, bilingual), and the
image (MM) subset of MedXpertQA.

Mirrors the download+inspect pattern of python/0_preparing.py.

All three benchmarks are handled via a raw snapshot+join, because in
every case the convenient Hugging Face mirror turned out to drop
metadata (answer_type, question_type, etc.) that the official release
has and that later pipeline steps (correctness scoring) need:

- VQA-RAD: not on Hugging Face at all in its full form -- every HF
  mirror (flaviagiammarino/vqa-rad, etc.) strips it down to just
  image/question/answer. The official release (Lau et al. 2018) is
  hosted on OSF (osf.io/89kps) and has answer_type (CLOSED/OPEN),
  question_type (11 categories), image_organ, etc., but no train/test
  split field -- it's a single pool of 2248 questions. We fetch the
  official JSON + image folder directly from OSF, then reconcile the
  standard 1793/451 train/test split (which only exists in HF mirrors)
  by matching each question back to flaviagiammarino/vqa-rad via an
  exact image-pixel hash (verified byte-identical to the OSF source)
  plus normalized question/answer text.
- SLAKE: the official BoKelvin/SLAKE repo only ships question metadata
  as JSON (bilingual, image referenced by a relative path like
  "xmlab1/source.jpg") plus a separate imgs.zip -- `load_dataset` alone
  can't produce usable images for it, so we snapshot the raw repo,
  extract the images, keep only the English-language rows, and join
  each row to its image file ourselves. We use this official version
  instead of a pre-merged English-only mirror because it keeps
  `answer_type` (OPEN/CLOSED), `content_type`, and `modality`, which the
  mirror drops -- `answer_type` in particular determines which
  correctness-scoring path (exact-match vs LLM-judge) a question needs
  later.
- MedXpertQA: `load_dataset("TsinghuaC3I/MedXpertQA", name="MM")` fails
  under this project's pinned `datasets==3.6.0` (its Hub metadata uses a
  `List` feature type that version doesn't recognize). The repo itself
  is just raw JSONL (`MM/dev.jsonl`, `MM/test.jsonl` -- `Text/*.jsonl`
  skipped on purpose since we don't need the text-only questions) plus
  a single `images.zip`, so we snapshot only the MM files, extract the
  images, and join each row's `images` list to the extracted files
  ourselves -- same approach as SLAKE, and it avoids touching the
  shared environment's `datasets` version.
"""
import hashlib
import json
import os
import re
import shutil
import urllib.request
import zipfile

import numpy as np
import PIL.Image
from datasets import Dataset, DatasetDict, Image, Sequence, load_dataset, load_from_disk
from huggingface_hub import snapshot_download

VQA_RAD_JSON_URL = "https://osf.io/download/6qdas/"
VQA_RAD_IMAGES_ZIP_URL = (
    "https://files.osf.io/v1/resources/89kps/providers/osfstorage/5b21453986d8510011c277bc/?zip="
)


def _download_file(url: str, dest_path: str):
    if os.path.exists(dest_path):
        return
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    tmp_path = dest_path + ".part"
    urllib.request.urlretrieve(url, tmp_path)
    os.rename(tmp_path, dest_path)


def _normalize_text(s):
    # Some VQA-RAD answers are numeric (e.g. counts) rather than strings.
    return re.sub(r"\s+", " ", str(s) if s is not None else "").strip().lower()


def _image_pixel_hash(im):
    arr = np.array(im.convert("RGB"))
    return hashlib.md5(arr.tobytes()).hexdigest()


def download_and_inspect(repo_id: str, local_dir: str, config_name: str = None):
    label = f"{repo_id!r}" + (f" (config={config_name!r})" if config_name else "")
    print(f"\n▶ Downloading {label} into {local_dir!r} …")
    ds = load_dataset(repo_id, name=config_name)

    os.makedirs(local_dir, exist_ok=True)
    ds.save_to_disk(local_dir)

    for split_name, split_data in ds.items():
        print(f"  • Split {split_name!r}: {len(split_data)} rows")
        print(f"    - Columns: {split_data.column_names}")
        if len(split_data) > 0:
            print(f"    - First record keys: {list(split_data[0].keys())}")

    return ds


def download_and_prepare_vqa_rad(local_dir: str, mirror_dir: str):
    print(f"\n▶ Downloading official VQA-RAD release (OSF) into {local_dir!r} …")
    raw_dir = os.path.join(local_dir, "_raw")
    _download_file(VQA_RAD_JSON_URL, os.path.join(raw_dir, "VQA_RAD_Dataset_Public.json"))
    _download_file(VQA_RAD_IMAGES_ZIP_URL, os.path.join(raw_dir, "VQA_RAD_Image_Folder.zip"))

    imgs_dir = os.path.join(local_dir, "imgs")
    if not os.path.isdir(imgs_dir):
        print(f"  • Extracting image folder -> {imgs_dir!r}")
        with zipfile.ZipFile(os.path.join(raw_dir, "VQA_RAD_Image_Folder.zip")) as zf:
            zf.extractall(imgs_dir)

    with open(os.path.join(raw_dir, "VQA_RAD_Dataset_Public.json"), encoding="utf-8-sig") as f:
        rows = json.load(f)

    # The official release has no train/test split field. Reconcile the
    # standard split by matching each row back to the already-downloaded
    # flaviagiammarino/vqa-rad mirror via (question, answer, image pixel
    # hash) -- verified byte-identical image content between sources.
    print("  • Reconciling train/test split against the flaviagiammarino/vqa-rad mirror …")
    mirror = load_from_disk(mirror_dir)
    split_lookup = {}
    for split_name, split_data in mirror.items():
        for row in split_data:
            key = (
                _normalize_text(row["question"]),
                _normalize_text(row["answer"]),
                _image_pixel_hash(row["image"]),
            )
            split_lookup.setdefault(key, split_name)

    unmatched = 0
    for r in rows:
        # qid/answer are inconsistently typed (str vs int) across rows in
        # the raw JSON, which pyarrow can't build a single column from.
        r["qid"] = str(r["qid"])
        r["answer"] = str(r["answer"])
        r["answer_type"] = (r.get("answer_type") or "").strip()
        img_path = os.path.join(imgs_dir, r["image_name"])
        r["image"] = img_path
        with PIL.Image.open(img_path) as im:
            key = (_normalize_text(r["question"]), _normalize_text(r["answer"]), _image_pixel_hash(im))
        split_name = split_lookup.get(key)
        if split_name is None:
            unmatched += 1
        r["split"] = split_name or "unmatched"

    print(f"  • {unmatched}/{len(rows)} rows had no match in the mirror (kept under split='unmatched')")

    # Wipe any stale split folders from a previous run before writing.
    for stale in ("train", "test", "unmatched", "dataset_dict.json"):
        stale_path = os.path.join(local_dir, stale)
        if os.path.isdir(stale_path):
            shutil.rmtree(stale_path)
        elif os.path.isfile(stale_path):
            os.remove(stale_path)

    by_split = {name: [r for r in rows if r["split"] == name] for name in ("train", "test", "unmatched")}

    splits = {}
    for split_name, split_rows in by_split.items():
        if not split_rows:
            continue
        for r in split_rows:
            del r["split"]
        ds = Dataset.from_list(split_rows).cast_column("image", Image())
        splits[split_name] = ds
        print(f"  • Split {split_name!r}: {len(ds)} rows")
        print(f"    - Columns: {ds.column_names}")

    dataset = DatasetDict(splits)
    dataset.save_to_disk(local_dir)
    return dataset


def download_and_prepare_slake(local_dir: str):
    repo_id = "BoKelvin/SLAKE"
    print(f"\n▶ Downloading raw files for {repo_id!r} into {local_dir!r} …")
    raw_dir = os.path.join(local_dir, "_raw")
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=raw_dir)

    imgs_dir = os.path.join(local_dir, "imgs")
    if not os.path.isdir(imgs_dir):
        print(f"  • Extracting imgs.zip -> {imgs_dir!r}")
        with zipfile.ZipFile(os.path.join(raw_dir, "imgs.zip")) as zf:
            zf.extractall(imgs_dir)

    split_files = {
        "train": "train.json",
        "validation": "validation.json",
        "test": "test.json",
    }

    splits = {}
    for split_name, fname in split_files.items():
        with open(os.path.join(raw_dir, fname), encoding="utf-8") as f:
            rows = json.load(f)

        # Bilingual dataset (English + Chinese) -- keep English rows only.
        rows = [r for r in rows if r.get("q_lang") == "en"]

        for r in rows:
            # imgs.zip contains its own top-level "imgs/" folder.
            r["image"] = os.path.join(imgs_dir, "imgs", r["img_name"])

        ds = Dataset.from_list(rows).cast_column("image", Image())
        splits[split_name] = ds
        print(f"  • Split {split_name!r}: {len(ds)} English-language rows")
        print(f"    - Columns: {ds.column_names}")
        if len(ds) > 0:
            print(f"    - First record keys: {list(ds[0].keys())}")

    dataset = DatasetDict(splits)
    dataset.save_to_disk(local_dir)
    return dataset


def download_and_prepare_medxpertqa_mm(local_dir: str):
    repo_id = "TsinghuaC3I/MedXpertQA"
    print(f"\n▶ Downloading raw MM files for {repo_id!r} into {local_dir!r} …")
    raw_dir = os.path.join(local_dir, "_raw")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=raw_dir,
        allow_patterns=["MM/*", "images.zip", "README.md", ".gitattributes"],
    )

    imgs_dir = os.path.join(local_dir, "images")
    if not os.path.isdir(imgs_dir):
        print(f"  • Extracting images.zip -> {imgs_dir!r}")
        with zipfile.ZipFile(os.path.join(raw_dir, "images.zip")) as zf:
            zf.extractall(imgs_dir)

    split_files = {
        "dev": "MM/dev.jsonl",
        "test": "MM/test.jsonl",
    }

    splits = {}
    for split_name, fname in split_files.items():
        with open(os.path.join(raw_dir, fname), encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]

        for r in rows:
            # images.zip contains its own top-level "images/" folder.
            r["images"] = [os.path.join(imgs_dir, "images", fn) for fn in r["images"]]

        ds = Dataset.from_list(rows).cast_column("images", Sequence(feature=Image()))
        splits[split_name] = ds
        print(f"  • Split {split_name!r}: {len(ds)} rows")
        print(f"    - Columns: {ds.column_names}")
        if len(ds) > 0:
            print(f"    - First record keys: {list(ds[0].keys())}")

    dataset = DatasetDict(splits)
    dataset.save_to_disk(local_dir)
    return dataset


if __name__ == "__main__":
    base_dir = "dataset_mm"

    # 1) VQA-RAD -- official OSF release, train/test split reconciled
    #    against the flaviagiammarino/vqa-rad mirror.
    #    Columns: qid, phrase_type, qid_linked_id, image_case_url,
    #    image_name, image_organ, evaluation, question, question_rephrase,
    #    question_relation, question_frame, question_type, answer,
    #    answer_type (CLOSED/OPEN), image.
    download_and_inspect(
        repo_id="flaviagiammarino/vqa-rad",
        local_dir=os.path.join(base_dir, "vqa_rad_split_reference"),
    )
    download_and_prepare_vqa_rad(
        local_dir=os.path.join(base_dir, "vqa_rad"),
        mirror_dir=os.path.join(base_dir, "vqa_rad_split_reference"),
    )

    # 2) SLAKE (official, English rows only after filtering) --
    #    Splits: train, validation, test (~4920/1050/1060 English rows).
    #    Columns: img_name, img_id, question, answer, q_lang, answer_type
    #    (OPEN/CLOSED), content_type, modality, location, base_type, triple,
    #    qid, image.
    download_and_prepare_slake(
        local_dir=os.path.join(base_dir, "slake"),
    )

    # 3) MedXpertQA -- MM (image) subset only, Text/*.jsonl skipped on purpose.
    #    Splits: dev (5), test (2000).
    #    Columns: id, question, options, label, images, medical_task, body_system, question_type.
    download_and_prepare_medxpertqa_mm(
        local_dir=os.path.join(base_dir, "medxpertqa_mm"),
    )
