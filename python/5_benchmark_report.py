#!/usr/bin/env python
# coding: utf-8
"""
Compute Best-of-N, Self-Consistency (SC), and SC+RM hybrid accuracy for a
fixed set of benchmark groupings from an already-scored dataset produced by
4_scoring_PRM.py. No model inference is performed here -- this only
re-aggregates the per-solution PRM_min_score / PRM_score / score fields
already stored in the scored JSON.
"""

import argparse
import json
import os
from collections import Counter, defaultdict

# Benchmark column -> data_source value(s) that populate it.
BENCHMARK_GROUPS = {
    "MedQA-4": ["med_qa_4op"],
    "MedQA-5": ["med_qa"],
    "MedMCQA": ["medmc_qa"],
    "MMLU-Med": [
        "mmlu_professional_medicine",
        "mmlu_clinical_knowledge",
        "mmlu_college_medicine",
        "mmlu_college_biology",
        "mmlu_anatomy",
        "mmlu_medical_genetics",
    ],
    "DDXPlus": ["ddxplus"],
    # Approximate mapping: this scored dataset has no `agent_clinic`
    # data_source. `nejm` / `osce` are used as stand-ins per user direction;
    # flagged with '*' in the report.
    "AgentClinic (NEJM*)": ["nejm"],
    "AgentClinic (MedQA*)": ["osce"],
}

PRIMARY_COLUMNS = list(BENCHMARK_GROUPS.keys())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate Best-of-N / SC / SC+RM hybrid accuracy from a scored dataset."
    )
    parser.add_argument(
        "--input_json_file",
        type=str,
        default="dataset/dataset_4_scored_dataset/"
        "Llama-3.1-8B-Instruct-gemini_label-filter_yes-ep3-20260731_191535-RAG_yes__sol64_2_test_dataset.json",
        help="Path to the scored dataset JSON (output of 4_scoring_PRM.py)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dataset/dataset_5_benchmark_report",
        help="Directory to write benchmark_report.json / .md into",
    )
    return parser.parse_args()


def best_of_n_correct(sols, score_key="PRM_min_score"):
    valid = [s for s in sols if s.get(score_key, float("-inf")) != float("-inf")]
    if not valid:
        return False
    pred = max(valid, key=lambda s: s[score_key])
    return pred.get("score", 0) == 1


def self_consistency_correct(sols):
    if not sols:
        return False
    most_common_ans, _ = Counter(s["answer"] for s in sols).most_common(1)[0]
    mv_sols = [s for s in sols if s["answer"] == most_common_ans]
    return any(s.get("score", 0) == 1 for s in mv_sols)


def sc_rm_hybrid_correct(sols, score_key="PRM_min_score"):
    if not sols:
        return False
    sums = defaultdict(float)
    for s in sols:
        sc = s.get(score_key, float("-inf"))
        if sc == float("-inf"):
            continue
        sums[s["answer"]] += sc
    if not sums:
        return False
    pred_ans = max(sums, key=sums.get)
    hyb_sols = [s for s in sols if s["answer"] == pred_ans]
    return any(s.get("score", 0) == 1 for s in hyb_sols)


def score_group(items):
    """Return dict of method -> accuracy% and correct/total counts for a list of question items."""
    total = len(items)
    bon_correct = sum(best_of_n_correct(it["solutions"], "PRM_min_score") for it in items)
    sc_correct = sum(self_consistency_correct(it["solutions"]) for it in items)
    hybrid_min_correct = sum(sc_rm_hybrid_correct(it["solutions"], "PRM_min_score") for it in items)
    hybrid_final_correct = sum(sc_rm_hybrid_correct(it["solutions"], "PRM_score") for it in items)

    def pct(c):
        return round(100 * c / total, 2) if total else None

    return {
        "n": total,
        "best_of_n": {"correct": bon_correct, "accuracy": pct(bon_correct)},
        "self_consistency": {"correct": sc_correct, "accuracy": pct(sc_correct)},
        "sc_rm_hybrid_min_score": {"correct": hybrid_min_correct, "accuracy": pct(hybrid_min_correct)},
        "sc_rm_hybrid_final_score": {"correct": hybrid_final_correct, "accuracy": pct(hybrid_final_correct)},
    }


def macro_average(rows):
    """Macro-average (unweighted mean) of accuracy across a list of per-group result dicts."""
    methods = ["best_of_n", "self_consistency", "sc_rm_hybrid_min_score", "sc_rm_hybrid_final_score"]
    out = {"n": None}
    for m in methods:
        vals = [r[m]["accuracy"] for r in rows if r[m]["accuracy"] is not None]
        out[m] = {"correct": None, "accuracy": round(sum(vals) / len(vals), 2) if vals else None}
    return out


def main():
    args = parse_args()

    with open(args.input_json_file, encoding="utf-8") as f:
        data = json.load(f)

    by_source = defaultdict(list)
    for item in data:
        by_source[item.get("data_source")].append(item)

    report = {}
    for col_name, sources in BENCHMARK_GROUPS.items():
        pooled = []
        for src in sources:
            pooled.extend(by_source.get(src, []))
        report[col_name] = score_group(pooled)

    # Derived rows.
    report["MedQA* Average"] = macro_average([report["MedQA-4"], report["MedQA-5"]])
    report["Average"] = macro_average([report[c] for c in PRIMARY_COLUMNS])

    os.makedirs(args.output_dir, exist_ok=True)

    json_path = os.path.join(args.output_dir, "benchmark_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    md_path = os.path.join(args.output_dir, "benchmark_report.md")
    row_order = PRIMARY_COLUMNS + ["MedQA* Average", "Average"]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"Source: `{args.input_json_file}`\n\n")
        f.write(
            "AgentClinic columns are mapped from `nejm` / `osce` data_source "
            "values as stand-ins (no `agent_clinic` data_source exists in "
            "this scored dataset) -- treat as approximate, not a confirmed "
            "match to the paper's AgentClinic benchmark.\n\n"
        )
        f.write("| Benchmark | N | Best-of-N | SC | SC+RM hybrid (PRM_min_score) | SC+RM hybrid (PRM_score) |\n")
        f.write("|---|---|---|---|---|---|\n")
        for name in row_order:
            r = report[name]
            n = r["n"] if r["n"] is not None else "-"
            bon = r["best_of_n"]["accuracy"]
            sc = r["self_consistency"]["accuracy"]
            hmin = r["sc_rm_hybrid_min_score"]["accuracy"]
            hfin = r["sc_rm_hybrid_final_score"]["accuracy"]
            f.write(f"| {name} | {n} | {bon}% | {sc}% | {hmin}% | {hfin}% |\n")

    print(f"Saved {json_path}")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
