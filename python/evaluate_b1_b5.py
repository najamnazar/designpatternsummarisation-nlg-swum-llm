#!/usr/bin/env python3
"""
Evaluate B1..B5 iterations against human summaries.

Computes BERTScore and Cosine Similarity for each of the five generated
summary CSVs (B1.csv through B5.csv) by comparing them against human summaries.
Produces a consolidated report with per-iteration scores and aggregate
min/max/average statistics.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import numpy as np

# Reuse loading/matching logic from evaluate_summaries.py
sys.path.insert(0, str(Path(__file__).parent))
from evaluate_summaries import (
    normalize_project_identifier,
    normalize_filename,
    extract_full_project_path_from_url,
    extract_base_project_name,
    MetricsCalculator,
)


def load_human_summaries(csv_path: Path) -> pd.DataFrame:
    """Load human summaries with normalized project/filename keys."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    required = ["Project", "File Name", "Human Summary", "URL"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in human CSV: {missing}")
    result = df[required].copy()
    result.columns = ["base_project", "filename", "summary", "url"]
    result = result.fillna("")
    result["project"] = result.apply(
        lambda r: extract_full_project_path_from_url(r["url"], r["base_project"]),
        axis=1,
    )
    result = result[result["summary"].str.len() > 0]
    return result[["project", "base_project", "filename", "summary"]]


def load_generated_summaries(csv_path: Path) -> pd.DataFrame:
    """Load a B{i}.csv file with project/folder/filename/summary."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    # Attempt to find columns flexibly
    proj_col = None
    folder_col = None
    file_col = None
    summ_col = None
    for col in df.columns:
        low = col.lower()
        if low in {"project", "project name", "project_name"}:
            proj_col = col
        elif low in {"folder", "folder name", "folder_name"}:
            folder_col = col
        elif low in {"filename", "file name", "file_name", "class"}:
            file_col = col
        elif low == "summary":
            summ_col = col
    if not all([proj_col, file_col, summ_col]):
        raise ValueError(
            f"Could not locate project/filename/summary columns in {csv_path.name}"
        )
    cols = [proj_col]
    if folder_col:
        cols.append(folder_col)
    cols.extend([file_col, summ_col])
    result = df[cols].copy()
    rename = {proj_col: "project", file_col: "filename", summ_col: "summary"}
    if folder_col:
        rename[folder_col] = "folder"
    result = result.rename(columns=rename)
    result["project"] = result["project"].astype(str).str.strip()
    result["folder"] = (
        result.get("folder", "").astype(str).str.strip() if folder_col else ""
    )
    result["filename"] = result["filename"].astype(str).str.strip()
    result["summary"] = result["summary"].astype(str).str.strip()
    result["base_project"] = result["project"].apply(extract_base_project_name)
    result = result[result["summary"].str.len() > 0]
    return result[["project", "base_project", "filename", "summary"]]


def match_summaries(human_df: pd.DataFrame, gen_df: pd.DataFrame) -> pd.DataFrame:
    """Match human and generated summaries via normalized base_project + filename."""
    human_df = human_df.copy()
    gen_df = gen_df.copy()
    human_df["norm_proj"] = human_df["base_project"].apply(normalize_project_identifier)
    gen_df["norm_proj"] = gen_df["base_project"].apply(normalize_project_identifier)
    human_df["norm_file"] = human_df["filename"].apply(normalize_filename)
    gen_df["norm_file"] = gen_df["filename"].apply(normalize_filename)
    human_df["match_key"] = human_df["norm_proj"] + "::" + human_df["norm_file"]
    gen_df["match_key"] = gen_df["norm_proj"] + "::" + gen_df["norm_file"]
    human_df["dup_idx"] = human_df.groupby("match_key").cumcount()
    gen_df["dup_idx"] = gen_df.groupby("match_key").cumcount()
    merged = human_df.merge(
        gen_df,
        on=["match_key", "dup_idx"],
        how="inner",
        suffixes=("_human", "_gen"),
    )
    if merged.empty:
        raise ValueError("No matches found between human and generated summaries.")
    return merged[
        [
            "base_project_human",
            "filename_human",
            "summary_human",
            "summary_gen",
        ]
    ]


def evaluate_iteration(
    human_csv: Path, gen_csv: Path, iteration: int
) -> pd.DataFrame:
    """Evaluate a single Bi.csv against human summaries, return per-pair scores."""
    print(f"Evaluating iteration {iteration}: {gen_csv.name}")
    human_df = load_human_summaries(human_csv)
    gen_df = load_generated_summaries(gen_csv)
    matched = match_summaries(human_df, gen_df)
    print(f"  Matched {len(matched)} pairs")
    
    # Compute cosine similarities pair-by-pair
    cosines = []
    for _, row in matched.iterrows():
        cos = MetricsCalculator.cosine_similarity(row["summary_human"], row["summary_gen"])
        cosines.append(cos)
    
    # Compute BERTScores in batch for efficiency
    candidates = matched["summary_gen"].tolist()
    references = matched["summary_human"].tolist()
    try:
        _, _, f1_scores = MetricsCalculator.bert_scores(candidates, references)
        berts = f1_scores.tolist()
    except Exception as e:
        print(f"  Warning: BERTScore computation failed ({e}). Using zeros.")
        berts = [0.0] * len(matched)
    
    result = pd.DataFrame(
        {
            "iteration": iteration,
            "base_project": matched["base_project_human"].values,
            "filename": matched["filename_human"].values,
            "cosine_similarity": cosines,
            "bert_f1": berts,
        }
    )
    return result


def compute_aggregate_stats(all_results: pd.DataFrame) -> pd.DataFrame:
    """Compute min/max/avg for cosine and BERTScore across iterations."""
    grouped = (
        all_results.groupby("iteration")
        .agg(
            {
                "cosine_similarity": ["min", "max", "mean"],
                "bert_f1": ["min", "max", "mean"],
            }
        )
        .reset_index()
    )
    grouped.columns = [
        "iteration",
        "cosine_min",
        "cosine_max",
        "cosine_avg",
        "bert_min",
        "bert_max",
        "bert_avg",
    ]
    # Also compute overall (across all iterations) stats
    overall = pd.DataFrame(
        {
            "iteration": ["Overall"],
            "cosine_min": [all_results["cosine_similarity"].min()],
            "cosine_max": [all_results["cosine_similarity"].max()],
            "cosine_avg": [all_results["cosine_similarity"].mean()],
            "bert_min": [all_results["bert_f1"].min()],
            "bert_max": [all_results["bert_f1"].max()],
            "bert_avg": [all_results["bert_f1"].mean()],
        }
    )
    return pd.concat([grouped, overall], ignore_index=True)


def main() -> None:
    repo_root = Path(__file__).parent.parent
    human_csv = repo_root / "input" / "DPS_Human_Summaries.csv"
    out_dir = repo_root / "evaluation-results"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not human_csv.exists():
        raise FileNotFoundError(f"Human summaries not found: {human_csv}")

    iterations = 5
    gen_csvs = [
        repo_root / "output" / "summary-output" / f"B{i}.csv" for i in range(1, iterations + 1)
    ]
    missing = [c for c in gen_csvs if not c.exists()]
    if missing:
        raise FileNotFoundError(f"Missing generated CSVs: {[str(c) for c in missing]}")

    all_results: List[pd.DataFrame] = []
    for i, gen_csv in enumerate(gen_csvs, start=1):
        iter_results = evaluate_iteration(human_csv, gen_csv, i)
        all_results.append(iter_results)

    combined = pd.concat(all_results, ignore_index=True)
    aggregate = compute_aggregate_stats(combined)

    # Write per-pair detail CSV
    detail_csv = out_dir / "b1_b5_vs_human_detail.csv"
    combined.to_csv(detail_csv, index=False)
    print(f"\nPer-pair results written to: {detail_csv}")

    # Write aggregate summary CSV
    summary_csv = out_dir / "b1_b5_vs_human_summary.csv"
    aggregate.to_csv(summary_csv, index=False)
    print(f"Aggregate summary written to: {summary_csv}")

    # Also write a plain text report for easy reading
    txt_report = out_dir / "b1_b5_vs_human_report.txt"
    with open(txt_report, "w", encoding="utf-8") as f:
        f.write("Evaluation of B1..B5 vs Human Summaries\n")
        f.write("=" * 60 + "\n\n")
        f.write("Per-Iteration Statistics:\n")
        f.write("-" * 60 + "\n")
        for _, row in aggregate.iterrows():
            f.write(f"Iteration {row['iteration']}:\n")
            f.write(f"  Cosine Similarity: min={row['cosine_min']:.4f}, "
                    f"max={row['cosine_max']:.4f}, avg={row['cosine_avg']:.4f}\n")
            f.write(f"  BERTScore F1:      min={row['bert_min']:.4f}, "
                    f"max={row['bert_max']:.4f}, avg={row['bert_avg']:.4f}\n\n")
    print(f"Text report written to: {txt_report}")

    print("\n=== Summary ===")
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
