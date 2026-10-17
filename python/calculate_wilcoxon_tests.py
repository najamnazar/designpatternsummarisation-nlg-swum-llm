"""Compute paired Wilcoxon signed-rank tests for DPS approaches.

This script compares per-file metric scores between:
- DPS_NLG vs DPS_LLM
- DPS_SWUM vs DPS_LLM

It runs tests for both cosine similarity and BERTScore F1, reports matched-pairs
rank-biserial effect sizes alongside exact p-values, and writes the results into
a delimited block in evaluation-results/results.txt (replacing any previous
block rather than appending a second one).

The DPS_LLM arm must be the same generation run that produces every other
LLM number in the study, i.e. the scores derived from
output/summary-output/llm_summaries.csv (== B.csv). That run is stored in
"llm (mixtral)_vs_human_class_scores.csv"; see DEFAULT_LLM_CSV below.

CHANGE: the Mixtral run is no longer evaluated. By default the LLM side is now the
four GPT/Mistral runs scored by evaluate_summaries.py (see LLM_ARMS below):
- DPS_NLG  vs DPS_GPT, DPS_GPT_NONCONCISE, DPS_MISTRAL, DPS_MISTRAL_NONCONCISE
- DPS_SWUM vs DPS_GPT, DPS_GPT_NONCONCISE, DPS_MISTRAL, DPS_MISTRAL_NONCONCISE
Run it after evaluate_summaries.py, which rewrites results.txt from scratch.
--llm-csv still runs the old single-arm comparison (labelled DPS_LLM).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


ALPHA = 0.05

# The canonical LLM arm: scores for output/summary-output/llm_summaries.csv,
# which is also the corpus used for the rubric rankings and every other
# LLM-derived table in the paper.
# DEFAULT_LLM_CSV = "llm (mixtral)_vs_human_class_scores.csv"

# CHANGE: the LLM arms tested against NLG and SWUM, as (label, class-score file in
# evaluation-results/). The files are written by evaluate_summaries.py.
LLM_ARMS: List[Tuple[str, str]] = [
    ("DPS_GPT", "llm (gpt)_vs_human_class_scores.csv"),
    ("DPS_GPT_NONCONCISE", "llm (gpt non-concise)_vs_human_class_scores.csv"),
    ("DPS_MISTRAL", "llm (mistral)_vs_human_class_scores.csv"),
    ("DPS_MISTRAL_NONCONCISE", "llm (mistral non-concise)_vs_human_class_scores.csv"),
]

# CHANGE: the deterministic baselines each LLM arm is compared with.
BASELINE_ARMS: List[Tuple[str, str]] = [
    ("DPS_NLG", "nlg_vs_human_class_scores.csv"),
    ("DPS_SWUM", "swum_vs_human_class_scores.csv"),
]

# CHANGE: width of the comparison column; "DPS_SWUM vs DPS_MISTRAL_NONCONCISE" is 34 characters.
COMPARISON_WIDTH = 36

# Marker delimiting this script's block inside results.txt, so that re-running
# replaces the previous results instead of stacking a contradictory copy.
BLOCK_START = "<<<BEGIN WILCOXON SIGNED-RANK TESTS>>>"
BLOCK_END = "<<<END WILCOXON SIGNED-RANK TESTS>>>"


@dataclass
class TestResult:
    comparison: str
    metric: str
    n_pairs: int
    w_statistic: float
    p_value: float
    median_diff: float
    effect_size_r: float
    effect_magnitude: str
    direction: str
    significant: bool
    note: str = ""


def rank_biserial_effect_size(diffs: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation for the Wilcoxon signed-rank test.

    r = (W+ - W-) / (W+ + W-), computed over the non-zero paired differences
    with ties resolved by average ranks. The sign follows the direction of the
    paired difference, so a negative r means the second-named method scored
    higher. |r| is interpreted with the usual thresholds (see effect_magnitude).
    """
    nonzero = diffs[diffs != 0]
    if nonzero.size == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero))
    w_plus = float(ranks[nonzero > 0].sum())
    w_minus = float(ranks[nonzero < 0].sum())
    total = w_plus + w_minus
    if total == 0:
        return 0.0
    return (w_plus - w_minus) / total


def effect_magnitude(r: float) -> str:
    """Label |r| using the conventional Cohen/Rosenthal thresholds."""
    magnitude = abs(r)
    if magnitude < 0.10:
        return "negligible"
    if magnitude < 0.30:
        return "small"
    if magnitude < 0.50:
        return "medium"
    if magnitude < 0.70:
        return "large"
    return "very large"


def normalize_text(value: object) -> str:
    """Normalize text keys to improve stable matching across files."""
    if pd.isna(value):
        return ""
    text = str(value).strip().lower().replace("\\", "/")
    return " ".join(text.split())


def normalize_filename(value: object) -> str:
    """Normalize filename by extracting basename and lowercasing."""
    text = normalize_text(value)
    if not text:
        return ""
    return Path(text).name


def build_pair_keys(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Build a stable per-file key used for pairwise merges.

    Returns (primary_key, fallback_key):
    - primary_key: base_project + project_path + filename (if project_path exists)
    - fallback_key: base_project + filename
    """
    required_common = {"base_project", "filename"}
    missing = required_common.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required key columns: {sorted(missing)}")

    base_project = df["base_project"].map(normalize_text)
    filename = df["filename"].map(normalize_filename)

    fallback_key = base_project + "::" + filename

    if "project_path" in df.columns:
        project_path = df["project_path"].map(normalize_text)
        primary_key = base_project + "::" + project_path + "::" + filename
    else:
        primary_key = fallback_key

    return primary_key, fallback_key


def validate_input_frame(name: str, df: pd.DataFrame) -> None:
    """Validate minimum columns and duplicate key conditions."""
    required_cols = {"cosine_similarity", "bert_f1", "base_project", "filename"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"{name}: missing required columns: {sorted(missing)}")

    dup_count = int(df["pair_key"].duplicated().sum())
    if dup_count:
        sample = df.loc[df["pair_key"].duplicated(), "pair_key"].head(5).tolist()
        raise ValueError(
            f"{name}: found {dup_count} duplicate pair keys. Sample: {sample}"
        )


def prepare_frame(csv_path: Path, label: str) -> pd.DataFrame:
    """Load and normalize a class-score CSV for pairwise matching."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Input file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    pair_key, fallback_key = build_pair_keys(df)
    df["pair_key"] = pair_key
    df["fallback_key"] = fallback_key

    # Keep only columns needed for paired statistical testing.
    out = df[["pair_key", "fallback_key", "cosine_similarity", "bert_f1"]].copy()
    out.rename(
        columns={
            "cosine_similarity": f"cosine_similarity_{label}",
            "bert_f1": f"bert_f1_{label}",
        },
        inplace=True,
    )
    validate_input_frame(label, df)
    return out


def merge_pairwise(
    lhs: pd.DataFrame,
    rhs: pd.DataFrame,
    lhs_label: str,
    rhs_label: str,
    allow_fallback: bool = False,
) -> pd.DataFrame:
    """Pairwise merge on the primary per-file key.

    Every row must match on base_project + project_path + filename. A shortfall
    means the two arms were generated over different file sets, which silently
    changes what the test is comparing; that is treated as an error rather than
    something to paper over.

    With allow_fallback=True the unmatched rows are additionally matched on
    base_project + filename (deterministic occurrence index per key). That
    recovery pairs files that merely share a name, so it is opt-in and reported.
    """
    strict = lhs.merge(rhs.drop(columns=["fallback_key"]), on="pair_key", how="inner")

    lhs_unmatched = lhs.loc[~lhs["pair_key"].isin(strict["pair_key"])].copy()
    rhs_unmatched = rhs.loc[~rhs["pair_key"].isin(strict["pair_key"])].copy()

    if not lhs_unmatched.empty or not rhs_unmatched.empty:
        detail = (
            f"{lhs_label} vs {rhs_label}: strict key match covers "
            f"{len(strict)} pairs, but {len(lhs_unmatched)} {lhs_label} row(s) and "
            f"{len(rhs_unmatched)} {rhs_label} row(s) are unmatched.\n"
            f"  unmatched in {lhs_label}: {sorted(lhs_unmatched['pair_key'])[:5]}\n"
            f"  unmatched in {rhs_label}: {sorted(rhs_unmatched['pair_key'])[:5]}"
        )
        if not allow_fallback:
            raise ValueError(
                "Incomplete pairing; the two arms do not cover the same files.\n"
                + detail
                + "\nCheck that both CSVs come from the same evaluation corpus, "
                "or re-run with --allow-fallback to match on filename only."
            )
        print("WARNING: falling back to filename-only matching.\n" + detail)

    recovered = pd.DataFrame(columns=strict.columns)
    if allow_fallback and not lhs_unmatched.empty and not rhs_unmatched.empty:
        lhs_unmatched["fallback_idx"] = lhs_unmatched.groupby("fallback_key").cumcount()
        rhs_unmatched["fallback_idx"] = rhs_unmatched.groupby("fallback_key").cumcount()

        recovered = lhs_unmatched.merge(
            rhs_unmatched.drop(columns=["pair_key"]),
            on=["fallback_key", "fallback_idx"],
            how="inner",
        )
        if not recovered.empty:
            recovered.drop(columns=["fallback_idx"], inplace=True)

    merged = pd.concat([strict, recovered], ignore_index=True)
    if merged.empty:
        raise ValueError(f"No matched pairs after merge: {lhs_label} vs {rhs_label}")

    merged.drop(columns=["fallback_key"], errors="ignore", inplace=True)
    return merged


def run_wilcoxon_test(
    merged: pd.DataFrame,
    metric: str,
    lhs_label: str,
    rhs_label: str,
) -> TestResult:
    """Run Wilcoxon signed-rank test for a single metric and pairwise comparison."""
    lhs_col = f"{metric}_{lhs_label}"
    rhs_col = f"{metric}_{rhs_label}"

    pair_df = merged[[lhs_col, rhs_col]].dropna()
    x = pair_df[lhs_col].to_numpy(dtype=float)
    y = pair_df[rhs_col].to_numpy(dtype=float)

    n_pairs = int(len(pair_df))
    if n_pairs == 0:
        raise ValueError(f"No valid non-NaN pairs for {lhs_label} vs {rhs_label} ({metric})")

    diffs = x - y
    median_diff = float(np.median(diffs))

    if np.allclose(diffs, 0.0):
        return TestResult(
            comparison=f"{lhs_label} vs {rhs_label}",
            metric=metric,
            n_pairs=n_pairs,
            w_statistic=0.0,
            p_value=1.0,
            median_diff=median_diff,
            effect_size_r=0.0,
            effect_magnitude="negligible",
            direction="no difference",
            significant=False,
            note="All paired differences are zero.",
        )

    result = wilcoxon(x, y, alternative="two-sided")
    r_effect = rank_biserial_effect_size(diffs)

    if median_diff > 0:
        direction = f"{lhs_label} tends higher"
    elif median_diff < 0:
        direction = f"{rhs_label} tends higher"
    else:
        direction = "no median difference"

    return TestResult(
        comparison=f"{lhs_label} vs {rhs_label}",
        metric=metric,
        n_pairs=n_pairs,
        w_statistic=float(result.statistic),
        p_value=float(result.pvalue),
        median_diff=median_diff,
        effect_size_r=r_effect,
        effect_magnitude=effect_magnitude(r_effect),
        direction=direction,
        significant=bool(result.pvalue < ALPHA),
    )


def format_results_section(
    results: Sequence[TestResult],
    matched_info: Dict[str, int],
    sources: Dict[str, Path],
) -> str:
    """Format the delimited text block written into evaluation-results/results.txt."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines: List[str] = []
    lines.append(BLOCK_START)
    lines.append("=" * 60)
    lines.append("WILCOXON SIGNED-RANK TESTS (PAIRED)")
    lines.append("=" * 60)
    lines.append(f"Generated at: {ts}")
    lines.append("Dataset: paired per-file class scores")
    lines.append(f"Significance level (alpha): {ALPHA:.2f}")
    lines.append("Tests: two-sided; metrics = cosine_similarity, bert_f1")
    lines.append("Effect size: matched-pairs rank-biserial r = (W+ - W-) / (W+ + W-)")
    lines.append("")
    lines.append("Input files:")
    for label, path in sources.items():
        lines.append(f"  - {label}: {path.name}")

    lines.append("")
    lines.append("Matched pairs before NaN filtering:")
    for comp, n in matched_info.items():
        lines.append(f"  - {comp}: {n}")

    lines.append("")
    # lines.append(
    #     "comparison           metric              n_pairs   W_stat        p_value       "
    #     "median_diff        r  magnitude    interpretation"
    # )
    # lines.append("-" * 140)
    # CHANGE: the comparison column is wider to fit the longer GPT/Mistral labels.
    lines.append(
        f"{'comparison':<{COMPARISON_WIDTH}} metric              n_pairs   W_stat        p_value       "
        "median_diff        r  magnitude    interpretation"
    )
    lines.append("-" * (120 + COMPARISON_WIDTH))

    for r in results:
        sig_label = "significant" if r.significant else "not-significant"
        interpretation = f"{sig_label}; {r.direction}"
        row = (
            # f"{r.comparison:<20} "
            f"{r.comparison:<{COMPARISON_WIDTH}} "  # CHANGE: wider comparison column
            f"{r.metric:<18} "
            f"{r.n_pairs:>7} "
            f"{r.w_statistic:>10.4f} "
            f"{r.p_value:>13.6g} "
            f"{r.median_diff:>12.6f} "
            f"{r.effect_size_r:>8.3f} "
            f"{r.effect_magnitude:<12} "
            f"{interpretation}"
        )
        lines.append(row)
        if r.note:
            lines.append(f"  note: {r.note}")

    lines.append(BLOCK_END)
    return "\n".join(lines) + "\n"


def write_block(results_txt: Path, section: str) -> None:
    """Write the section into results.txt, replacing any previous block.

    Appending would leave two blocks in the file whenever the script is re-run,
    and a reader has no way to tell which one is current.
    """
    existing = results_txt.read_text(encoding="utf-8") if results_txt.exists() else ""

    start = existing.find(BLOCK_START)
    end = existing.find(BLOCK_END)
    if start != -1 and end != -1:
        updated = existing[:start] + section + existing[end + len(BLOCK_END) + 1 :]
    else:
        separator = "" if existing.endswith("\n\n") or not existing else "\n\n"
        updated = existing + separator + section

    results_txt.write_text(updated, encoding="utf-8")


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired Wilcoxon signed-rank tests with rank-biserial effect sizes."
    )
    # parser.add_argument(
    #     "--llm-csv",
    #     type=str,
    #     default=DEFAULT_LLM_CSV,
    #     help=(
    #         "Filename (inside evaluation-results/) of the DPS_LLM class scores. "
    #         f"Default: {DEFAULT_LLM_CSV}"
    #     ),
    # )
    # CHANGE: no default any more; without --llm-csv every arm in LLM_ARMS is tested.
    parser.add_argument(
        "--llm-csv",
        type=str,
        default=None,
        help=(
            "Optional filename (inside evaluation-results/) of a single LLM class-score file, "
            "tested as DPS_LLM instead of the GPT/Mistral arms. "
            "Default: all of " + ", ".join(name for _, name in LLM_ARMS)
        ),
    )
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help=(
            "Permit filename-only matching for rows the strict key cannot pair. "
            "Off by default: an incomplete pairing usually means the arms were "
            "generated over different file sets."
        ),
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Optional path to also write the test results as CSV.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_arguments(argv)

    root = Path(__file__).resolve().parent.parent
    eval_dir = root / "evaluation-results"

    # nlg_csv = eval_dir / "nlg_vs_human_class_scores.csv"
    # swum_csv = eval_dir / "swum_vs_human_class_scores.csv"
    # llm_csv = eval_dir / args.llm_csv
    # results_txt = eval_dir / "results.txt"

    # nlg = prepare_frame(nlg_csv, "DPS_NLG")
    # swum = prepare_frame(swum_csv, "DPS_SWUM")
    # llm = prepare_frame(llm_csv, "DPS_LLM")

    # nlg_vs_llm = merge_pairwise(nlg, llm, "DPS_NLG", "DPS_LLM", args.allow_fallback)
    # swum_vs_llm = merge_pairwise(swum, llm, "DPS_SWUM", "DPS_LLM", args.allow_fallback)

    # matched_info = {
    #     "DPS_NLG vs DPS_LLM": int(len(nlg_vs_llm)),
    #     "DPS_SWUM vs DPS_LLM": int(len(swum_vs_llm)),
    # }
    # sources = {"DPS_NLG": nlg_csv, "DPS_SWUM": swum_csv, "DPS_LLM": llm_csv}

    # test_results: List[TestResult] = []
    # for metric in ("cosine_similarity", "bert_f1"):
    #     test_results.append(run_wilcoxon_test(nlg_vs_llm, metric, "DPS_NLG", "DPS_LLM"))
    #     test_results.append(run_wilcoxon_test(swum_vs_llm, metric, "DPS_SWUM", "DPS_LLM"))
    # CHANGE: test every LLM arm against both baselines instead of a single DPS_LLM arm.
    results_txt = eval_dir / "results.txt"
    if args.llm_csv is not None:
        llm_arms = [("DPS_LLM", args.llm_csv)]
    else:
        llm_arms = LLM_ARMS

    sources: Dict[str, Path] = {}
    baselines: List[Tuple[str, pd.DataFrame]] = []
    for label, filename in BASELINE_ARMS:
        csv_path = eval_dir / filename
        baselines.append((label, prepare_frame(csv_path, label)))
        sources[label] = csv_path

    llms: List[Tuple[str, pd.DataFrame]] = []
    for label, filename in llm_arms:
        csv_path = eval_dir / filename
        # CHANGE: a missing LLM score file is skipped with a warning (run evaluate_summaries.py first).
        if not csv_path.exists():
            print(f"WARNING: {csv_path} not found; skipping {label}.")
            continue
        llms.append((label, prepare_frame(csv_path, label)))
        sources[label] = csv_path
    if not llms:
        raise FileNotFoundError(
            f"No LLM class-score files found in {eval_dir}; run python/evaluate_summaries.py first."
        )

    # CHANGE: one paired merge per (baseline, LLM arm); tests are grouped by metric, then LLM arm.
    merged_pairs: List[Tuple[str, str, pd.DataFrame]] = []
    matched_info: Dict[str, int] = {}
    for llm_label, llm in llms:
        for base_label, base in baselines:
            merged = merge_pairwise(base, llm, base_label, llm_label, args.allow_fallback)
            merged_pairs.append((base_label, llm_label, merged))
            matched_info[f"{base_label} vs {llm_label}"] = int(len(merged))

    test_results: List[TestResult] = []
    for metric in ("cosine_similarity", "bert_f1"):
        for base_label, llm_label, merged in merged_pairs:
            test_results.append(run_wilcoxon_test(merged, metric, base_label, llm_label))

    section = format_results_section(test_results, matched_info, sources)
    write_block(results_txt, section)

    if args.csv_out is not None:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([vars(r) for r in test_results]).to_csv(args.csv_out, index=False)

    print("Wilcoxon signed-rank tests completed.")
    # print(f"DPS_LLM arm: {llm_csv.name}")
    print("LLM arms: " + ", ".join(f"{label} ({sources[label].name})" for label, _ in llms))  # CHANGE
    print(f"Wrote results block to: {results_txt}")
    if args.csv_out is not None:
        print(f"Wrote CSV to: {args.csv_out}")
    for result in test_results:
        print(
            f"- {result.comparison} | {result.metric} | n={result.n_pairs} | "
            f"W={result.w_statistic:.4f} | p={result.p_value:.6g} | "
            f"r={result.effect_size_r:.3f} ({result.effect_magnitude})"
        )


if __name__ == "__main__":
    main()
