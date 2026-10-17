#!/usr/bin/env python3
"""Word and character statistics per summary for every DPS summary set.

Computes the average, minimum and maximum number of words and characters per
summary (the measures of the paper's ``tab:wordstats`` table) for:

- DPS_NLG and DPS_SWUM
- DPS_GPT and DPS_MISTRAL: concise prompt (llm_summaries_{gpt,mistral}.csv)
- DPS_GPT_NONCONCISE and DPS_MISTRAL_NONCONCISE: non-concise 50-word prompt
- DPS_GPT_20/40/60/80 and DPS_MISTRAL_20/40/60/80: summary-length variants
- Human: the ground-truth summaries, as a reference row

Words are whitespace-separated tokens; characters are the length of the
whitespace-normalised summary, spaces included. Summary sets whose CSV is
missing are skipped with a warning.

Outputs:
- evaluation-results/summary_length_stats.csv (also holds the median and standard deviation)
- a delimited block in evaluation-results/results.txt, replaced on every re-run,
  with a plain-text table and a LaTeX table ready to paste into the paper.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd

from project_paths import HUMAN_SUMMARIES_CSV, REPO_ROOT

SUMMARY_DIR = REPO_ROOT / "output" / "summary-output"
EVAL_DIR = REPO_ROOT / "evaluation-results"

# (label, CSV path, summary column), in table order.
SUMMARY_SETS: List[Tuple[str, Path, str]] = [
    ("DPS_NLG", SUMMARY_DIR / "nlg_summaries.csv", "Summary"),
    ("DPS_SWUM", SUMMARY_DIR / "swum_summaries.csv", "Summary"),
    ("DPS_GPT", SUMMARY_DIR / "llm_summaries_gpt.csv", "Summary"),
    ("DPS_GPT_NONCONCISE", SUMMARY_DIR / "llm_summaries_nonconcise_gpt.csv", "Summary"),
    *[(f"DPS_GPT_{n}", SUMMARY_DIR / f"llm_summaries_gpt_{n}.csv", "Summary") for n in (20, 40, 60, 80)],
    ("DPS_MISTRAL", SUMMARY_DIR / "llm_summaries_mistral.csv", "Summary"),
    ("DPS_MISTRAL_NONCONCISE", SUMMARY_DIR / "llm_summaries_nonconcise_mistral.csv", "Summary"),
    *[(f"DPS_MISTRAL_{n}", SUMMARY_DIR / f"llm_summaries_mistral_{n}.csv", "Summary") for n in (20, 40, 60, 80)],
    ("Human", HUMAN_SUMMARIES_CSV, "Human Summary"),
]

# Marker delimiting this script's block inside results.txt, so that re-running
# replaces the previous results instead of stacking a second copy.
BLOCK_START = "<<<BEGIN SUMMARY LENGTH STATISTICS>>>"
BLOCK_END = "<<<END SUMMARY LENGTH STATISTICS>>>"


@dataclass
class LengthStats:
    method: str
    source: str
    n_summaries: int
    n_empty: int
    avg_words: float
    min_words: int
    max_words: int
    median_words: float
    std_words: float
    avg_chars: float
    min_chars: int
    max_chars: int
    median_chars: float
    std_chars: float


def normalise(text: object) -> str:
    """Collapse all whitespace (including newlines) to single spaces."""
    if pd.isna(text):
        return ""
    return " ".join(str(text).split())


def compute_stats(label: str, csv_path: Path, column: str) -> Optional[LengthStats]:
    if not csv_path.exists():
        print(f"WARNING: {csv_path} not found; skipping {label}.")
        return None
    df = pd.read_csv(csv_path)
    if column not in df.columns:
        raise ValueError(f"{csv_path} has no '{column}' column (columns: {list(df.columns)})")

    summaries = df[column].map(normalise)
    non_empty = summaries[summaries != ""]
    if non_empty.empty:
        print(f"WARNING: {csv_path} has no non-empty summaries; skipping {label}.")
        return None

    words = non_empty.str.split().str.len()
    chars = non_empty.str.len()
    return LengthStats(
        method=label,
        source=str(csv_path.relative_to(REPO_ROOT)),
        n_summaries=int(len(non_empty)),
        n_empty=int(len(summaries) - len(non_empty)),
        avg_words=round(float(words.mean()), 2),
        min_words=int(words.min()),
        max_words=int(words.max()),
        median_words=round(float(words.median()), 2),
        std_words=round(float(words.std(ddof=1)), 2) if len(words) > 1 else 0.0,
        avg_chars=round(float(chars.mean()), 2),
        min_chars=int(chars.min()),
        max_chars=int(chars.max()),
        median_chars=round(float(chars.median()), 2),
        std_chars=round(float(chars.std(ddof=1)), 2) if len(chars) > 1 else 0.0,
    )


def format_text_table(stats: Sequence[LengthStats]) -> List[str]:
    width = max(len("Method"), *(len(s.method) for s in stats)) + 2
    header = (
        f"{'Method':<{width}}{'N':>5}{'AvgWords':>10}{'MinWords':>10}{'MaxWords':>10}"
        f"{'AvgChars':>10}{'MinChars':>10}{'MaxChars':>10}"
    )
    lines = [header, "-" * len(header)]
    for s in stats:
        lines.append(
            f"{s.method:<{width}}{s.n_summaries:>5}{s.avg_words:>10.2f}{s.min_words:>10}{s.max_words:>10}"
            f"{s.avg_chars:>10.2f}{s.min_chars:>10}{s.max_chars:>10}"
        )
    return lines


def format_latex_table(stats: Sequence[LengthStats]) -> List[str]:
    lines = [
        r"\begin{table}[htb]",
        r"  \caption{Average, minimum, and maximum words and characters per summary}",
        r"  \label{tab:wordstats}",
        r"  \centering",
        r"  \begin{tabular}{l@{\hspace{3pt}}r@{\hspace{4pt}}r@{\hspace{4pt}}r@{\hspace{8pt}}r@{\hspace{4pt}}r@{\hspace{4pt}}r}",
        r"    \toprule",
        r"    \textbf{Method} & \textbf{Avg} & \textbf{Min} & \textbf{Max} & \textbf{Avg} & \textbf{Min} & \textbf{Max} \\",
        r"     & \textbf{Words} & \textbf{Words} & \textbf{Words} & \textbf{Chars} & \textbf{Chars} & \textbf{Chars} \\",
        r"    \midrule",
    ]
    for i, s in enumerate(stats):
        # Rule before the human reference row so it reads as separate from the DPS methods.
        if s.method == "Human" and i > 0:
            lines.append(r"    \midrule")
        name = s.method.replace("_", r"\_")
        lines.append(
            f"    {name} & {s.avg_words:.2f} & {s.min_words} & {s.max_words} & "
            f"{s.avg_chars:.2f} & {s.min_chars} & {s.max_chars} \\\\"
        )
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return lines


def format_results_section(stats: Sequence[LengthStats]) -> str:
    lines = [
        BLOCK_START,
        "=" * 60,
        "SUMMARY LENGTH STATISTICS (WORDS AND CHARACTERS PER SUMMARY)",
        "=" * 60,
        f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "Words = whitespace-separated tokens; Chars = characters incl. spaces (whitespace normalised).",
        "",
        "Input files:",
        *[f"  - {s.method}: {s.source}" + (f" ({s.n_empty} empty skipped)" if s.n_empty else "") for s in stats],
        "",
        *format_text_table(stats),
        "",
        "LaTeX:",
        *format_latex_table(stats),
        BLOCK_END,
    ]
    return "\n".join(lines) + "\n"


def write_block(results_txt: Path, section: str) -> None:
    """Write the section into results.txt, replacing any previous block."""
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
        description="Average, minimum and maximum words and characters per summary."
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=EVAL_DIR / "summary_length_stats.csv",
        help="CSV output path. Default: evaluation-results/summary_length_stats.csv",
    )
    parser.add_argument(
        "--skip-human",
        action="store_true",
        help="Leave out the human ground-truth reference row.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_arguments(argv)

    stats: List[LengthStats] = []
    for label, csv_path, column in SUMMARY_SETS:
        if args.skip_human and label == "Human":
            continue
        result = compute_stats(label, csv_path, column)
        if result is not None:
            stats.append(result)
    if not stats:
        raise FileNotFoundError(f"No summary CSVs found in {SUMMARY_DIR}.")

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([vars(s) for s in stats]).to_csv(args.csv_out, index=False)

    results_txt = EVAL_DIR / "results.txt"
    write_block(results_txt, format_results_section(stats))

    print("\n".join(format_text_table(stats)))
    print(f"\nWrote CSV to: {args.csv_out}")
    print(f"Wrote results block to: {results_txt}")


if __name__ == "__main__":
    main()
