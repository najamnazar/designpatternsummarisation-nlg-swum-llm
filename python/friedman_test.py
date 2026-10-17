"""Friedman tests comparing the four DPS methods across the 5 ranking criteria.

Input is the four-way LLM-judge ranking written by rank_summaries_four_way.py
(evaluation-results/four_way_ranking_detail.csv). The methods and criteria are
imported from that script, so both always use the same lists:
    DPS_NLG, DPS_LLM (GPT), DPS_LLM (Mistral), DPS_SWUM

Two-level analysis:
  PART 1 — Five per-criterion Friedman tests.
            Each test uses the class files as blocks (n <= 150) and k = 4
            methods as treatments (one test per criterion).
  PART 2 — One cross-criteria Friedman test.
            Uses n = 5 criteria as blocks, k = 4 methods as treatments.
            Each cell is the mean points of a method on that criterion.

Scoring convention (the *_points_<slug> columns of the detail CSV):
    1st place = 4 pts,  2nd = 3 pts,  3rd = 2 pts,  4th = 1 pt

Only classes the ranking script marked as ranked, with a valid judgement for
the criterion, are used. Each such row must give the four methods the points
4, 3, 2 and 1 exactly once; any other row stops the script with an error.

Post-hoc pairwise Wilcoxon signed-rank tests (Bonferroni-corrected over the
6 method pairs) are run for any significant Friedman result where n >= 10.
For Part 2 (n = 5) the pairwise mean differences are reported as a
descriptive alternative.

Results are printed to the console and written to a delimited block in
evaluation-results/results.txt (replaced on re-run).
"""

from __future__ import annotations

from datetime import datetime
from itertools import combinations
from typing import NamedTuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

from rank_summaries_four_way import CRITERIA, DETAIL_CSV, METHODS, RESULTS_TXT, Method


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALPHA = 0.05
K = len(METHODS)                                # number of methods (treatments)
METHOD_PAIRS = list(combinations(METHODS, 2))   # C(4,2) = 6 pairwise comparisons
BONFERRONI_PAIRS = len(METHOD_PAIRS)
ALPHA_ADJ = ALPHA / BONFERRONI_PAIRS            # = 0.00833
MIN_POSTHOC_BLOCKS = 10                         # below this, pairwise Wilcoxon is not run

# Points the ranking script gives to positions 1st..4th; every valid row holds each once.
EXPECTED_POINTS = list(range(K, 0, -1))

# Marker delimiting this script's block inside results.txt, so that re-running
# replaces the previous results instead of stacking a second, contradictory copy.
BLOCK_START = "<<<BEGIN FRIEDMAN TESTS>>>"
BLOCK_END = "<<<END FRIEDMAN TESTS>>>"

NAME_WIDTH = max(len(m.label) for m in METHODS)
PAIR_WIDTH = 2 * NAME_WIDTH + 4


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def points_column(criterion: str, method: Method) -> str:
    return f"{criterion}_points_{method.slug}"


def load_rankings() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (all rows, ranked rows) of the four-way detail CSV, after checking its columns."""
    if not DETAIL_CSV.exists():
        raise FileNotFoundError(
            f"Rankings file not found: {DETAIL_CSV}\n"
            "Run python/rank_summaries_four_way.py first."
        )

    df_all = pd.read_csv(DETAIL_CSV)
    required = ["status"]
    for crit_key, _ in CRITERIA:
        required.append(f"{crit_key}_valid")
        required.extend(points_column(crit_key, m) for m in METHODS)
    missing = [c for c in required if c not in df_all.columns]
    if missing:
        raise ValueError(f"{DETAIL_CSV.name} is missing columns: {', '.join(missing)}")

    return df_all, df_all[df_all["status"] == "ranked"].copy()


def criterion_scores(df: pd.DataFrame, criterion: str) -> np.ndarray:
    """Return an (n, K) array of points for one criterion, columns in METHODS order.

    Rows with an invalid judgement or a missing value are dropped, so every
    column has the same length. Each remaining row must be a permutation of
    EXPECTED_POINTS; anything else means the detail CSV is corrupt.
    """
    valid = df[f"{criterion}_valid"].astype(str).str.lower() == "true"
    cols = [points_column(criterion, m) for m in METHODS]
    sub = df.loc[valid, cols].dropna()
    scores = sub.to_numpy(dtype=float)

    bad = ~np.all(np.sort(scores, axis=1)[:, ::-1] == EXPECTED_POINTS, axis=1)
    if bad.any():
        rows = ", ".join(str(i) for i in sub.index[bad][:10])
        raise ValueError(
            f"{criterion}: {int(bad.sum())} row(s) do not give points "
            f"{EXPECTED_POINTS} once each (CSV rows {rows})"
        )
    return scores


def kendalls_w(chi2: float, n: int, k: int) -> float:
    """Kendall's W from a Friedman chi² statistic.

    W = chi2 / (n * (k – 1))  where n = blocks, k = treatments.
    Interpretation: 0.1 weak, 0.3 moderate, 0.5 strong, 0.7 very strong.
    """
    if n == 0 or k <= 1:
        return float("nan")
    return chi2 / (n * (k - 1))


def concordance_label(W: float) -> str:
    if W >= 0.7:
        return "very strong"
    if W >= 0.5:
        return "strong"
    if W >= 0.3:
        return "moderate"
    return "weak"


def pair_label(a: Method, b: Method) -> str:
    return f"{a.label} vs {b.label}"


# ---------------------------------------------------------------------------
# Post-hoc pairwise Wilcoxon (Bonferroni-corrected)
# ---------------------------------------------------------------------------

class PostHocResult(NamedTuple):
    label: str
    w_stat: Optional[float]
    p_raw: Optional[float]
    p_adj: Optional[float]
    significant: bool
    direction: str
    a_higher: int
    b_higher: int
    note: str = ""


def pairwise_posthoc(scores: np.ndarray) -> list[PostHocResult]:
    """Run a Wilcoxon signed-rank test for every method pair, Bonferroni-corrected.

    The direction names the method ranked higher in more classes; with points
    that are a ranking, the median difference is often 0 and says little.
    """
    index = {m.slug: i for i, m in enumerate(METHODS)}
    results: list[PostHocResult] = []

    for a, b in METHOD_PAIRS:
        x = scores[:, index[a.slug]]
        y = scores[:, index[b.slug]]
        diff = x - y
        a_higher = int((diff > 0).sum())
        b_higher = int((diff < 0).sum())
        label = pair_label(a, b)

        if np.all(diff == 0.0):
            results.append(PostHocResult(
                label=label, w_stat=None, p_raw=None, p_adj=None,
                significant=False, direction="tie", a_higher=a_higher,
                b_higher=b_higher, note="all differences = 0",
            ))
            continue

        try:
            w_stat, p_raw = wilcoxon(x, y, alternative="two-sided")
        except ValueError as exc:
            results.append(PostHocResult(
                label=label, w_stat=None, p_raw=None, p_adj=None,
                significant=False, direction="n/a", a_higher=a_higher,
                b_higher=b_higher, note=str(exc),
            ))
            continue

        p_adj = min(p_raw * BONFERRONI_PAIRS, 1.0)
        if a_higher > b_higher:
            direction = f"{a.label} > {b.label}"
        elif b_higher > a_higher:
            direction = f"{b.label} > {a.label}"
        else:
            direction = "tie"

        results.append(PostHocResult(
            label=label, w_stat=float(w_stat), p_raw=float(p_raw), p_adj=float(p_adj),
            significant=p_adj < ALPHA, direction=direction,
            a_higher=a_higher, b_higher=b_higher,
        ))

    return results


# ---------------------------------------------------------------------------
# Formatting utilities
# ---------------------------------------------------------------------------

SEP_HEAVY = "=" * 72
SEP_LIGHT = "-" * 72


def fmt_section_header(title: str) -> str:
    return f"\n{SEP_HEAVY}\n{title}\n{SEP_HEAVY}"


def fmt_posthoc_block(results: list[PostHocResult]) -> list[str]:
    lines = [
        "",
        f"  Post-hoc pairwise Wilcoxon signed-rank tests"
        f" (Bonferroni over {BONFERRONI_PAIRS} pairs, α_adj = {ALPHA_ADJ:.4f}):",
        f"  {'Comparison':<{PAIR_WIDTH}} {'W-stat':>10} {'p (raw)':>12} "
        f"{'p (adj)':>12} {'Sig?':>6} {'A>B':>5} {'B>A':>5}  Direction",
        f"  {'-' * (PAIR_WIDTH + 75)}",
    ]
    for r in results:
        if r.note:
            lines.append(f"  {r.label:<{PAIR_WIDTH}}  [skipped — {r.note}]")
        else:
            sig_flag = "YES *" if r.significant else "no"
            lines.append(
                f"  {r.label:<{PAIR_WIDTH}} {r.w_stat:>10.2f} {r.p_raw:>12.3e} "
                f"{r.p_adj:>12.3e} {sig_flag:>6} {r.a_higher:>5d} {r.b_higher:>5d}"
                f"  {r.direction}"
            )
    lines.append("  (A>B / B>A = classes where the first / second method of the pair ranked higher)")
    return lines


def method_list() -> str:
    return ", ".join(m.label for m in METHODS)


# ---------------------------------------------------------------------------
# PART 1 — Per-criterion Friedman tests (class files as blocks)
# ---------------------------------------------------------------------------

def run_part1(df: pd.DataFrame) -> tuple[str, list[dict]]:
    """One Friedman test per criterion; class files are blocks."""
    lines: list[str] = [
        fmt_section_header("PART 1: PER-CRITERION FRIEDMAN TESTS  (class files as blocks)"),
        "",
        "  Test design : one Friedman test per criterion",
        "  Blocks      : individual class files with a valid judgement for the criterion",
        f"  Treatments  : k = {K} methods  ({method_list()})",
        "  Scoring     : 1st = 4 pts | 2nd = 3 pts | 3rd = 2 pts | 4th = 1 pt",
        f"  H0          : no significant difference among the {K} methods",
        f"  α           : {ALPHA}  |  post-hoc Bonferroni α_adj = {ALPHA_ADJ:.4f}",
    ]

    summary_rows: list[dict] = []

    for crit_key, crit_label in CRITERIA:
        scores = criterion_scores(df, crit_key)
        n = scores.shape[0]

        chi2, pval = friedmanchisquare(*scores.T)
        W = kendalls_w(chi2, n, K)

        means = {m.label: float(scores[:, i].mean()) for i, m in enumerate(METHODS)}
        best = max(means, key=means.get)
        sig = pval < ALPHA

        lines += [
            "",
            SEP_LIGHT,
            f"  Criterion      : {crit_label}",
            f"  N (blocks)     : {n} class files",
            f"  Friedman chi²  = {chi2:.4f}  (df = {K - 1})",
            f"  p-value        = {pval:.6g}  "
            + ("  < 0.05  →  REJECT H0 *" if sig else "  >= 0.05  →  fail to reject H0"),
            f"  Kendall's W    = {W:.4f}  ({concordance_label(W)} concordance)",
            f"  Significant    : {'YES' if sig else 'NO'}",
            "",
            "  Mean points per method (higher = better ranked; 4 = always 1st):",
        ]

        for m in METHODS:
            marker = "  <-- best" if m.label == best else ""
            lines.append(f"    {m.label:<{NAME_WIDTH}} : {means[m.label]:.4f}{marker}")

        if sig and n >= MIN_POSTHOC_BLOCKS:
            lines.extend(fmt_posthoc_block(pairwise_posthoc(scores)))
        elif sig:
            lines.append(
                f"\n  Post-hoc tests : not conducted (n = {n} < {MIN_POSTHOC_BLOCKS} blocks)"
            )
        else:
            lines.append(
                "\n  Post-hoc tests : not conducted (overall Friedman not significant)"
            )

        summary_rows.append({
            "criterion":   crit_label,
            "n":           n,
            "chi2":        chi2,
            "p_value":     pval,
            "kendalls_w":  W,
            "significant": sig,
            "best":        best,
        })

    return "\n".join(lines), summary_rows


# ---------------------------------------------------------------------------
# PART 2 — Cross-criteria Friedman test (criteria as blocks)
# ---------------------------------------------------------------------------

def run_part2(df: pd.DataFrame) -> str:
    """One Friedman test with the 5 criteria as blocks and the methods as treatments.

    Each cell in the 5 × K matrix is the mean points of a method on that
    criterion, averaged across the class files valid for that criterion.
    """
    criteria_names = ", ".join(label for _, label in CRITERIA)
    lines: list[str] = [
        fmt_section_header("PART 2: CROSS-CRITERIA FRIEDMAN TEST  (criteria as blocks)"),
        "",
        "  Test design : one Friedman test across all criteria",
        f"  Blocks      : n = {len(CRITERIA)} criteria  ({criteria_names})",
        f"  Treatments  : k = {K} methods  ({method_list()})",
        "  Cell value  : mean points of method on criterion across all class files",
        "  H0          : rank ordering of methods does not differ across criteria",
        f"  α           : {ALPHA}",
        "",
    ]

    mean_matrix = np.array(
        [criterion_scores(df, crit_key).mean(axis=0) for crit_key, _ in CRITERIA]
    )

    col_width = max(NAME_WIDTH, 10)
    header = f"  {'Criterion':<16s}" + "".join(f" {m.label:>{col_width}s}" for m in METHODS)
    rule = f"  {'-' * (len(header) - 2)}"
    lines += ["  Mean points matrix:", header, rule]
    for i, (_, crit_label) in enumerate(CRITERIA):
        row = mean_matrix[i]
        best_col = int(np.argmax(row))
        cells = "".join(
            f" {row[j]:>{col_width - 2}.4f}{' *' if j == best_col else '  '}"
            for j in range(K)
        )
        lines.append(f"  {crit_label:<16s}{cells}")
    col_means = mean_matrix.mean(axis=0)
    lines += [
        rule,
        f"  {'COLUMN MEAN':<16s}" + "".join(f" {v:>{col_width - 2}.4f}  " for v in col_means),
        "  (* = best method for that criterion)",
        "",
    ]

    n = len(CRITERIA)
    chi2, pval = friedmanchisquare(*mean_matrix.T)
    W = kendalls_w(chi2, n, K)
    sig = pval < ALPHA

    lines += [
        f"  Friedman chi²  = {chi2:.4f}  (df = {K - 1})",
        f"  p-value        = {pval:.6g}  "
        + ("  < 0.05  →  REJECT H0 *" if sig else "  >= 0.05  →  fail to reject H0"),
        f"  Kendall's W    = {W:.4f}  ({concordance_label(W)} concordance)",
        f"  Significant    : {'YES' if sig else 'NO'}",
    ]

    if sig:
        index = {m.slug: i for i, m in enumerate(METHODS)}
        lines += [
            "",
            f"  Note: n = {n} blocks — too few for a Wilcoxon signed-rank test.",
            "  Pairwise mean-points differences (descriptive):",
            f"  {'Comparison':<{PAIR_WIDTH}} {'Mean diff':>10}  Direction",
            f"  {'-' * (PAIR_WIDTH + 30)}",
        ]
        for a, b in METHOD_PAIRS:
            md = float((mean_matrix[:, index[a.slug]] - mean_matrix[:, index[b.slug]]).mean())
            if md > 0:
                direction = f"{a.label} > {b.label}"
            elif md < 0:
                direction = f"{b.label} > {a.label}"
            else:
                direction = "tie"
            lines.append(f"  {pair_label(a, b):<{PAIR_WIDTH}} {md:>+10.4f}  {direction}")
    else:
        lines.append(
            "\n  Post-hoc : not conducted (overall Friedman not significant)"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Summary table (PART 1 results)
# ---------------------------------------------------------------------------

def run_summary_table(summary_rows: list[dict]) -> str:
    lines: list[str] = [
        fmt_section_header("SUMMARY TABLE — PER-CRITERION FRIEDMAN RESULTS"),
        "",
        f"  {'Criterion':<16s} {'n':>5s} {'chi²':>10s} {'p-value':>12s} "
        f"{'Kendall W':>10s} {'Sig?':>5s}  Best Method",
        f"  {'-' * 80}",
    ]
    for r in summary_rows:
        sig = "YES *" if r["significant"] else "no"
        lines.append(
            f"  {r['criterion']:<16s} {r['n']:>5d} {r['chi2']:>10.4f} "
            f"{r['p_value']:>12.4g} {r['kendalls_w']:>10.4f} {sig:>5s}  {r['best']}"
        )
    lines += [
        "",
        f"  α = {ALPHA}  |  post-hoc Bonferroni α_adj = {ALPHA_ADJ:.4f}  |  df = {K - 1}",
        "  Kendall's W : 0.1 weak | 0.3 moderate | 0.5 strong | 0.7 very strong",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def write_block(section: str) -> None:
    """Write the section into results.txt, replacing any previous block."""
    existing = RESULTS_TXT.read_text(encoding="utf-8") if RESULTS_TXT.exists() else ""
    start = existing.find(BLOCK_START)
    end = existing.find(BLOCK_END)
    if start != -1 and end != -1:
        updated = existing[:start] + section + existing[end + len(BLOCK_END) + 1:]
    else:
        separator = "" if not existing or existing.endswith("\n\n") else "\n\n"
        updated = existing + separator + section
    RESULTS_TXT.write_text(updated, encoding="utf-8")


def main() -> None:
    df_all, df = load_rankings()
    if df.empty:
        raise ValueError(f"{DETAIL_CSV.name} has no rows with status 'ranked'.")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    title = " vs ".join(m.label for m in METHODS)

    header = "\n".join([
        "",
        SEP_HEAVY,
        f"FRIEDMAN TEST ANALYSIS — {title}",
        SEP_HEAVY,
        f"  Run date   : {timestamp}",
        f"  Data file  : {DETAIL_CSV.name}",
        f"  Total rows : {len(df_all)}  |  Rows used (status=ranked): {len(df)}",
        "",
        f"  Methods    :  {method_list()}",
        "  Scoring    :  1st place = 4 pts  |  2nd = 3 pts  |  3rd = 2 pts  |  4th = 1 pt",
        f"  Alpha      :  {ALPHA}",
        f"  Post-hoc   :  Wilcoxon signed-rank, Bonferroni α_adj = {ALPHA_ADJ:.4f}",
        "",
        "  H0 (each Friedman test): no significant difference in points",
        f"  among the {K} methods under the given criterion / criteria.",
    ])

    print(header)

    part1_text, summary_rows = run_part1(df)
    print(part1_text)

    part2_text = run_part2(df)
    print(part2_text)

    summary_text = run_summary_table(summary_rows)
    print(summary_text)

    # ------------------------------------------------------------------
    # Write results into a delimited block in results.txt, replacing any
    # previous block so the file never holds two versions of these tests.
    # ------------------------------------------------------------------
    write_block(
        BLOCK_START + "\n"
        + header + "\n"
        + part1_text + "\n"
        + part2_text + "\n"
        + summary_text + "\n"
        + BLOCK_END + "\n"
    )

    print(f"\nResults successfully written to: {RESULTS_TXT}")


if __name__ == "__main__":
    main()
