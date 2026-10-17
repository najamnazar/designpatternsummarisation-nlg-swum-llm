#!/usr/bin/env python3
"""
rank_summaries_four_way.py - Four-way multi-criteria ranking of DPS summaries

Replaces rank_summaries.py (A vs B vs C) and rank_b1_b5_iterations.py (A vs B1..B5 vs C),
both now in archive/. An LLM judge ranks four summaries of every class against the human
(gold) summary, in a single iteration, on 5 quality criteria:
1. Accuracy
2. Conciseness
3. Adequacy
4. Code Context
5. Design Pattern Recognition

The four methods, shown to the judge in this fixed order:
    Summary 1 = DPS_NLG            output/summary-output/nlg_summaries.csv
    Summary 2 = DPS_LLM (GPT)      output/summary-output/llm_summaries_gpt.csv
    Summary 3 = DPS_LLM (Mistral)  output/summary-output/llm_summaries_mistral.csv
    Summary 4 = DPS_SWUM           output/summary-output/swum_summaries.csv

Because all four are ranked in one judgement, the two LLMs are compared with each other
as well as with NLG and SWUM. Points per criterion: 1st = 4, 2nd = 3, 3rd = 2, 4th = 1.

Configuration comes from .env; the code has no fallback values for it:
    OPENROUTER_API_KEY         API key
    RANK_SUMMARIES_MODEL       judge model id
    RANK_SUMMARIES_MAX_TOKENS  max completion tokens per judge call
    RANK_SUMMARIES_API_URL     optional endpoint; OPENROUTER_API_URL is used when it is empty
Prompts: resources/prompts.json, section "summary_ranking" ({summary_a}..{summary_d} = Summary 1..4).

Outputs (evaluation-results/):
    four_way_ranking_detail.csv        one row per class: each method's rank and points per
                                       criterion, plus the raw judge output
    four_way_ranking_summary.csv       per criterion and method: 1st..4th counts, mean rank, mean points
    four_way_ranking_head_to_head.csv  per criterion and method pair: how often each was ranked higher
    four_way_ranking_results.csv       every figure of the results.txt block, one row each
                                       (section, criterion, method, metric, value)
    results.txt                        delimited block (replaced on re-run), plain-text and LaTeX tables

A new run calls the judge for every class and overwrites the files above. To change only the
scoring of a finished run, use:
    --reparse        score the replies saved in four_way_ranking_detail.csv again with the current
                     parser (no judge calls) and rewrite all outputs
    --retry-invalid  as --reparse, then ask the judge once more for each judgement still invalid
Both keep the detail file as it was before the first re-score in
four_way_ranking_detail_before_rescore.csv.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv

from project_paths import HUMAN_SUMMARIES_CSV, REPO_ROOT

SUMMARY_DIR = REPO_ROOT / "output" / "summary-output"
EVAL_DIR = REPO_ROOT / "evaluation-results"
PROMPTS_JSON = REPO_ROOT / "resources" / "prompts.json"
PROMPT_SECTION = "summary_ranking"
# Placeholders in the prompt templates, in the order the summaries are shown (Summary 1..4).
PROMPT_PLACEHOLDERS = ["summary_a", "summary_b", "summary_c", "summary_d"]

DETAIL_CSV = EVAL_DIR / "four_way_ranking_detail.csv"
SUMMARY_CSV = EVAL_DIR / "four_way_ranking_summary.csv"
HEAD_TO_HEAD_CSV = EVAL_DIR / "four_way_ranking_head_to_head.csv"
RESULTS_CSV = EVAL_DIR / "four_way_ranking_results.csv"
DETAIL_BACKUP_CSV = EVAL_DIR / "four_way_ranking_detail_before_rescore.csv"
RESULTS_TXT = EVAL_DIR / "results.txt"


@dataclass(frozen=True)
class Method:
    label: str     # name used in reports
    slug: str      # column suffix in the CSVs
    csv_name: str  # file in output/summary-output/


# List order is the position shown to the judge (Summary 1..4). It is fixed, as the
# A/B/C order was in the earlier runs, and recorded in the results.txt block.
METHODS: List[Method] = [
    Method("DPS_NLG", "nlg", "nlg_summaries.csv"),
    Method("DPS_LLM (GPT)", "gpt", "llm_summaries_gpt.csv"),
    Method("DPS_LLM (Mistral)", "mistral", "llm_summaries_mistral.csv"),
    Method("DPS_SWUM", "swum", "swum_summaries.csv"),
]

# (prompt key in prompts.json, title used in reports)
CRITERIA: List[Tuple[str, str]] = [
    ("accuracy", "ACCURACY"),
    ("conciseness", "CONCISENESS"),
    ("adequacy", "ADEQUACY"),
    ("code_context", "CODE CONTEXT"),
    ("design_patterns", "DESIGN PATTERN"),
]

N_METHODS = len(METHODS)
POSITIONS = ["1st", "2nd", "3rd", "4th"]

# Sampling temperature of the judge; 0 keeps judgements as repeatable as the API allows.
JUDGE_TEMPERATURE = 0.0
REQUEST_TIMEOUT_SECONDS = 45
MAX_ATTEMPTS = 3

# Marker delimiting this script's block inside results.txt, so that re-running
# replaces the previous results instead of stacking a second copy.
BLOCK_START = "<<<BEGIN FOUR-WAY SUMMARY RANKING>>>"
BLOCK_END = "<<<END FOUR-WAY SUMMARY RANKING>>>"


def points_for_rank(rank: int) -> int:
    """1st place earns N_METHODS points, last place earns 1."""
    return N_METHODS + 1 - rank


# --------------------------------------------------------------------------- matching keys

def _strip_extension(filename: str) -> str:
    """Remove common source-file extensions when building comparison keys."""
    if not isinstance(filename, str):
        return ""
    return re.sub(r"\.(java|txt|md)$", "", filename.strip(), flags=re.IGNORECASE)


def _normalize_component(value: str) -> str:
    """Normalize project or filename segments for reliable cross-file matching."""
    if not isinstance(value, str):
        return ""
    value = _strip_extension(value)
    value = value.lower().strip()
    value = re.sub(r"\s+", "", value)
    return re.sub(r"[^a-z0-9]", "", value)


def build_match_key(project: str, filename: str, folder: str = "") -> str:
    """Build a canonical match key that tolerates naming and formatting differences.

    The folder component is required to make the key unique: 20 files in the
    corpus share a (project, filename) pair with another file (Test.java appears
    7 times under AbdurRKhalid, Application.java 9 times under JamesZBL), so a
    project::filename key silently collapses them and the summaries compared for
    such a file can come from different source classes. The folder is the design
    pattern directory ("Folder Name") in the summary CSVs and the "Design Pattern"
    column in DPS_Human_Summaries.csv, which normalise to the same token.
    """
    return "::".join(
        (
            _normalize_component(project),
            _normalize_component(folder),
            _normalize_component(filename),
        )
    )


def assert_unique_match_keys(df: pd.DataFrame, label: str) -> None:
    """Fail loudly if a frame's match keys collapse two different files."""
    duplicated = df["match_key"].duplicated(keep=False)
    if duplicated.any():
        samples = sorted(df.loc[duplicated, "match_key"].unique())[:5]
        raise ValueError(
            f"{label}: match keys are not unique ({int(duplicated.sum())} rows). "
            f"Summaries for different files would be compared against each other. "
            f"Sample keys: {samples}"
        )


def _clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    return df


def load_method_summaries(method: Method) -> Dict[str, str]:
    """Return {match_key: summary} for one method's CSV."""
    path = SUMMARY_DIR / method.csv_name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; generate the {method.label} summaries first.")
    df = _normalise_columns(pd.read_csv(path))
    missing = {"project", "folder_name", "file_name", "summary"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)} in {path.name}")
    df["summary"] = df["summary"].map(_clean_text)
    df["match_key"] = df.apply(
        lambda row: build_match_key(row["project"], row["file_name"], row["folder_name"]), axis=1
    )
    assert_unique_match_keys(df, path.name)
    return dict(zip(df["match_key"], df["summary"]))


def load_human_summaries() -> pd.DataFrame:
    df = _normalise_columns(pd.read_csv(HUMAN_SUMMARIES_CSV))
    missing = {"project", "file_name", "design_pattern", "human_summary"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)} in {HUMAN_SUMMARIES_CSV.name}")
    df["human_summary"] = df["human_summary"].map(_clean_text)
    df["match_key"] = df.apply(
        lambda row: build_match_key(row["project"], row["file_name"], row["design_pattern"]), axis=1
    )
    assert_unique_match_keys(df, HUMAN_SUMMARIES_CSV.name)
    return df


# --------------------------------------------------------------------------- configuration

@dataclass(frozen=True)
class JudgeConfig:
    api_key: str
    api_url: str
    model: str
    max_tokens: int


def require_env(name: str) -> str:
    value = (os.getenv(name) or "").strip()
    if not value:
        raise ValueError(f"{name} is not set in .env")
    return value


def load_judge_config() -> JudgeConfig:
    """Read the judge settings from .env; nothing here has a built-in default."""
    # override=True: the values in .env win over variables already set in the shell.
    load_dotenv(REPO_ROOT / ".env", override=True)

    api_key = require_env("OPENROUTER_API_KEY")
    api_url = (os.getenv("RANK_SUMMARIES_API_URL") or "").strip() or require_env("OPENROUTER_API_URL")
    model = require_env("RANK_SUMMARIES_MODEL")

    max_tokens_raw = require_env("RANK_SUMMARIES_MAX_TOKENS")
    try:
        max_tokens = int(max_tokens_raw)
    except ValueError as exc:
        raise ValueError(f"RANK_SUMMARIES_MAX_TOKENS must be an integer, got {max_tokens_raw!r}") from exc
    if max_tokens <= 0:
        raise ValueError(f"RANK_SUMMARIES_MAX_TOKENS must be positive, got {max_tokens}")

    return JudgeConfig(api_key=api_key, api_url=api_url, model=model, max_tokens=max_tokens)


def load_prompts() -> Dict[str, str]:
    """Load the four-summary ranking templates, one per criterion."""
    if not PROMPTS_JSON.exists():
        raise FileNotFoundError(f"Prompt file not found: {PROMPTS_JSON}")
    data = json.loads(PROMPTS_JSON.read_text(encoding="utf-8"))
    section = data.get(PROMPT_SECTION) if isinstance(data, dict) else None
    if not isinstance(section, dict):
        raise ValueError(f"prompts.json is missing the {PROMPT_SECTION} section")
    missing = [key for key, _ in CRITERIA if not section.get(key)]
    if missing:
        raise ValueError(f"prompts.json section {PROMPT_SECTION} has no prompt for: {missing}")
    # Every template must show all four summaries; a three-summary prompt would silently drop one method.
    for key, _ in CRITERIA:
        absent = [p for p in ["human_summary", *PROMPT_PLACEHOLDERS] if "{" + p + "}" not in section[key]]
        if absent:
            raise ValueError(f"prompts.json {PROMPT_SECTION}.{key} is missing placeholders: {absent}")
    return {key: section[key] for key, _ in CRITERIA}


# --------------------------------------------------------------------------- judge

def parse_order(text: str) -> Optional[List[int]]:
    """Read the judge's best-to-worst order of summary numbers 1..N_METHODS.

    Accepts "2, 1, 4, 3", "Summary 2 > Summary 1 > ...", numbered lists such as
    "1. Summary 2\\n2. Summary 1 ..." or "1. 2\\n2. 1 ...", and rank-per-summary
    answers such as "Summary 1: 3, Summary 2: 1, ...". Anything that is not a
    permutation of 1..N_METHODS is rejected, so a malformed answer is recorded as
    invalid instead of biasing the points.
    """
    if not text:
        return None
    expected = list(range(1, N_METHODS + 1))
    digit = f"[1-{N_METHODS}]"

    def as_order(values: Sequence[str]) -> Optional[List[int]]:
        order = [int(v) for v in values]
        return order if sorted(order) == expected else None

    # Rank given per summary ("Summary 3: 1" = summary 3 is ranked 1st): invert to an order.
    pairs = re.findall(rf"summary\s*#?\s*({digit})\s*[:=]\s*(?:rank(?:ed)?\s*)?#?\s*({digit})(?!\d)",
                       text, flags=re.IGNORECASE)
    if len(pairs) == N_METHODS:
        summaries = [int(s) for s, _ in pairs]
        ranks = [int(r) for _, r in pairs]
        if sorted(summaries) == expected and sorted(ranks) == expected:
            return [s for _, s in sorted(zip(ranks, summaries))]

    # Numbered list, one place per line ("1. 3\n2. 2\n..." or "1) Summary 3\n..."): the list
    # markers are places 1..N in sequence and the item values are the summary numbers. Reading
    # every digit instead would mix the markers into the order ("1. 3\n2. 2" -> 1, 3, 2, 2).
    numbered = re.findall(rf"^\s*({digit})\s*[.):]\s*(?:summary\s*#?\s*)?({digit})(?!\d)",
                          text, flags=re.IGNORECASE | re.MULTILINE)
    if len(numbered) == N_METHODS and [int(p) for p, _ in numbered] == expected:
        return as_order([v for _, v in numbered])

    # Summary numbers in order, labelled ("Summary 2 > Summary 1").
    labelled = re.findall(rf"summary\s*#?\s*({digit})(?!\d)", text, flags=re.IGNORECASE)
    if len(labelled) >= N_METHODS:
        order = as_order(labelled[:N_METHODS])
        if order:
            return order

    # Bare summary numbers ("2, 1, 4, 3", "3\n2\n1\n4", "2. 3. 4. 1."): only when the reply is
    # exactly N digits and separators, so stray digits elsewhere cannot form a wrong order.
    separator = r"[\s,;.>\-]"
    bare = rf"{separator}*{digit}(?:{separator}+{digit}){{{N_METHODS - 1}}}{separator}*"
    if re.fullmatch(bare, text):
        return as_order(re.findall(digit, text))
    # The same bare ranking alone on the first line, followed by an explanation
    # ("3, 2, 4, 1\n\nExplanation ..."). A first line with anything else on it is not read.
    first_line = text.strip().splitlines()[0]
    if re.fullmatch(bare, first_line):
        return as_order(re.findall(digit, first_line))
    return None


class FourWayJudge:
    """Asks the judge model to rank the four summaries on one criterion."""

    def __init__(self, config: JudgeConfig, prompts: Dict[str, str]) -> None:
        self.config = config
        self.prompts = prompts
        self.session = requests.Session()

    def rank(self, criterion: str, human_summary: str, summaries: Sequence[str]) -> Tuple[Optional[List[int]], str]:
        """Return (order, raw_output); order lists summary numbers best first, or None if invalid."""
        prompt = self.prompts[criterion].format(
            human_summary=human_summary,
            **dict(zip(PROMPT_PLACEHOLDERS, summaries)),
        )
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": JUDGE_TEMPERATURE,
            "max_tokens": self.config.max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Optional[Exception] = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = self.session.post(
                    self.config.api_url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT_SECONDS
                )
                response.raise_for_status()
                content = (response.json()["choices"][0]["message"]["content"] or "").strip()
            except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as exc:
                last_error = exc
                if attempt < MAX_ATTEMPTS:
                    time.sleep(2 ** attempt)
                continue
            # A reply that cannot be parsed is not retried: re-asking until the answer
            # parses would favour whichever answers the model happens to format cleanly.
            return parse_order(content), content
        return None, f"ERROR: {last_error}"


Judgement = Tuple[Optional[List[int]], str]  # (order of summary numbers best first or None, raw reply)


def rank_class(judge: FourWayJudge, human_summary: str, summaries: Sequence[str]) -> Dict[str, object]:
    """Rank one class on every criterion and return its detail-row fields."""
    judgements = {criterion: judge.rank(criterion, human_summary, summaries) for criterion, _ in CRITERIA}
    return score_judgements(judgements)


def score_judgements(judgements: Dict[str, Judgement], verbose: bool = True) -> Dict[str, object]:
    """Turn each criterion's judgement into the detail-row fields: ranks, points, totals and winner."""
    record: Dict[str, object] = {}
    totals = {m.slug: 0 for m in METHODS}
    valid = 0

    for criterion, _ in CRITERIA:
        order, raw = judgements[criterion]
        record[f"{criterion}_valid"] = order is not None
        if order is None:
            for position in POSITIONS:
                record[f"{criterion}_{position}"] = ""
            for m in METHODS:
                record[f"{criterion}_rank_{m.slug}"] = None
                record[f"{criterion}_points_{m.slug}"] = None
            if verbose:
                print(f"    {criterion}: invalid ({raw[:60]!r})")
        else:
            valid += 1
            for position, summary_number in enumerate(order, start=1):
                method = METHODS[summary_number - 1]
                points = points_for_rank(position)
                record[f"{criterion}_{POSITIONS[position - 1]}"] = method.slug
                record[f"{criterion}_rank_{method.slug}"] = position
                record[f"{criterion}_points_{method.slug}"] = points
                totals[method.slug] += points
            if verbose:
                print(f"    {criterion}: " + " > ".join(METHODS[n - 1].slug for n in order))
        record[f"{criterion}_raw"] = raw

    record["criteria_ranked"] = valid
    for m in METHODS:
        record[f"total_points_{m.slug}"] = totals[m.slug] if valid else None
        # Averaged over the criteria actually judged, so an invalid judgement does not count as 0 points.
        record[f"avg_points_{m.slug}"] = round(totals[m.slug] / valid, 2) if valid else None
    if valid:
        best = max(totals.values())
        record["winner"] = ", ".join(m.label for m in METHODS if totals[m.slug] == best)
    else:
        record["winner"] = ""
    return record


def _is_true(value: object) -> bool:
    """Read a boolean column back from CSV, where it may be True, 'True' or empty."""
    return str(value).strip().lower() == "true"


def _ranks_of(record: Dict[str, object], criterion: str) -> Optional[Tuple[object, ...]]:
    """The method ranks recorded for one criterion, or None if that judgement was invalid."""
    if not _is_true(record.get(f"{criterion}_valid")):
        return None
    return tuple(int(float(record[f"{criterion}_rank_{m.slug}"])) for m in METHODS)


def rescore_detail(detail: pd.DataFrame, judge: Optional[FourWayJudge] = None) -> pd.DataFrame:
    """Score the saved judge replies again with the current parser, without calling the judge.

    With a judge, each judgement that is still invalid after re-parsing is asked once more and
    marked in a "<criterion>_retried" column. Every reply that does parse is kept as it is.
    """
    records = detail.to_dict("records")
    stats = {key: {"valid_before": 0, "valid_after": 0, "changed": 0, "retried": 0} for key, _ in CRITERIA}

    for record in records:
        if record.get("status") != "ranked":
            continue
        judgements: Dict[str, Judgement] = {}
        before = {}
        for criterion, _ in CRITERIA:
            before[criterion] = _ranks_of(record, criterion)
            raw = _clean_text(record.get(f"{criterion}_raw"))
            # A failed call is stored as "ERROR: ..."; it is not a reply and must not be parsed.
            order = None if raw.startswith("ERROR:") else parse_order(raw)
            retried = _is_true(record.get(f"{criterion}_retried"))
            if order is None and judge is not None:
                summaries = [_clean_text(record.get(f"summary_{m.slug}")) for m in METHODS]
                print(f"  re-asking {criterion} for {record['project']}/{record['design_pattern']}/{record['file']}")
                order, raw = judge.rank(criterion, _clean_text(record.get("human_summary")), summaries)
                retried = True
                stats[criterion]["retried"] += 1
            record[f"{criterion}_retried"] = retried
            judgements[criterion] = (order, raw)
        record.update(score_judgements(judgements, verbose=False))

        for criterion, _ in CRITERIA:
            after = _ranks_of(record, criterion)
            stats[criterion]["valid_before"] += before[criterion] is not None
            stats[criterion]["valid_after"] += after is not None
            stats[criterion]["changed"] += before[criterion] is not None and after is not None \
                and before[criterion] != after

    print(f"{'Criterion':<16}{'valid before':>14}{'valid after':>13}{'re-asked':>10}{'ranks changed':>15}")
    for key, _ in CRITERIA:
        s = stats[key]
        print(f"{key:<16}{s['valid_before']:>14}{s['valid_after']:>13}{s['retried']:>10}{s['changed']:>15}")
    return pd.DataFrame(records)


# --------------------------------------------------------------------------- aggregation

def _ranks(ranked: pd.DataFrame, criterion: str, method: Method) -> pd.Series:
    column = f"{criterion}_rank_{method.slug}"
    if column not in ranked.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(ranked[column], errors="coerce").dropna()


def _rank_row(criterion: str, method: Method, ranks: pd.Series) -> Dict[str, object]:
    row: Dict[str, object] = {"criterion": criterion, "method": method.label, "n_valid": int(len(ranks))}
    for i, position in enumerate(POSITIONS, start=1):
        row[position] = int((ranks == i).sum())
    row["mean_rank"] = round(float(ranks.mean()), 3) if len(ranks) else None
    row["mean_points"] = round(float((N_METHODS + 1 - ranks).mean()), 3) if len(ranks) else None
    return row


def build_summary(detail: pd.DataFrame) -> pd.DataFrame:
    """Per criterion and method: 1st..4th counts, mean rank and mean points; plus an overall row."""
    ranked = detail[detail["status"] == "ranked"]
    rows = []
    for criterion, _ in CRITERIA:
        for m in METHODS:
            rows.append(_rank_row(criterion, m, _ranks(ranked, criterion, m)))
    for m in METHODS:
        all_ranks = pd.concat([_ranks(ranked, criterion, m) for criterion, _ in CRITERIA], ignore_index=True)
        rows.append(_rank_row("overall", m, all_ranks))
    return pd.DataFrame(rows)


def build_head_to_head(detail: pd.DataFrame) -> pd.DataFrame:
    """For every method pair: classes where each was ranked higher.

    Per criterion this compares ranks; the overall row compares total points over
    the criteria judged for that class, where ties are possible.
    """
    ranked = detail[detail["status"] == "ranked"]
    rows = []
    for a, b in combinations(METHODS, 2):
        for criterion, _ in CRITERIA:
            col_a, col_b = f"{criterion}_rank_{a.slug}", f"{criterion}_rank_{b.slug}"
            if col_a not in ranked.columns:
                pair = pd.DataFrame(columns=["a", "b"])
            else:
                pair = pd.DataFrame({
                    "a": pd.to_numeric(ranked[col_a], errors="coerce"),
                    "b": pd.to_numeric(ranked[col_b], errors="coerce"),
                }).dropna()
            rows.append({
                "criterion": criterion, "method_a": a.label, "method_b": b.label, "n": int(len(pair)),
                "a_higher": int((pair["a"] < pair["b"]).sum()), "b_higher": int((pair["a"] > pair["b"]).sum()),
                "ties": 0,
            })
        col_a, col_b = f"total_points_{a.slug}", f"total_points_{b.slug}"
        if col_a not in ranked.columns:
            pair = pd.DataFrame(columns=["a", "b"])
        else:
            pair = pd.DataFrame({
                "a": pd.to_numeric(ranked[col_a], errors="coerce"),
                "b": pd.to_numeric(ranked[col_b], errors="coerce"),
            }).dropna()
        rows.append({
            "criterion": "overall", "method_a": a.label, "method_b": b.label, "n": int(len(pair)),
            "a_higher": int((pair["a"] > pair["b"]).sum()), "b_higher": int((pair["a"] < pair["b"]).sum()),
            "ties": int((pair["a"] == pair["b"]).sum()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- report

def _latex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def _judged(ranked: pd.DataFrame) -> pd.DataFrame:
    """Ranked classes with at least one valid judgement, i.e. those that have a winner."""
    if "criteria_ranked" not in ranked.columns:
        return ranked.iloc[0:0]
    return ranked[pd.to_numeric(ranked["criteria_ranked"], errors="coerce") > 0]


def _retried_counts(ranked: pd.DataFrame) -> Dict[str, int]:
    """Per criterion, how many judgements were asked again by --retry-invalid."""
    return {
        key: int(ranked[f"{key}_retried"].map(_is_true).sum()) if f"{key}_retried" in ranked.columns else 0
        for key, _ in CRITERIA
    }


def format_results_section(detail: pd.DataFrame, summary: pd.DataFrame, head_to_head: pd.DataFrame,
                           config: JudgeConfig, generated_at: str) -> str:
    ranked = detail[detail["status"] == "ranked"]
    skipped = detail[detail["status"] != "ranked"]
    titles = dict(CRITERIA)
    by_key = {(r["criterion"], r["method"]): r for r in summary.to_dict("records")}

    def counts(criterion: str, method: Method) -> str:
        r = by_key[(criterion, method.label)]
        return "/".join(str(r[position]) for position in POSITIONS)

    def n_valid(criterion: str) -> int:
        return int(by_key[(criterion, METHODS[0].label)]["n_valid"])

    def mean_points(criterion: str, method: Method) -> str:
        value = by_key[(criterion, method.label)]["mean_points"]
        return f"{value:.2f}" if value is not None and not pd.isna(value) else "-"

    criterion_width = max(len(t) for _, t in CRITERIA) + 2
    col_width = max(max(len(m.label) for m in METHODS), len("000/000/000/000")) + 2

    lines = [
        BLOCK_START,
        "=" * 70,
        "FOUR-WAY MULTI-CRITERIA SUMMARY RANKING (LLM JUDGE, SINGLE ITERATION)",
        "=" * 70,
        f"Generated at: {generated_at}",
        f"Judge model (RANK_SUMMARIES_MODEL): {config.model}",
        f"Max tokens (RANK_SUMMARIES_MAX_TOKENS): {config.max_tokens}",
        f"Prompts: resources/prompts.json [{PROMPT_SECTION}]",
        "Order shown to the judge: " + ", ".join(f"{i} = {m.label}" for i, m in enumerate(METHODS, start=1)),
        "Points per criterion: " + ", ".join(f"{p} = {points_for_rank(i)}" for i, p in enumerate(POSITIONS, start=1)),
        f"Classes ranked: {len(ranked)}; skipped (missing summaries): {len(skipped)}",
        "Invalid judgements (unparseable reply or failed call): "
        + ", ".join(f"{key} = {len(ranked) - n_valid(key)}" for key, _ in CRITERIA),
    ]
    retried = _retried_counts(ranked)
    if any(retried.values()):
        lines.append("Judgements asked again after an invalid reply (--retry-invalid): "
                     + ", ".join(f"{key} = {count}" for key, count in retried.items()))
    lines += [
        "",
        "Rank counts per criterion (1st/2nd/3rd/4th):",
        f"{'Criterion':<{criterion_width}}" + "".join(f"{m.label:<{col_width}}" for m in METHODS) + "n",
    ]
    for key, title in CRITERIA:
        lines.append(f"{title:<{criterion_width}}" + "".join(f"{counts(key, m):<{col_width}}" for m in METHODS)
                     + str(n_valid(key)))

    lines += [
        "",
        f"Mean points per criterion ({N_METHODS} = always 1st, 1 = always last):",
        f"{'Criterion':<{criterion_width}}" + "".join(f"{m.label:<{col_width}}" for m in METHODS),
    ]
    for key, title in CRITERIA + [("overall", "OVERALL")]:
        lines.append(f"{title:<{criterion_width}}" + "".join(f"{mean_points(key, m):<{col_width}}" for m in METHODS))

    lines += ["", "Winner distribution (highest total points per class; ties listed together):"]
    judged = _judged(ranked)
    if judged.empty:
        lines.append("  (no judged classes)")
    else:
        for winner, count in judged["winner"].value_counts().items():
            lines.append(f"  {winner}: {count}/{len(judged)} ({100 * count / len(judged):.1f}%)")

    gpt, mistral = METHODS[1], METHODS[2]
    lines += ["", f"Head-to-head, {gpt.label} vs {mistral.label} (classes where each was ranked higher):"]
    h2h = head_to_head[(head_to_head["method_a"] == gpt.label) & (head_to_head["method_b"] == mistral.label)]
    for r in h2h.itertuples(index=False):
        title = titles.get(r.criterion, "OVERALL (total points)")
        tie_text = f", ties {r.ties}" if r.criterion == "overall" else ""
        lines.append(f"  {title:<{criterion_width + 10}}{gpt.label} {r.a_higher}, {mistral.label} {r.b_higher}"
                     f"{tie_text} (n = {r.n})")

    lines += [
        "",
        "LaTeX (rank counts):",
        r"\begin{table}[htb]",
        r"  \caption{Rank counts per criterion (1st/2nd/3rd/4th) from the four-way LLM-judge ranking}",
        r"  \label{tab:fourwayrankcounts}",
        r"  \centering",
        r"  \begin{tabular}{l" + "l" * N_METHODS + "r}",
        r"    \toprule",
        "    Metric & " + " & ".join(_latex_escape(m.label) for m in METHODS) + r" & n \\",
        r"    \midrule",
    ]
    for key, title in CRITERIA:
        lines.append(f"    {title} & " + " & ".join(counts(key, m).replace("/", " / ") for m in METHODS)
                     + f" & {n_valid(key)} \\\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]

    lines += [
        "LaTeX (mean points):",
        r"\begin{table}[htb]",
        rf"  \caption{{Mean points per criterion from the four-way LLM-judge ranking ({N_METHODS} = always first, 1 = always last)}}",
        r"  \label{tab:fourwaymeanpoints}",
        r"  \centering",
        r"  \begin{tabular}{l" + "r" * N_METHODS + "}",
        r"    \toprule",
        "    Metric & " + " & ".join(_latex_escape(m.label) for m in METHODS) + r" \\",
        r"    \midrule",
    ]
    for key, title in CRITERIA:
        lines.append(f"    {title} & " + " & ".join(mean_points(key, m) for m in METHODS) + r" \\")
    lines.append(r"    \midrule")
    lines.append("    OVERALL & " + " & ".join(mean_points("overall", m) for m in METHODS) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", BLOCK_END]
    return "\n".join(lines) + "\n"


def build_results_table(detail: pd.DataFrame, summary: pd.DataFrame, head_to_head: pd.DataFrame,
                        config: JudgeConfig, generated_at: str) -> pd.DataFrame:
    """Every figure of the results.txt block as one long table (section, criterion, method, metric, value)."""
    ranked = detail[detail["status"] == "ranked"]
    rows: List[Dict[str, object]] = []

    def add(section: str, metric: str, value: object, criterion: str = "", method: str = "") -> None:
        rows.append({"section": section, "criterion": criterion, "method": method, "metric": metric, "value": value})

    for metric, value in [
        ("generated_at", generated_at),
        ("judge_model", config.model),
        ("max_tokens", config.max_tokens),
        ("prompts", f"resources/prompts.json [{PROMPT_SECTION}]"),
        ("order_shown_to_judge", ", ".join(f"{i} = {m.label}" for i, m in enumerate(METHODS, start=1))),
        ("points_per_criterion", ", ".join(f"{p} = {points_for_rank(i)}" for i, p in enumerate(POSITIONS, start=1))),
        ("classes_ranked", len(ranked)),
        ("classes_skipped", len(detail) - len(ranked)),
    ]:
        add("run", metric, value)

    retried = _retried_counts(ranked)
    by_key = {(r["criterion"], r["method"]): r for r in summary.to_dict("records")}
    for key, _ in CRITERIA:
        n_valid = int(by_key[(key, METHODS[0].label)]["n_valid"])
        add("judgements", "invalid", len(ranked) - n_valid, key)
        add("judgements", "retried", retried[key], key)
        add("judgements", "valid", n_valid, key)

    for key in [k for k, _ in CRITERIA] + ["overall"]:
        for m in METHODS:
            r = by_key[(key, m.label)]
            if key != "overall":
                for position in POSITIONS:
                    add("rank_counts", position, r[position], key, m.label)
                add("rank_counts", "n_valid", r["n_valid"], key, m.label)
            add("mean_points", "mean_points", r["mean_points"], key, m.label)
            add("mean_points", "mean_rank", r["mean_rank"], key, m.label)

    judged = _judged(ranked)
    for winner, count in judged["winner"].value_counts().items():
        add("winner_distribution", "classes", int(count), method=winner)
        add("winner_distribution", "percent", round(100 * count / len(judged), 1), method=winner)

    gpt, mistral = METHODS[1], METHODS[2]
    h2h = head_to_head[(head_to_head["method_a"] == gpt.label) & (head_to_head["method_b"] == mistral.label)]
    for r in h2h.itertuples(index=False):
        add("head_to_head_gpt_vs_mistral", "ranked_higher", r.a_higher, r.criterion, gpt.label)
        add("head_to_head_gpt_vs_mistral", "ranked_higher", r.b_higher, r.criterion, mistral.label)
        add("head_to_head_gpt_vs_mistral", "ties", r.ties, r.criterion)
        add("head_to_head_gpt_vs_mistral", "n", r.n, r.criterion)
    return pd.DataFrame(rows, columns=["section", "criterion", "method", "metric", "value"])


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


# --------------------------------------------------------------------------- pipeline

def run(limit: Optional[int]) -> None:
    config = load_judge_config()
    prompts = load_prompts()
    human = load_human_summaries()
    method_maps = {m.slug: load_method_summaries(m) for m in METHODS}
    if limit is not None:
        human = human.head(limit)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    judge = FourWayJudge(config, prompts)

    total = len(human)
    print("=" * 70)
    print("FOUR-WAY MULTI-CRITERIA SUMMARY RANKING")
    print("=" * 70)
    print(f"Judge model: {config.model} (max_tokens={config.max_tokens})")
    print("Order shown to the judge: " + ", ".join(f"{i} = {m.label}" for i, m in enumerate(METHODS, start=1)))
    print(f"Classes: {total}; up to {total * len(CRITERIA)} judge calls")

    records: List[Dict[str, object]] = []
    for idx, row in enumerate(human.itertuples(index=False), start=1):
        record: Dict[str, object] = {
            "project": row.project,
            "design_pattern": row.design_pattern,
            "file": row.file_name,
            "match_key": row.match_key,
            "human_summary": row.human_summary,
        }
        summaries = [method_maps[m.slug].get(row.match_key, "") for m in METHODS]
        for m, summary in zip(METHODS, summaries):
            record[f"summary_{m.slug}"] = summary

        missing = ([] if row.human_summary else ["Human"]) + [m.label for m, s in zip(METHODS, summaries) if not s]
        if missing:
            record.update(status="missing", missing_methods=", ".join(missing))
            print(f"[{idx}/{total}] Skipping {row.project}/{row.design_pattern}/{row.file_name} "
                  f"(missing: {', '.join(missing)})")
        else:
            print(f"[{idx}/{total}] {row.project}/{row.design_pattern}/{row.file_name}")
            record.update(status="ranked", missing_methods="")
            record.update(rank_class(judge, row.human_summary, summaries))
            print(f"    winner: {record['winner'] or '-'}")
        records.append(record)
        # Saved after every class so an interrupted run keeps what it has judged so far.
        pd.DataFrame(records).to_csv(DETAIL_CSV, index=False)

    write_outputs(pd.DataFrame(records), config)


def write_outputs(detail: pd.DataFrame, config: JudgeConfig) -> None:
    """Write the detail, summary, head-to-head and results CSVs and the results.txt block."""
    detail.to_csv(DETAIL_CSV, index=False)
    summary = build_summary(detail)
    summary.to_csv(SUMMARY_CSV, index=False)
    head_to_head = build_head_to_head(detail)
    head_to_head.to_csv(HEAD_TO_HEAD_CSV, index=False)

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    build_results_table(detail, summary, head_to_head, config, generated_at).to_csv(RESULTS_CSV, index=False)
    section = format_results_section(detail, summary, head_to_head, config, generated_at)
    write_block(RESULTS_TXT, section)

    print("\n" + section)
    print(f"Detail written to: {DETAIL_CSV}")
    print(f"Summary written to: {SUMMARY_CSV}")
    print(f"Head-to-head written to: {HEAD_TO_HEAD_CSV}")
    print(f"Results table written to: {RESULTS_CSV}")
    print(f"Results block written to: {RESULTS_TXT}")


def rescore(retry_invalid: bool) -> None:
    """Re-score the saved run from four_way_ranking_detail.csv; optionally re-ask invalid judgements."""
    if not DETAIL_CSV.exists():
        raise FileNotFoundError(f"{DETAIL_CSV} not found; run the ranking first.")
    config = load_judge_config()
    judge = FourWayJudge(config, load_prompts()) if retry_invalid else None
    detail = pd.read_csv(DETAIL_CSV)

    # Keep the detail file as the first full run wrote it, before any re-scoring replaces it.
    if not DETAIL_BACKUP_CSV.exists():
        detail.to_csv(DETAIL_BACKUP_CSV, index=False)
        print(f"Original detail kept as: {DETAIL_BACKUP_CSV}")

    print("=" * 70)
    print("FOUR-WAY RANKING: " + ("RE-ASKING INVALID JUDGEMENTS" if retry_invalid else "RE-SCORING SAVED REPLIES"))
    print("=" * 70)
    print(f"Note: results.txt reports the judge model and max tokens from .env ({config.model}, "
          f"{config.max_tokens}); they should match the settings of the saved run.")
    write_outputs(rescore_detail(detail, judge), config)


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank NLG, GPT, Mistral and SWUM summaries against the human summaries with an LLM judge."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Rank only the first N human summaries (for a cheap trial run). Default: all.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--reparse",
        action="store_true",
        help="Do not call the judge: score the replies saved in four_way_ranking_detail.csv again with the "
             "current parser and rewrite all outputs.",
    )
    mode.add_argument(
        "--retry-invalid",
        action="store_true",
        help="As --reparse, then ask the judge once more for each judgement that is still invalid.",
    )
    args = parser.parse_args(argv)
    if args.limit is not None and (args.reparse or args.retry_invalid):
        parser.error("--limit applies to a new run only, not to --reparse or --retry-invalid")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_arguments(argv)
    try:
        if args.reparse or args.retry_invalid:
            rescore(retry_invalid=args.retry_invalid)
        else:
            run(args.limit)
    except Exception as exc:
        print(f"\nERROR: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
