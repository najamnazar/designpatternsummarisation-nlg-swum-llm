"""Build the final-results/ directory: per-RQ data and paper-ready tables.

Every table in the paper is regenerated here directly from the raw artefacts
(summary corpora, class-level score files, judge rankings) so that no number in
the manuscript is transcribed by hand. Tables are written twice: a CSV holding
the data and a .tex fragment holding the same rows in booktabs form.

Layout (paper table numbers are preserved so tables map one-to-one):

    final-results/
      corpus/   Table I   corpus composition
                Table II  example summaries incl. the human reference
      rq1/      Table III similarity by method (+ mean length)
                Table IV  length-matched comparison (needs regeneration; stub)
                Table V   pattern naming accuracy
                wilcoxon  paired tests with rank-biserial effect sizes
      rq2/      Table VI  rubric rank counts, regenerated from the raw judge output
                Friedman per-criterion tests and post-hoc pairwise comparisons
                Spearman criterion-to-aggregate sensitivity analysis
                conciseness vs. measured length (judge behaviour check)
      rq3/      Table VIII conciseness by pattern, for all three pipelines
                Table IX   per-pattern similarity
                Table X    file-weighted vs pattern-weighted aggregation
                Table XI   per-pattern ranking outcomes
                Table XII  results partitioned by repository provenance
      rq4/      Table XIII per-iteration corpus statistics (B1-B5)
                Table XIV  within-file variation across the five runs
                Table XV   length control under explicit target budgets
                Table XVI  first-place assignments aggregated over five runs

Columns that require a generation or scoring run this script cannot perform
(BERTScore on the length-controlled corpora, the length-matched regeneration,
the second judge models, a current-generation model) are emitted as empty cells
and listed in final-results/README.md under "Pending".

Usage:
    python python/build_final_results.py [--out final-results]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, mannwhitneyu, spearmanr, wilcoxon
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calculate_wilcoxon_tests import (  # noqa: E402  (local module, path set above)
    effect_magnitude,
    rank_biserial_effect_size,
)
# CHANGE (input layout): input paths come from one shared module (input/corpus, input/ground-truth).
from project_paths import HUMAN_SUMMARIES_CSV  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "evaluation-results"
SUMMARY_DIR = ROOT / "output" / "summary-output"
# HUMAN_CSV = ROOT / "input" / "DPS_Human_Summaries.csv"
HUMAN_CSV = HUMAN_SUMMARIES_CSV

# Pipeline label -> class-score file. The LLM arm is the mixtral run, which is
# the same generation as output/summary-output/llm_summaries.csv (== B.csv) and
# therefore the same corpus the rubric rankings were collected on.
SCORE_FILES = {
    "DPS_NLG": "nlg_vs_human_class_scores.csv",
    "DPS_SWUM": "swum_vs_human_class_scores.csv",
    "DPS_LLM": "llm (mixtral)_vs_human_class_scores.csv",
    "DPS_LLM (non-concise)": "llm (non-concise 50)_vs_human_class_scores.csv",
}

# In multi_criteria_rankings.csv the judge sees three summaries labelled 1/2/3,
# whose texts are carried in summary_a / summary_b / summary_c.
JUDGE_SLOTS = {1: ("DPS_NLG", "summary_a"), 2: ("DPS_LLM", "summary_b"), 3: ("DPS_SWUM", "summary_c")}
CRITERIA = ["accuracy", "conciseness", "adequacy", "code_context", "design_patterns"]
CRITERION_LABELS = {
    "accuracy": "ACCURACY",
    "conciseness": "CONCISENESS",
    "adequacy": "ADEQUACY",
    "code_context": "CODE CONTEXT",
    "design_patterns": "DESIGN PATTERN",
}
POINTS = {1: 3, 2: 2, 3: 1}  # rank -> points

# CHANGE (naming consistency): the Spring project is now "SpringFramework" everywhere (input
# folder, human CSV, summary CSVs), following the AbstractFactory style. Lookups go through
# repository_type(), which ignores case and punctuation, so older "Spring-Framework" /
# "spring-framework" labels still resolve.
# REPOSITORY_TYPE = {
#     "AbdurRKhalid": "Educational",
#     "JamesZBL": "Educational",
#     "Spring-Framework": "Commercial",
# }
REPOSITORY_TYPE = {
    "AbdurRKhalid": "Educational",
    "JamesZBL": "Educational",
    "SpringFramework": "Commercial",
}

# Length-control corpora. None = no explicit numeric target in the prompt.
LENGTH_CORPORA = [
    (20, "llm_summaries_20.csv"),
    (40, "llm_summaries_40.csv"),
    (50, "llm_summaries.csv"),
    (60, "llm_summaries_60.csv"),
    (80, "llm_summaries_80.csv"),
    (None, "llm_summaries_nonconcise.csv"),
]

# Canonical design-pattern vocabulary used for pattern naming accuracy and for
# the "named pattern changed" measure. Each entry maps a canonical name to the
# surface forms accepted in free text; matching is case-insensitive and
# whitespace/hyphen-insensitive, anchored on word boundaries.
PATTERN_VOCABULARY: Dict[str, List[str]] = {
    "Abstract Factory": ["abstract factory"],
    "Factory Method": ["factory method"],
    "Singleton": ["singleton"],
    "Builder": ["builder"],
    "Prototype": ["prototype"],
    "Adapter": ["adapter"],
    "Bridge": ["bridge"],
    "Composite": ["composite"],
    "Decorator": ["decorator"],
    "Facade": ["facade", "façade"],
    "Flyweight": ["flyweight"],
    "Proxy": ["proxy"],
    "Chain of Responsibility": ["chain of responsibility"],
    "Command": ["command"],
    "Interpreter": ["interpreter"],
    "Iterator": ["iterator"],
    "Mediator": ["mediator"],
    "Memento": ["memento"],
    "Observer": ["observer"],
    "State": ["state"],
    "Strategy": ["strategy"],
    "Template Method": ["template method"],
    "Visitor": ["visitor"],
}

# Ground-truth labels in DPS_Human_Summaries.csv are written without spaces.
GROUND_TRUTH_TO_CANONICAL = {
    "AbstractFactory": "Abstract Factory",
    "FactoryMethod": "Factory Method",
    "Singleton": "Singleton",
    "Adapter": "Adapter",
    "Decorator": "Decorator",
    "Facade": "Facade",
    "Memento": "Memento",
    "Observer": "Observer",
    "Visitor": "Visitor",
}


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------
def squash(value: object) -> str:
    """Collapse whitespace and lowercase, for join keys."""
    return " ".join(str(value).split()).lower()


def alnum(value: object) -> str:
    """Strip everything but alphanumerics, for folder/pattern name matching."""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


# CHANGE (naming consistency): spelling-insensitive repository lookup.
def repository_type(project: object) -> str:
    """Repository type for a project label in any spelling (SpringFramework, Spring-Framework, ...)."""
    by_key = {alnum(name): kind for name, kind in REPOSITORY_TYPE.items()}
    return by_key.get(alnum(project), "Unknown")


def tokens(text: str) -> List[str]:
    return [t for t in str(text).replace("\n", " ").split() if t]


def word_count(text: str) -> int:
    return len(tokens(text))


def char_count(text: str) -> int:
    """Non-whitespace characters, matching analyze_conciseness.py."""
    return sum(1 for c in str(text) if not c.isspace())


def sentence_count(text: str) -> int:
    """Split on . ! ? and drop empty fragments, matching analyze_conciseness.py."""
    parts, start = [], 0
    text = str(text).strip()
    for i, ch in enumerate(text):
        if ch in ".!?":
            segment = text[start:i].strip()
            if segment:
                parts.append(segment)
            start = i + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return len(parts)


def type_token_ratio(text: str) -> float:
    toks = tokens(text)
    if not toks:
        return 0.0
    cleaned = [t.strip(".,;:!?()[]{}\"'`").lower() for t in toks if t]
    return len(set(cleaned)) / len(cleaned) if cleaned else 0.0


def tfidf_cosine(text_a: str, text_b: str) -> float:
    """Pairwise TF-IDF cosine, identical to EvaluationMetrics.cosine_similarity."""
    try:
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform([str(text_a), str(text_b)])
        return float(sk_cosine(matrix[0:1], matrix[1:2])[0, 0])
    except ValueError:
        # Raised when both documents are empty after tokenisation.
        return 0.0


# Pattern names that are also ordinary English words in a class summary
# ("maintains state", "executes the command"). For these the word "pattern" must
# appear within PATTERN_WINDOW tokens, otherwise a summary that never mentions a
# design pattern would be scored as naming one.
AMBIGUOUS_PATTERN_NAMES = {
    "Builder", "Prototype", "Bridge", "Composite", "Proxy", "Command", "State",
    "Strategy", "Iterator", "Mediator", "Interpreter",
}
PATTERN_WINDOW = 4


def named_patterns(text: str) -> frozenset:
    """Canonical design patterns explicitly named in a summary.

    Matching is case-insensitive and ignores punctuation, so "abstract_factory",
    "Abstract Factory" and "abstract-factory" all count. Ambiguous single-word
    names additionally require the word "pattern" nearby; see
    AMBIGUOUS_PATTERN_NAMES.
    """
    words = re.sub(r"[^a-z0-9]+", " ", str(text).lower()).split()
    found = set()
    for canonical, surfaces in PATTERN_VOCABULARY.items():
        for surface in surfaces:
            needle = re.sub(r"[^a-z0-9]+", " ", surface.lower()).split()
            span = len(needle)
            for start in range(len(words) - span + 1):
                if words[start:start + span] != needle:
                    continue
                if canonical not in AMBIGUOUS_PATTERN_NAMES:
                    found.add(canonical)
                    break
                window = words[max(0, start - PATTERN_WINDOW): start + span + PATTERN_WINDOW]
                if "pattern" in window or "patterns" in window:
                    found.add(canonical)
                    break
            if canonical in found:
                break
    return frozenset(found)


def pct(numerator: float, denominator: float) -> float:
    return 100.0 * numerator / denominator if denominator else float("nan")


# ---------------------------------------------------------------------------
# Table emission
# ---------------------------------------------------------------------------
@dataclass
class Emitted:
    relative_path: str
    title: str
    rows: int


EMITTED: List[Emitted] = []


def latex_escape(value: object) -> str:
    text = str(value)
    if text == "nan":
        return ""
    for old, new in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                     ("_", r"\_"), ("#", r"\#"), ("$", r"\$")):
        text = text.replace(old, new)
    return text


def emit(df: pd.DataFrame, out_dir: Path, stem: str, title: str,
         note: str = "", tex: bool = True) -> None:
    """Write one table as CSV (data) and .tex (paste-ready rows)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{stem}.csv", index=False)

    if tex:
        alignment = "".join(
            "r" if pd.api.types.is_numeric_dtype(df[c]) else "l" for c in df.columns
        )
        lines = [
            f"% {title}",
            f"% generated by python/build_final_results.py -- do not edit by hand",
        ]
        if note:
            lines.append(f"% {note}")
        lines += [
            r"\begin{tabular}{" + alignment + "}",
            r"\toprule",
            " & ".join(latex_escape(c) for c in df.columns) + r" \\",
            r"\midrule",
        ]
        for row in df.itertuples(index=False):
            lines.append(" & ".join(latex_escape(v) for v in row) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (out_dir / f"{stem}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    EMITTED.append(
        Emitted(str((out_dir / f"{stem}.csv").relative_to(ROOT)), title, len(df))
    )
    print(f"  wrote {stem:<44} ({len(df)} rows)  {title}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_human() -> pd.DataFrame:
    human = pd.read_csv(HUMAN_CSV)
    human = human.rename(
        columns={
            "Project": "project",
            "File Name": "filename",
            "Design Pattern": "pattern_raw",
            "Human Summary": "summary_human",
            "URL": "url",
        }
    )
    human["pattern"] = human["pattern_raw"].map(
        lambda p: GROUND_TRUTH_TO_CANONICAL.get(str(p).strip(), str(p).strip())
    )
    # Key on project + filename + reference text. Filenames repeat across
    # folders, but the reference summary disambiguates them exactly, and the
    # reference text is carried in every score file.
    human["key"] = (
        human["project"].map(squash) + "::" + human["filename"].map(squash)
        + "::" + human["summary_human"].map(squash)
    )
    human["folder_key"] = (
        human["project"].map(alnum) + "::" + human["pattern_raw"].map(alnum)
        + "::" + human["filename"].map(alnum)
    )
    if human["key"].duplicated().any() or human["folder_key"].duplicated().any():
        raise ValueError("Human reference keys are not unique; joins would be ambiguous.")
    return human


def load_scores(human: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Class-level scores per pipeline, carrying the ground-truth pattern label."""
    labels = human.set_index("key")[["pattern", "project", "filename"]]
    frames: Dict[str, pd.DataFrame] = {}
    for label, filename in SCORE_FILES.items():
        df = pd.read_csv(EVAL_DIR / filename)
        df["key"] = (
            df["base_project"].map(squash) + "::" + df["filename"].map(squash)
            + "::" + df["summary_human"].map(squash)
        )
        missing = (~df["key"].isin(labels.index)).sum()
        if missing:
            raise ValueError(f"{filename}: {missing} rows do not match a human reference.")
        df["pattern"] = df["key"].map(labels["pattern"])
        df["words"] = df["summary_method"].map(word_count)
        frames[label] = df
    return frames


def load_corpus(path: Path, human: pd.DataFrame) -> pd.DataFrame:
    """A summary-output CSV joined to the human references by project/folder/file."""
    df = pd.read_csv(path)
    df["folder_key"] = (
        df["Project"].map(alnum) + "::" + df["Folder Name"].map(alnum)
        + "::" + df["File Name"].map(alnum)
    )
    merged = df.merge(
        human[["folder_key", "pattern", "summary_human", "project", "filename"]],
        on="folder_key",
        how="left",
        validate="one_to_one",
    )
    if merged["summary_human"].isna().any():
        raise ValueError(f"{path.name}: rows without a matching human reference.")
    merged["summary"] = merged["Summary"].fillna("")
    merged["words"] = merged["summary"].map(word_count)
    return merged


def load_rankings(path: Path) -> pd.DataFrame:
    """Judge rankings with per-criterion ranks resolved to pipeline labels."""
    df = pd.read_csv(path)
    df = df[df["status"] == "ranked"].copy()
    for criterion in CRITERIA:
        for rank, column in enumerate(
            [f"{criterion}_rank_1st", f"{criterion}_rank_2nd", f"{criterion}_rank_3rd"], start=1
        ):
            df[column] = pd.to_numeric(df[column], errors="coerce")
        # A valid judgement is a permutation of {1,2,3}; anything else (blank or
        # malformed judge output) is dropped for that criterion only.
        ranks = df[[f"{criterion}_rank_{p}" for p in ("1st", "2nd", "3rd")]]
        valid = ranks.apply(
            lambda row: sorted(row.dropna().tolist()) == [1.0, 2.0, 3.0], axis=1
        )
        df[f"{criterion}_valid"] = valid
        for slot, (pipeline, _) in JUDGE_SLOTS.items():
            df[f"{criterion}_rank__{pipeline}"] = np.where(
                valid,
                np.select(
                    [ranks[f"{criterion}_rank_1st"] == slot,
                     ranks[f"{criterion}_rank_2nd"] == slot,
                     ranks[f"{criterion}_rank_3rd"] == slot],
                    [1, 2, 3],
                    default=np.nan,
                ),
                np.nan,
            )
    return df


# ---------------------------------------------------------------------------
# Corpus tables
# ---------------------------------------------------------------------------
def build_corpus_tables(out: Path, human: pd.DataFrame, scores: Dict[str, pd.DataFrame]) -> None:
    print("\ncorpus/")
    grouped = human.groupby("project")
    rows = []
    for project, group in grouped:
        rows.append(
            {
                "Repository": project,
                # CHANGE (naming consistency): spelling-insensitive lookup.
                # "Type": REPOSITORY_TYPE.get(project, "Unknown"),
                "Type": repository_type(project),
                "Files": len(group),
                "Patterns covered": group["pattern"].nunique(),
            }
        )
    rows.append(
        {
            "Repository": "Total",
            "Type": "",
            "Files": len(human),
            "Patterns covered": human["pattern"].nunique(),
        }
    )
    emit(pd.DataFrame(rows), out, "table01_corpus_composition",
         "Table I: corpus composition by source repository")

    # Table II: the three files used as qualitative examples, with the human
    # reference the submitted version left as a placeholder.
    examples = ["WindowsOSFactory.java", "RoundPeg.java", "EventManager.java"]
    rows = []
    for filename in examples:
        row = {"File name": filename.replace(".java", "")}
        for label in ("DPS_NLG", "DPS_SWUM", "DPS_LLM"):
            match = scores[label][scores[label]["filename"] == filename]
            row[f"{label} Summary"] = match["summary_method"].iloc[0] if len(match) else ""
        match = human[human["filename"] == filename]
        row["Human Reference (DPS Corpus)"] = (
            match["summary_human"].iloc[0] if len(match) else ""
        )
        rows.append(row)
    emit(pd.DataFrame(rows), out, "table02_example_summaries",
         "Table II: example summaries across pipelines, including the human reference",
         tex=False)


# ---------------------------------------------------------------------------
# RQ1
# ---------------------------------------------------------------------------
def build_rq1(out: Path, human: pd.DataFrame, scores: Dict[str, pd.DataFrame]) -> Dict[str, object]:
    print("\nrq1/")
    facts: Dict[str, object] = {}

    # --- Table III: similarity by method, with the length row filled in -----
    metrics = [
        ("Average Cosine Similarity", "cosine_similarity", 4),
        ("Average BERTScore Precision", "bert_precision", 4),
        ("Average BERTScore Recall", "bert_recall", 4),
        ("Average BERTScore F1", "bert_f1", 4),
    ]
    rows = []
    for label, column, places in metrics:
        row = {"Metric": label}
        for pipeline, df in scores.items():
            row[pipeline] = round(float(df[column].mean()), places)
        rows.append(row)
    for label, fn in (
        ("Mean summary length (words)", lambda s: s.mean()),
        ("SD of summary length (words)", lambda s: s.std(ddof=0)),
        ("Max summary length (words)", lambda s: float(s.max())),
        ("Summaries over 50 words (%)", lambda s: 100.0 * (s > 50).mean()),
    ):
        row = {"Metric": label}
        for pipeline, df in scores.items():
            row[pipeline] = round(float(fn(df["words"])), 2)
        rows.append(row)
    table3 = pd.DataFrame(rows)
    emit(table3, out, "table03_similarity_by_method",
         "Table III: similarity against human references, with output length",
         note="Length rows are new: the similarity metrics are length-sensitive.")

    facts["length"] = {
        p: (float(df["words"].mean()), float(df["words"].std(ddof=0)), int(df["words"].max()),
            float(100.0 * (df["words"] > 50).mean()))
        for p, df in scores.items()
    }

    # --- Paired Wilcoxon tests with effect sizes ----------------------------
    llm = scores["DPS_LLM"].set_index("key")
    rows = []
    for pipeline in ("DPS_NLG", "DPS_SWUM"):
        arm = scores[pipeline].set_index("key")
        shared = arm.index.intersection(llm.index)
        if len(shared) != len(arm):
            raise ValueError(f"{pipeline} and DPS_LLM do not cover the same files.")
        for metric, pretty in (("cosine_similarity", "Cosine"), ("bert_f1", "BERTScore F1")):
            x = arm.loc[shared, metric].to_numpy(dtype=float)
            y = llm.loc[shared, metric].to_numpy(dtype=float)
            diffs = x - y
            stat = wilcoxon(x, y, alternative="two-sided")
            r = rank_biserial_effect_size(diffs)
            rows.append(
                {
                    "Comparison": f"{pipeline} vs DPS_LLM",
                    "Metric": pretty,
                    "n": len(shared),
                    "W": round(float(stat.statistic), 1),
                    "p": f"{float(stat.pvalue):.3g}",
                    "Median diff": round(float(np.median(diffs)), 4),
                    "r": round(r, 3),
                    "Magnitude": effect_magnitude(r),
                    "Significant": "yes" if stat.pvalue < 0.05 else "no",
                    "Direction": "DPS_LLM higher" if np.median(diffs) < 0 else f"{pipeline} higher",
                }
            )
    wilcoxon_table = pd.DataFrame(rows)
    emit(wilcoxon_table, out, "table03b_wilcoxon_effect_sizes",
         "Paired Wilcoxon signed-rank tests with matched-pairs rank-biserial effect sizes",
         note="Effect sizes are new; the submitted version claimed them without reporting any.")
    facts["wilcoxon"] = wilcoxon_table

    # --- Table IV: length-matched comparison (stub; needs regeneration) -----
    stub = pd.DataFrame(
        [
            {"Budget": "DPS_NLG length", "Method": "DPS_NLG",
             "Mean words": round(float(scores["DPS_NLG"]["words"].mean()), 2),
             "Cosine": round(float(scores["DPS_NLG"]["cosine_similarity"].mean()), 4),
             "BERTScore F1": round(float(scores["DPS_NLG"]["bert_f1"].mean()), 4)},
            {"Budget": "DPS_NLG length", "Method": "DPS_LLM (matched)",
             "Mean words": "", "Cosine": "", "BERTScore F1": ""},
            {"Budget": "DPS_SWUM length", "Method": "DPS_SWUM",
             "Mean words": round(float(scores["DPS_SWUM"]["words"].mean()), 2),
             "Cosine": round(float(scores["DPS_SWUM"]["cosine_similarity"].mean()), 4),
             "BERTScore F1": round(float(scores["DPS_SWUM"]["bert_f1"].mean()), 4)},
            {"Budget": "DPS_SWUM length", "Method": "DPS_LLM (matched)",
             "Mean words": "", "Cosine": "", "BERTScore F1": ""},
            {"Budget": "Unconstrained", "Method": "DPS_LLM",
             "Mean words": round(float(scores["DPS_LLM"]["words"].mean()), 2),
             "Cosine": round(float(scores["DPS_LLM"]["cosine_similarity"].mean()), 4),
             "BERTScore F1": round(float(scores["DPS_LLM"]["bert_f1"].mean()), 4)},
        ]
    )
    emit(stub, out, "table04_length_matched_PENDING",
         "Table IV: length-matched comparison -- deterministic rows filled, matched-LLM rows pending",
         note="The matched-LLM rows require regenerating summaries at 58- and 14-word budgets.")

    # --- Table V: pattern naming accuracy -----------------------------------
    rows = []
    per_pattern_rows = []
    for pipeline in ("DPS_NLG", "DPS_SWUM", "DPS_LLM"):
        df = scores[pipeline]
        named = df["summary_method"].map(named_patterns)
        correct = [gt in got for gt, got in zip(df["pattern"], named)]
        names_any = [len(got) > 0 for got in named]
        wrong_only = [
            (len(got) > 0) and (gt not in got) for gt, got in zip(df["pattern"], named)
        ]
        rows.append(
            {
                "Measure": "Pattern naming accuracy (%)",
                "Pipeline": pipeline,
                "Value": round(pct(sum(correct), len(df)), 1),
            }
        )
        rows.append(
            {
                "Measure": "Names some pattern (%)",
                "Pipeline": pipeline,
                "Value": round(pct(sum(names_any), len(df)), 1),
            }
        )
        rows.append(
            {
                "Measure": "Names only an incorrect pattern (%)",
                "Pipeline": pipeline,
                "Value": round(pct(sum(wrong_only), len(df)), 1),
            }
        )
        rows.append(
            {
                "Measure": "Names no pattern at all (%)",
                "Pipeline": pipeline,
                "Value": round(pct(sum(not a for a in names_any), len(df)), 1),
            }
        )
        by_pattern = pd.DataFrame({"pattern": df["pattern"], "correct": correct})
        for pattern, group in by_pattern.groupby("pattern"):
            per_pattern_rows.append(
                {
                    "Pattern": pattern,
                    "Files": len(group),
                    "Pipeline": pipeline,
                    "Naming accuracy (%)": round(pct(group["correct"].sum(), len(group)), 1),
                }
            )
    naming = pd.DataFrame(rows).pivot(index="Measure", columns="Pipeline", values="Value").reset_index()
    naming = naming[["Measure", "DPS_NLG", "DPS_SWUM", "DPS_LLM"]]
    emit(naming, out, "table05_pattern_naming_accuracy",
         "Table V: pattern naming accuracy, scored against ground-truth labels",
         note="Reference-free and length-invariant; no judge model involved.")
    facts["naming"] = naming

    per_pattern = (
        pd.DataFrame(per_pattern_rows)
        .pivot(index=["Pattern", "Files"], columns="Pipeline", values="Naming accuracy (%)")
        .reset_index()
    )
    emit(per_pattern, out, "table05b_pattern_naming_by_pattern",
         "Pattern naming accuracy broken down by design pattern")
    return facts


# ---------------------------------------------------------------------------
# RQ2
# ---------------------------------------------------------------------------
def rank_count_table(rankings: pd.DataFrame) -> pd.DataFrame:
    """Table VI: 1st/2nd/3rd counts per criterion, with the per-criterion n."""
    rows = []
    for criterion in CRITERIA:
        subset = rankings[rankings[f"{criterion}_valid"]]
        row = {"Metric": CRITERION_LABELS[criterion]}
        for pipeline in ("DPS_NLG", "DPS_LLM", "DPS_SWUM"):
            ranks = subset[f"{criterion}_rank__{pipeline}"]
            counts = [int((ranks == r).sum()) for r in (1, 2, 3)]
            row[f"{pipeline} (1st/2nd/3rd)"] = " / ".join(str(c) for c in counts)
        row["n (valid judgements)"] = int(len(subset))
        rows.append(row)
    return pd.DataFrame(rows)


def friedman_tables(rankings: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-criterion Friedman tests plus Bonferroni-corrected post-hoc pairs."""
    summary_rows, posthoc_rows = [], []
    for criterion in CRITERIA:
        subset = rankings[rankings[f"{criterion}_valid"]]
        points = {
            pipeline: subset[f"{criterion}_rank__{pipeline}"].map(POINTS).to_numpy(dtype=float)
            for pipeline in ("DPS_NLG", "DPS_LLM", "DPS_SWUM")
        }
        stat, p_value = friedmanchisquare(*points.values())
        n_blocks, k = len(subset), 3
        kendall_w = float(stat) / (n_blocks * (k - 1))
        best = max(points, key=lambda pipeline: points[pipeline].mean())
        summary_rows.append(
            {
                "Criterion": CRITERION_LABELS[criterion],
                "n": n_blocks,
                "chi2": round(float(stat), 4),
                "p": f"{float(p_value):.3g}",
                "Kendall W": round(kendall_w, 4),
                "Significant": "yes" if p_value < 0.05 else "no",
                "Best": best,
                "Mean points DPS_NLG": round(float(points["DPS_NLG"].mean()), 4),
                "Mean points DPS_LLM": round(float(points["DPS_LLM"].mean()), 4),
                "Mean points DPS_SWUM": round(float(points["DPS_SWUM"].mean()), 4),
            }
        )
        for left, right in combinations(("DPS_NLG", "DPS_LLM", "DPS_SWUM"), 2):
            diffs = points[left] - points[right]
            if np.allclose(diffs, 0.0):
                w_stat, raw_p, r = 0.0, 1.0, 0.0
            else:
                result = wilcoxon(points[left], points[right], alternative="two-sided")
                w_stat, raw_p = float(result.statistic), float(result.pvalue)
                r = rank_biserial_effect_size(diffs)
            adjusted = min(1.0, raw_p * 3)  # Bonferroni over the three pairs
            posthoc_rows.append(
                {
                    "Criterion": CRITERION_LABELS[criterion],
                    "Comparison": f"{left} vs {right}",
                    "W": round(w_stat, 2),
                    "p (raw)": f"{raw_p:.3g}",
                    "p (Bonferroni)": f"{adjusted:.3g}",
                    "Significant": "yes" if adjusted < 0.05 else "no",
                    "r": round(r, 3),
                    "Magnitude": effect_magnitude(r),
                    "Direction": f"{left} > {right}" if diffs.mean() > 0 else f"{right} > {left}",
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(posthoc_rows)


def spearman_table(rankings: pd.DataFrame) -> pd.DataFrame:
    """Criterion-level points against aggregate points, per criterion."""
    rows = []
    for criterion in CRITERIA:
        subset = rankings[rankings[f"{criterion}_valid"]]
        criterion_points, total_points = [], []
        for pipeline, column in (("DPS_NLG", "total_points_a"), ("DPS_LLM", "total_points_b"),
                                 ("DPS_SWUM", "total_points_c")):
            criterion_points.extend(subset[f"{criterion}_rank__{pipeline}"].map(POINTS).tolist())
            total_points.extend(subset[column].tolist())
        rho, p_value = spearmanr(criterion_points, total_points)
        rows.append(
            {
                "Criterion": CRITERION_LABELS[criterion],
                "Blocks": int(len(subset)),
                "n (observations)": len(criterion_points),
                "Spearman rho": round(float(rho), 4),
                "p": f"{float(p_value):.3g}",
            }
        )
    return pd.DataFrame(rows)


def audit_ranking_sources(rankings: pd.DataFrame, human: pd.DataFrame) -> pd.DataFrame:
    """Check that the four texts in each judged row come from the same source file.

    rank_summaries.py keyed summaries on project + filename only and kept the
    first row per key, so for the 20 files that share a (project, filename) pair
    with another file the judge could be shown summaries of different classes.
    """
    corpora = {}
    for slot, filename in (("summary_a", "A.csv"), ("summary_b", "B.csv"), ("summary_c", "C.csv")):
        df = pd.read_csv(SUMMARY_DIR / filename)
        folder_keys = (
            df["Project"].map(alnum) + "::" + df["Folder Name"].map(alnum)
            + "::" + df["File Name"].map(alnum)
        )
        corpora[slot] = dict(zip(df["Summary"].map(squash), folder_keys))

    reference_lookup = dict(zip(human["key"], human["folder_key"]))
    rows = []
    for row in rankings.itertuples(index=False):
        reference_key = reference_lookup.get(
            f"{squash(row.project)}::{squash(row.file)}::{squash(row.human_summary)}"
        )
        sources = {
            slot: corpora[slot].get(squash(getattr(row, slot)), "unmatched")
            for slot in corpora
        }
        def folder_of(key: Optional[str]) -> str:
            # CHANGE: show the folder in the project's AbstractFactory naming style rather than the
            # lower-case join key (previously "abstractfactory"); unknown folders fall back to the key.
            # return key.split("::")[1] if key and key != "unmatched" else "unmatched"
            if not key or key == "unmatched":
                return "unmatched"
            folder = key.split("::")[1]
            return {alnum(label): label for label in GROUND_TRUTH_TO_CANONICAL}.get(folder, folder)
        rows.append(
            {
                "Project": row.project,
                "File": row.file,
                "Reference file (folder)": folder_of(reference_key),
                "DPS_NLG summary from": folder_of(sources["summary_a"]),
                "DPS_LLM summary from": folder_of(sources["summary_b"]),
                "DPS_SWUM summary from": folder_of(sources["summary_c"]),
                "All four from one file": len({reference_key, *sources.values()}) == 1,
            }
        )
    return pd.DataFrame(rows)


def build_rq2(out: Path, rankings: pd.DataFrame, human: pd.DataFrame) -> Dict[str, object]:
    print("\nrq2/")
    facts: Dict[str, object] = {}

    # --- Table VI: rank counts, regenerated ---------------------------------
    table6 = rank_count_table(rankings)
    emit(table6, out, "table06_rubric_rank_counts",
         "Table VI: rubric rank counts per criterion, regenerated from the raw judge output",
         note="Every rank column sums to the per-criterion n shown in the last column.")
    facts["table6"] = table6

    # Long form, easier to re-check than the packed "1 / 120 / 28" cells.
    long_rows = []
    for criterion in CRITERIA:
        subset = rankings[rankings[f"{criterion}_valid"]]
        for pipeline in ("DPS_NLG", "DPS_LLM", "DPS_SWUM"):
            ranks = subset[f"{criterion}_rank__{pipeline}"]
            long_rows.append(
                {
                    "Criterion": CRITERION_LABELS[criterion],
                    "Pipeline": pipeline,
                    "First": int((ranks == 1).sum()),
                    "Second": int((ranks == 2).sum()),
                    "Third": int((ranks == 3).sum()),
                    "n": int(len(subset)),
                    "Mean points": round(float(ranks.map(POINTS).mean()), 4),
                }
            )
    emit(pd.DataFrame(long_rows), out, "table06b_rubric_rank_counts_long",
         "Table VI in long form, with mean points per pipeline and criterion")

    # --- Source audit of the judged rows ------------------------------------
    audit = audit_ranking_sources(rankings, human)
    mismatched = audit[~audit["All four from one file"]]
    emit(mismatched.drop(columns=["All four from one file"]), out,
         "table06c_ranking_source_audit",
         "Judged rows whose four summaries do not all come from the same source file",
         note="Caused by the project::filename match key in rank_summaries.py; the key is now folder-aware.")
    facts["audit"] = audit
    facts["mismatched"] = mismatched

    clean = rankings.loc[audit["All four from one file"].to_numpy()]
    emit(rank_count_table(clean), out, "table06d_rubric_rank_counts_clean_subset",
         f"Table VI recomputed on the {len(clean)} rows with a verified single source file",
         note="Sensitivity check: does the affected pairing change any conclusion?")
    facts["clean_n"] = int(len(clean))

    # --- Friedman per criterion, with post-hoc pairwise Wilcoxon ------------
    friedman_summary, posthoc = friedman_tables(rankings)
    emit(friedman_summary, out, "table07_friedman_per_criterion",
         "Friedman tests per rubric criterion (class files as blocks)")
    emit(posthoc, out, "table07b_friedman_posthoc",
         "Post-hoc pairwise Wilcoxon comparisons with Bonferroni correction and effect sizes")
    facts["friedman"] = friedman_summary
    facts["posthoc"] = posthoc

    clean_friedman, clean_posthoc = friedman_tables(clean)
    emit(clean_friedman, out, "table07_friedman_clean_subset",
         f"Friedman tests recomputed on the {len(clean)} verified-source rows")
    emit(clean_posthoc, out, "table07b_friedman_posthoc_clean_subset",
         f"Post-hoc comparisons recomputed on the {len(clean)} verified-source rows")
    facts["clean_friedman"] = clean_friedman
    facts["clean_posthoc"] = clean_posthoc

    # --- Spearman criterion-to-aggregate sensitivity ------------------------
    spearman_full = spearman_table(rankings)
    emit(spearman_full, out, "table07c_spearman_sensitivity",
         "Spearman correlation between criterion-level points and aggregate points")
    emit(spearman_table(clean), out, "table07c_spearman_clean_subset",
         f"Spearman sensitivity recomputed on the {len(clean)} verified-source rows")
    facts["spearman"] = spearman_full

    # --- New: does the judge's conciseness criterion track length? ----------
    subset = rankings[rankings["conciseness_valid"]].copy()
    for pipeline, column in JUDGE_SLOTS.values():
        subset[f"words__{pipeline}"] = subset[column].map(word_count)

    word_columns = {pipeline: f"words__{pipeline}" for pipeline, _ in JUDGE_SLOTS.values()}
    shortest_pipeline = (
        subset[list(word_columns.values())].idxmin(axis=1).str.replace("words__", "", regex=False)
    )
    rows = []
    for pipeline in ("DPS_NLG", "DPS_LLM", "DPS_SWUM"):
        ranks = subset[f"conciseness_rank__{pipeline}"]
        is_first = ranks == 1
        is_shortest = shortest_pipeline == pipeline
        rows.append(
            {
                "Pipeline": pipeline,
                "Mean words": round(float(subset[word_columns[pipeline]].mean()), 2),
                "SD words": round(float(subset[word_columns[pipeline]].std(ddof=0)), 2),
                "Ranked 1st on conciseness": int(is_first.sum()),
                "Actually the shortest of the three": int(is_shortest.sum()),
                "Ranked 1st and shortest": int((is_first & is_shortest).sum()),
                "Mean conciseness points": round(float(ranks.map(POINTS).mean()), 4),
            }
        )
    conciseness_table = pd.DataFrame(rows)
    emit(conciseness_table, out, "table07d_conciseness_vs_length",
         "Judge conciseness rankings against measured summary length",
         note=("New analysis: the judge's conciseness criterion does not track word count. "
               "Lengths are of the texts as presented to the judge, so DPS_NLG reads 63.68 "
               "rather than the corpus mean of 57.90; the gap is the mis-sourced rows in "
               "table06c."))

    pooled_points, pooled_words = [], []
    for pipeline in ("DPS_NLG", "DPS_LLM", "DPS_SWUM"):
        pooled_points.extend(subset[f"conciseness_rank__{pipeline}"].map(POINTS).tolist())
        pooled_words.extend(subset[word_columns[pipeline]].tolist())
    rho, p_value = spearmanr(pooled_points, pooled_words)

    shortest_ranked_first = sum(
        subset.loc[index, f"conciseness_rank__{pipeline}"] == 1
        for index, pipeline in shortest_pipeline.items()
    )
    diagnostics = pd.DataFrame(
        [
            {"Quantity": "Files judged on conciseness", "Value": int(len(subset))},
            {"Quantity": "Judge's 1st choice is the shortest summary (%)",
             "Value": round(pct(shortest_ranked_first, len(subset)), 1)},
            {"Quantity": "Spearman rho (conciseness points vs word count, pooled)",
             "Value": round(float(rho), 4)},
            {"Quantity": "p", "Value": f"{float(p_value):.3g}"},
            {"Quantity": "n (pooled observations)", "Value": len(pooled_points)},
        ]
    )
    emit(diagnostics, out, "table07e_conciseness_length_diagnostics",
         "Whether the judge's conciseness ranking agrees with measured brevity")
    facts["conciseness"] = conciseness_table
    facts["conciseness_diagnostics"] = diagnostics
    return facts


def build_rq3(out: Path, human: pd.DataFrame, scores: Dict[str, pd.DataFrame],
              rankings: pd.DataFrame) -> Dict[str, object]:
    print("\nrq3/")
    facts: Dict[str, object] = {}

    # --- Table VIII: conciseness by pattern, for all three pipelines --------
    rows = []
    for pipeline in ("DPS_NLG", "DPS_SWUM", "DPS_LLM"):
        df = scores[pipeline]
        for pattern, group in df.groupby("pattern"):
            summaries = group["summary_method"].tolist()
            words = [word_count(s) for s in summaries]
            per_sentence = [
                word_count(s) / sentence_count(s) if sentence_count(s) else float(word_count(s))
                for s in summaries
            ]
            rows.append(
                {
                    "Pattern": pattern,
                    "Pipeline": pipeline,
                    "Count": len(group),
                    "AvgWords": round(float(np.mean(words)), 2),
                    "MedianWords": round(float(np.median(words)), 2),
                    "StdWords": round(float(np.std(words)), 2),
                    "AvgChars": round(float(np.mean([char_count(s) for s in summaries])), 2),
                    "AvgWordsPerSent": round(float(np.mean(per_sentence)), 2),
                    "AvgTTR": round(float(np.mean([type_token_ratio(s) for s in summaries])), 2),
                }
            )
    by_pattern = pd.DataFrame(rows).sort_values(["Pipeline", "AvgWords"]).reset_index(drop=True)
    emit(by_pattern, out, "table08b_conciseness_by_pattern_all_pipelines",
         "Conciseness by design pattern for all three pipelines")

    llm_only = (
        by_pattern[by_pattern["Pipeline"] == "DPS_LLM"]
        .drop(columns=["Pipeline"])
        .sort_values("AvgWords")
        .reset_index(drop=True)
    )
    emit(llm_only, out, "table08_conciseness_by_pattern_llm",
         "Table VIII: conciseness by design pattern, probabilistic pipeline only",
         note="Recomputed from the raw summaries; medians are real medians, not copies of the means.")
    facts["by_pattern_llm"] = llm_only

    # --- Table IX: per-pattern similarity -----------------------------------
    rows = []
    patterns = sorted(scores["DPS_LLM"]["pattern"].unique())
    for pattern in patterns:
        row = {"Pattern": pattern,
               "Pairs": int((scores["DPS_LLM"]["pattern"] == pattern).sum())}
        for pipeline in ("DPS_NLG", "DPS_SWUM", "DPS_LLM"):
            group = scores[pipeline][scores[pipeline]["pattern"] == pattern]
            row[f"{pipeline} Cos"] = round(float(group["cosine_similarity"].mean()), 4)
            row[f"{pipeline} F1"] = round(float(group["bert_f1"].mean()), 4)
        rows.append(row)
    emit(pd.DataFrame(rows), out, "table09_per_pattern_similarity",
         "Table IX: per-pattern comparison of the three pipelines")

    # --- Table X: file-weighted vs pattern-weighted -------------------------
    rows = []
    for metric, pretty in (("cosine_similarity", "Cosine"), ("bert_f1", "BERTScore F1")):
        file_row = {"Aggregation": "File-weighted", "Metric": pretty}
        pattern_row = {"Aggregation": "Pattern-weighted", "Metric": pretty}
        for pipeline in ("DPS_NLG", "DPS_SWUM", "DPS_LLM"):
            df = scores[pipeline]
            file_row[pipeline] = round(float(df[metric].mean()), 4)
            pattern_row[pipeline] = round(
                float(df.groupby("pattern")[metric].mean().mean()), 4
            )
        rows.extend([file_row, pattern_row])
    weighted = pd.DataFrame(rows).sort_values(["Metric", "Aggregation"]).reset_index(drop=True)
    emit(weighted, out, "table10_weighted_aggregation",
         "Table X: file-weighted vs pattern-weighted corpus aggregates",
         note="Pattern-weighted gives each of the nine patterns equal weight.")
    facts["weighted"] = weighted

    # --- Table XI: per-pattern ranking outcomes -----------------------------
    ranked = rankings.copy()
    order = []
    for row in ranked.itertuples(index=False):
        points = {
            "DPS_NLG": row.total_points_a,
            "DPS_LLM": row.total_points_b,
            "DPS_SWUM": row.total_points_c,
        }
        # Ties broken deterministically in slot order A > B > C, matching
        # rank_design_patterns.py.
        ordering = sorted(
            points.items(),
            key=lambda item: (-item[1], ["DPS_NLG", "DPS_LLM", "DPS_SWUM"].index(item[0])),
        )
        order.append({pipeline: rank for rank, (pipeline, _) in enumerate(ordering, start=1)})
    for pipeline in ("DPS_NLG", "DPS_LLM", "DPS_SWUM"):
        ranked[f"overall_rank__{pipeline}"] = [o[pipeline] for o in order]

    labels = human.set_index("key")["pattern"]
    ranked["key"] = (
        ranked["project"].map(squash) + "::" + ranked["file"].map(squash)
        + "::" + ranked["human_summary"].map(squash)
    )
    ranked["pattern"] = ranked["key"].map(labels)
    if ranked["pattern"].isna().any():
        raise ValueError("Ranking rows could not be mapped to a ground-truth pattern.")

    rows = []
    for pattern, group in ranked.groupby("pattern"):
        row = {"Design Pattern": pattern}
        for pipeline in ("DPS_NLG", "DPS_LLM", "DPS_SWUM"):
            counts = [int((group[f"overall_rank__{pipeline}"] == r).sum()) for r in (1, 2, 3)]
            row[f"{pipeline} (1st/2nd/3rd)"] = " / ".join(str(c) for c in counts)
        row["Total"] = len(group)
        rows.append(row)
    table11 = pd.DataFrame(rows).sort_values("Total", ascending=False).reset_index(drop=True)
    emit(table11, out, "table11_per_pattern_rankings",
         "Table XI: per-pattern overall ranking outcomes, aggregated across all five rubric criteria",
         note="Overall rank per file from total points across the five criteria; ties broken NLG > LLM > SWUM.")

    # --- Table XII: repository provenance / memorisation check --------------
    # CHANGE (naming consistency): map any spelling of the project name to its partition.
    # partition_of = {p: ("Spring (commercial)" if t == "Commercial" else "Educational")
    #                 for p, t in REPOSITORY_TYPE.items()}
    def partition_of(project: object) -> Optional[str]:
        kind = repository_type(project)
        if kind == "Unknown":
            return None
        return "Spring (commercial)" if kind == "Commercial" else "Educational"

    rows = []
    deltas: Dict[str, Dict[str, np.ndarray]] = {}
    for metric, pretty in (("cosine_similarity", "Cosine"), ("bert_f1", "BERTScore F1")):
        deltas[metric] = {}
        for partition in ("Educational", "Spring (commercial)"):
            row = {"Partition": partition, "Metric": pretty}
            for pipeline in ("DPS_NLG", "DPS_SWUM", "DPS_LLM"):
                df = scores[pipeline]
                mask = df["base_project"].map(partition_of) == partition
                row[pipeline] = round(float(df.loc[mask, metric].mean()), 4)
                row["Files"] = int(mask.sum())
            swum = scores["DPS_SWUM"].set_index("key")
            llm = scores["DPS_LLM"].set_index("key")
            keys = [k for k in swum.index
                    # if partition_of.get(swum.loc[k, "base_project"]) == partition]
                    if partition_of(swum.loc[k, "base_project"]) == partition]
            deltas[metric][partition] = (
                llm.loc[keys, metric].to_numpy(dtype=float)
                - swum.loc[keys, metric].to_numpy(dtype=float)
            )
            row["LLM - SWUM"] = round(float(np.mean(deltas[metric][partition])), 4)
            rows.append(row)
    provenance = pd.DataFrame(rows)[
        ["Partition", "Files", "Metric", "DPS_NLG", "DPS_SWUM", "DPS_LLM", "LLM - SWUM"]
    ]
    emit(provenance, out, "table12_provenance_partition",
         "Table XII: results partitioned by source repository",
         note="Tests whether the probabilistic advantage is larger on the widely mirrored commercial repository.")

    rows = []
    for metric, pretty in (("cosine_similarity", "Cosine"), ("bert_f1", "BERTScore F1")):
        educational = deltas[metric]["Educational"]
        spring = deltas[metric]["Spring (commercial)"]
        stat, p_value = mannwhitneyu(educational, spring, alternative="two-sided")
        # Common-language effect size: P(educational delta > spring delta).
        cles = float(stat) / (len(educational) * len(spring))
        rows.append(
            {
                "Metric": pretty,
                "n educational": len(educational),
                "n Spring": len(spring),
                "Mean delta educational": round(float(np.mean(educational)), 4),
                "Mean delta Spring": round(float(np.mean(spring)), 4),
                "U": round(float(stat), 1),
                "p": f"{float(p_value):.3g}",
                "P(edu > spring)": round(cles, 3),
                "Difference significant": "yes" if p_value < 0.05 else "no",
            }
        )
    memorisation = pd.DataFrame(rows)
    emit(memorisation, out, "table12b_provenance_mannwhitney",
         "Mann-Whitney U on the per-file LLM-minus-SWUM advantage, educational vs commercial",
         note="A non-significant difference is evidence against the memorisation explanation.")
    facts["memorisation"] = memorisation
    facts["provenance"] = provenance
    return facts


# ---------------------------------------------------------------------------
# RQ4
# ---------------------------------------------------------------------------
def build_rq4(out: Path, human: pd.DataFrame) -> Dict[str, object]:
    print("\nrq4/")
    facts: Dict[str, object] = {}

    # --- Table XIII: per-iteration corpus statistics ------------------------
    iterations = pd.read_csv(EVAL_DIR / "b1_b5_vs_human_summary.csv")
    table13 = pd.DataFrame(
        {
            "Iteration": iterations["iteration"],
            "Cosine (min)": iterations["cosine_min"].round(4),
            "Cosine (max)": iterations["cosine_max"].round(4),
            "Cosine (avg)": iterations["cosine_avg"].round(4),
            "BERTScore F1 (min)": iterations["bert_min"].round(4),
            "BERTScore F1 (max)": iterations["bert_max"].round(4),
            "BERTScore F1 (avg)": iterations["bert_avg"].round(4),
        }
    )
    emit(table13, out, "table13_iteration_corpus_statistics",
         "Table XIII: corpus-level similarity across five identical runs")

    # --- Table XIV: within-file variation -----------------------------------
    runs = {n: load_corpus(SUMMARY_DIR / f"B{n}.csv", human) for n in range(1, 6)}
    keys = set.intersection(*[set(df["folder_key"]) for df in runs.values()])
    keys = sorted(keys)
    indexed = {n: df.set_index("folder_key") for n, df in runs.items()}

    identical, word_ranges, word_sds, pattern_changed, pairwise_cosines = 0, [], [], 0, []
    for key in keys:
        texts = [str(indexed[n].loc[key, "summary"]) for n in range(1, 6)]
        counts = [word_count(t) for t in texts]
        if len(set(texts)) == 1:
            identical += 1
        word_ranges.append(max(counts) - min(counts))
        word_sds.append(float(np.std(counts, ddof=0)))
        if len({named_patterns(t) for t in texts}) > 1:
            pattern_changed += 1
        pairwise_cosines.append(
            float(np.mean([tfidf_cosine(a, b) for a, b in combinations(texts, 2)]))
        )

    # Did the judge's overall winner for a file change between iterations?
    detail = pd.read_csv(EVAL_DIR / "b1_b5_ranking_detail.csv")
    detail = detail[detail["status"] == "ranked"]
    winners = detail.groupby("match_key")["winner"].nunique()
    rank_changed = int((winners > 1).sum())

    n = len(keys)
    rows = [
        ("Mean pairwise TF-IDF cosine between runs", 1.000, 1.000,
         round(float(np.mean(pairwise_cosines)), 4)),
        ("Files with byte-identical output (%)", 100.0, 100.0, round(pct(identical, n), 1)),
        ("Mean within-file SD of word count", 0.0, 0.0, round(float(np.mean(word_sds)), 2)),
        ("Mean within-file word-count range", 0.0, 0.0, round(float(np.mean(word_ranges)), 2)),
        ("Max within-file word-count range", 0, 0, int(max(word_ranges))),
        ("Files varying by more than 5 words (%)", 0.0, 0.0,
         round(pct(sum(r > 5 for r in word_ranges), n), 1)),
        ("Files varying by more than 10 words (%)", 0.0, 0.0,
         round(pct(sum(r > 10 for r in word_ranges), n), 1)),
        ("Files whose rubric winner changed (%)", 0.0, 0.0, round(pct(rank_changed, len(winners)), 1)),
        ("Files where the named pattern set changed (%)", 0.0, 0.0,
         round(pct(pattern_changed, n), 1)),
    ]
    table14 = pd.DataFrame(
        [{"Measure (per file, across 5 runs)": m, "DPS_NLG": a, "DPS_SWUM": b, "DPS_LLM": c}
         for m, a, b, c in rows]
    )
    emit(table14, out, "table14_instance_level_variability",
         "Table XIV: within-file variation of the probabilistic pipeline across five identical runs",
         note=f"n = {n} files. Deterministic pipelines are invariant by construction.")
    facts["table14"] = table14
    facts["n_files"] = n

    per_file = pd.DataFrame(
        {
            "project": [indexed[1].loc[k, "project"] for k in keys],
            "filename": [indexed[1].loc[k, "filename"] for k in keys],
            "pattern": [indexed[1].loc[k, "pattern"] for k in keys],
            "word_range": word_ranges,
            "word_sd": [round(s, 3) for s in word_sds],
            "mean_pairwise_cosine": [round(c, 4) for c in pairwise_cosines],
            "named_pattern_changed": [
                len({named_patterns(str(indexed[n].loc[k, "summary"])) for n in range(1, 6)}) > 1
                for k in keys
            ],
        }
    )
    emit(per_file.sort_values("word_range", ascending=False), out,
         "table14b_instance_variability_per_file",
         "Per-file within-run variation, sorted by word-count range", tex=False)

    # --- Table XV: length control under explicit budgets --------------------
    rows = []
    for target, filename in LENGTH_CORPORA:
        path = SUMMARY_DIR / filename
        if not path.exists():
            continue
        corpus = load_corpus(path, human)
        words = corpus["words"].to_numpy(dtype=float)
        cosines = [
            tfidf_cosine(row.summary_human, row.summary) for row in corpus.itertuples(index=False)
        ]
        within = (
            pct(int(np.sum(np.abs(words - target) <= 0.1 * target)), len(words))
            if target else float("nan")
        )
        rows.append(
            {
                "Target (words)": target if target else "No length instruction",
                "n": len(corpus),
                "Achieved mean": round(float(words.mean()), 2),
                "SD": round(float(words.std(ddof=0)), 2),
                "Min": int(words.min()),
                "Max": int(words.max()),
                "Shortfall vs target (%)": (
                    round(100.0 * (words.mean() - target) / target, 1) if target else ""
                ),
                "Within +/-10% (%)": round(within, 1) if target else "",
                "Cosine": round(float(np.mean(cosines)), 4),
                "BERTScore F1": "",
            }
        )
    for label, filename in (("DPS_NLG (deterministic)", "nlg_summaries.csv"),
                            ("DPS_SWUM (deterministic)", "swum_summaries.csv")):
        corpus = load_corpus(SUMMARY_DIR / filename, human)
        words = corpus["words"].to_numpy(dtype=float)
        cosines = [
            tfidf_cosine(row.summary_human, row.summary) for row in corpus.itertuples(index=False)
        ]
        rows.append(
            {
                "Target (words)": label,
                "n": len(corpus),
                "Achieved mean": round(float(words.mean()), 2),
                "SD": round(float(words.std(ddof=0)), 2),
                "Min": int(words.min()),
                "Max": int(words.max()),
                "Shortfall vs target (%)": "",
                "Within +/-10% (%)": "",
                "Cosine": round(float(np.mean(cosines)), 4),
                "BERTScore F1": "",
            }
        )
    table15 = pd.DataFrame(rows)
    emit(table15, out, "table15_length_control",
         "Table XV: length control under explicit prompt-level target budgets",
         note="Cosine is computed here (TF-IDF, as in the pipeline); BERTScore F1 needs evaluate_summaries.py.")
    facts["table15"] = table15

    # --- Table XVI: first-place assignments over five runs ------------------
    rows = []
    for criterion in CRITERIA:
        valid = detail[
            detail[[f"{criterion}_rank_{p}" for p in ("1st", "2nd", "3rd")]]
            .apply(lambda row: sorted(pd.to_numeric(row, errors="coerce").dropna().tolist())
                   == [1.0, 2.0, 3.0], axis=1)
        ]
        row = {"Criterion": CRITERION_LABELS[criterion], "Total judgements": len(valid)}
        for slot, (pipeline, _) in JUDGE_SLOTS.items():
            firsts = int((pd.to_numeric(valid[f"{criterion}_rank_1st"], errors="coerce") == slot).sum())
            row[f"{pipeline} 1st"] = firsts
            row[f"{pipeline} %"] = round(pct(firsts, len(valid)), 1)
        rows.append(row)
    emit(pd.DataFrame(rows), out, "table16_first_place_over_five_runs",
         "Table XVI: first-place assignments aggregated over five iterations",
         note="Counts are first-place assignments, not per-iteration rankings.")
    return facts


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------
def write_readme(out: Path, facts: Dict[str, Dict[str, object]]) -> None:
    lines: List[str] = []
    lines.append("# Final results\n")
    lines.append(
        "Regenerated by `python/build_final_results.py` from the raw artefacts in "
        "`output/summary-output/`, `evaluation-results/` and `input/`. Every table "
        "below is machine-generated; nothing here is transcribed by hand. Each table "
        "exists as a `.csv` (the data) and a `.tex` (paste-ready booktabs rows), and "
        "the file names carry the paper's table numbers.\n"
    )
    lines.append(
        "`ERRATA.md` in this directory lists what changed against the submitted "
        "manuscript, table by table, and what each change does to the claims in "
        "the text. Read that first if you are revising the paper.\n"
    )
    lines.append("## Provenance\n")
    lines.append(
        "- The probabilistic arm throughout is the Mixtral run stored in "
        "`output/summary-output/llm_summaries.csv` (identical to `B.csv`), scored in "
        "`evaluation-results/llm (mixtral)_vs_human_class_scores.csv`. This is the same "
        "generation the rubric rankings were collected on, so the similarity tests and "
        "the judge tables now describe one corpus rather than two.\n"
        "- Ground-truth design-pattern labels come from `input/ground-truth/DPS_Human_Summaries.csv`. "
        "Files are joined on repository + filename + reference text, which is unique "
        "across all 150 files (filenames alone are not).\n"
        "- Cosine similarity is TF-IDF cosine over the summary/reference pair, matching "
        "`EvaluationMetrics.cosine_similarity` in `evaluate_summaries.py` to within 1e-16.\n"
    )

    lines.append("## Contents\n")
    current_dir = None
    for item in EMITTED:
        directory = str(Path(item.relative_path).parent)
        if directory != current_dir:
            lines.append(f"\n**{directory}/**\n")
            current_dir = directory
        lines.append(f"- `{Path(item.relative_path).name}` — {item.title}")
    lines.append("")

    lines.append("\n## Pending: what this script cannot compute\n")
    lines.append(
        "These need a generation or scoring run, not post-processing:\n\n"
        "1. **BERTScore F1 for the length-controlled corpora** (Table XV). The summaries "
        "exist; scoring them needs `bert-score`. Run:\n"
        "   ```\n"
        "   python python/evaluate_summaries.py --dps-llm-csv output/summary-output/llm_summaries_20.csv --output-dir evaluation-results/length-20\n"
        "   ```\n"
        "   once per budget (20/40/60/80, changing both numbers; the separate output folder keeps the canonical Mixtral scores intact), then fill the F1 column of Table XV. The "
        "Cosine column is already computed here.\n"
        "2. **Table XIII, re-run of the B1-B5 evaluation.** `evaluate_b1_b5.py` used to "
        "pair 7 of 150 files with the wrong reference; the matcher is fixed but the "
        "committed scores predate the fix. Run `python python/evaluate_b1_b5.py`.\n"
        "3. **A clean judge run** (Tables VI, XI, XVI, Friedman, Spearman). The committed "
        "rankings were produced with a project::filename match key, which mixed sources "
        "in 18 of 150 rows (`rq2/table06c_ranking_source_audit.csv`). The key is fixed in "
        "`rank_summaries.py`; re-running needs judge API access. The clean-subset tables "
        "here show the conclusions are expected to survive.\n"
        "4. **Table IV, the length-matched comparison.** Requires regenerating the LLM "
        "summaries at a 58-word budget (DPS_NLG's mean) and a 14-word budget "
        "(DPS_SWUM's mean). The existing 20-word corpus is the closest available proxy "
        "and is not a match.\n"
        "5. **Table VII, multi-judge agreement.** Requires running the rubric procedure "
        "with two further judge models.\n"
        "6. **The current-generation model column.** Requires a second probabilistic arm.\n"
        "7. **Role-assignment and collaborator fidelity** (the rest of Table V). The DPS "
        "JSON in `output/json-output/llm/` carries role assignments for only 63 of the "
        "150 evaluated classes, and several entries assign implausible roles (for "
        "example `Button` and `CheckBox` as Memento originators). Scoring against it "
        "would produce numbers that cannot be defended, so only pattern naming "
        "accuracy — which uses the corpus's own ground-truth labels and covers all 150 "
        "files — is reported.\n"
    )
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n  wrote {out.name}/README.md")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Build the final-results directory.")
    parser.add_argument("--out", type=Path, default=ROOT / "final-results",
                        help="Output directory (default: final-results/)")
    args = parser.parse_args(argv)

    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    human = load_human()
    scores = load_scores(human)
    rankings = load_rankings(EVAL_DIR / "multi_criteria_rankings.csv")

    facts: Dict[str, Dict[str, object]] = {}
    build_corpus_tables(out / "corpus", human, scores)
    facts["rq1"] = build_rq1(out / "rq1", human, scores)
    facts["rq2"] = build_rq2(out / "rq2", rankings, human)
    facts["rq3"] = build_rq3(out / "rq3", human, scores, rankings)
    facts["rq4"] = build_rq4(out / "rq4", human)

    write_readme(out, facts)
    print(f"\nDone. {len(EMITTED)} tables written to {out}")


if __name__ == "__main__":
    main()
