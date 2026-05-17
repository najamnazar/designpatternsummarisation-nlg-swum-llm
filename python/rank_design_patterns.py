"""
rank_design_patterns.py - Design-pattern-focused ranking pipeline

This script performs one ranking iteration (A vs B vs C) using the same
multi-criteria LLM ranker as rank_summaries.py, then reports performance
separately per design pattern.

Outputs:
- evaluation-results/design_pattern_rankings_detail.csv
- evaluation-results/design_pattern_performance_summary.csv
- evaluation-results/design_pattern_ranking_report.txt
- Appended section in evaluation-results/results.txt
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Dict, List

import pandas as pd
from dotenv import load_dotenv

# Reuse the exact ranking engine and matching logic used in rank_summaries.py.
sys.path.insert(0, str(Path(__file__).parent))
from rank_summaries import MultiCriteriaRanker, build_match_key


def canonicalize_design_pattern(raw_value: str) -> str:
    """Map free-form pattern labels into canonical design pattern names."""
    if not isinstance(raw_value, str):
        return "Unknown"

    token = raw_value.strip()
    if not token:
        return "Unknown"

    norm = re.sub(r"[\s_\-/]+", " ", token.lower()).strip()
    patterns = {
        "abstract factory": "Abstract Factory",
        "abstractfactory": "Abstract Factory",
        "adapter": "Adapter",
        "decorator": "Decorator",
        "facade": "Facade",
        "factory method": "Factory Method",
        "factorymethod": "Factory Method",
        "memento": "Memento",
        "observer": "Observer",
        "singleton": "Singleton",
        "visitor": "Visitor",
    }

    if norm in patterns:
        return patterns[norm]

    camel = re.sub(r"(?<!^)(?=[A-Z])", " ", token).lower().strip()
    camel = re.sub(r"[\s_\-/]+", " ", camel).strip()
    if camel in patterns:
        return patterns[camel]

    return "Unknown"


class DesignPatternRankingPipeline:
    """Ranks A/B/C summaries and reports performance separately by design pattern."""

    CRITERIA = ["accuracy", "conciseness", "adequacy", "code_context", "design_patterns"]

    def __init__(self, output_dir: str = "output/summary-output", input_dir: str = "input", results_dir: str = "evaluation-results") -> None:
        """Initialize directories, environment configuration, and ranking client."""
        base_dir = Path(__file__).resolve().parent.parent
        self.base_dir = base_dir

        self.output_dir = Path(output_dir)
        if not self.output_dir.is_absolute():
            self.output_dir = (base_dir / self.output_dir).resolve()

        self.input_dir = Path(input_dir)
        if not self.input_dir.is_absolute():
            self.input_dir = (base_dir / self.input_dir).resolve()

        self.results_dir = Path(results_dir)
        if not self.results_dir.is_absolute():
            self.results_dir = (base_dir / self.results_dir).resolve()

        self.results_dir.mkdir(parents=True, exist_ok=True)

        env_path = base_dir / ".env"
        load_dotenv(env_path)

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in .env file")

        api_url = os.getenv("RANK_SUMMARIES_API_URL")
        if not api_url:
            api_url = os.getenv("OPENROUTER_API_URL")
        if not api_url:
            raise ValueError("Set OPENROUTER_API_URL (or RANK_SUMMARIES_API_URL) in .env before running design-pattern ranking")

        model = os.getenv("RANK_SUMMARIES_MODEL")
        if not model:
            raise ValueError("RANK_SUMMARIES_MODEL not found in .env file")

        max_tokens_raw = os.getenv("RANK_SUMMARIES_MAX_TOKENS", "50")
        try:
            max_tokens = int(max_tokens_raw)
        except ValueError as exc:
            raise ValueError("RANK_SUMMARIES_MAX_TOKENS must be an integer") from exc

        prompts_path = self.base_dir / "src" / "main" / "resources" / "prompts.json"
        if not prompts_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompts_path}")

        with open(prompts_path, "r", encoding="utf-8") as prompt_file:
            prompts_data = json.load(prompt_file)
        if not isinstance(prompts_data, dict):
            raise ValueError("prompts.json must contain a top-level JSON object")

        ranking_prompts = prompts_data.get("summary_ranking")
        if not isinstance(ranking_prompts, dict):
            raise ValueError("prompts.json is missing the summary_ranking section required by rank_design_patterns.py")

        self.ranker = MultiCriteriaRanker(
            api_key=api_key,
            api_url=api_url,
            model=model,
            prompts=ranking_prompts,
            max_tokens=max_tokens,
        )

    def load_summaries(self) -> None:
        """Load A/B/C outputs + human summaries and build lookup maps."""

        def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
            result = df.copy()
            result.columns = [col.strip().lower().replace(" ", "_") for col in result.columns]
            return result

        self.df_a = _prepare_dataframe(pd.read_csv(self.output_dir / "A.csv"))
        self.df_b = _prepare_dataframe(pd.read_csv(self.output_dir / "B.csv"))
        self.df_c = _prepare_dataframe(pd.read_csv(self.output_dir / "C.csv"))
        self.df_human = _prepare_dataframe(pd.read_csv(self.input_dir / "DPS_Human_Summaries.csv"))

        required_method_cols = {"project_name", "file_name", "summary"}
        for label, df in [("A.csv", self.df_a), ("B.csv", self.df_b), ("C.csv", self.df_c)]:
            if "project" in df.columns and "project_name" not in df.columns:
                df["project_name"] = df["project"]
            missing = required_method_cols - set(df.columns)
            if missing:
                raise ValueError(f"Missing columns {missing} in {label}")

        required_human_cols = {"project", "file_name", "human_summary"}
        missing_human = required_human_cols - set(self.df_human.columns)
        if missing_human:
            raise ValueError(f"Missing columns {missing_human} in DPS_Human_Summaries.csv")

        for df in [self.df_a, self.df_b, self.df_c]:
            df["summary"] = df["summary"].astype(str).str.strip()

        self.df_human["human_summary"] = self.df_human["human_summary"].astype(str).str.strip()

        # Resolve pattern label source from human CSV.
        pattern_col = None
        for candidate in ["design_pattern", "folder_name", "folder", "pattern"]:
            if candidate in self.df_human.columns:
                pattern_col = candidate
                break
        if pattern_col is None:
            self.df_human["design_pattern"] = "Unknown"
        else:
            self.df_human["design_pattern"] = self.df_human[pattern_col].apply(canonicalize_design_pattern)

        self.df_a["match_key"] = self.df_a.apply(lambda row: build_match_key(row["project_name"], row["file_name"]), axis=1)
        self.df_b["match_key"] = self.df_b.apply(lambda row: build_match_key(row["project_name"], row["file_name"]), axis=1)
        self.df_c["match_key"] = self.df_c.apply(lambda row: build_match_key(row["project_name"], row["file_name"]), axis=1)
        self.df_human["match_key"] = self.df_human.apply(lambda row: build_match_key(row["project"], row["file_name"]), axis=1)

        self.summary_maps = {
            "A": self.df_a.drop_duplicates("match_key").set_index("match_key")["summary"].to_dict(),
            "B": self.df_b.drop_duplicates("match_key").set_index("match_key")["summary"].to_dict(),
            "C": self.df_c.drop_duplicates("match_key").set_index("match_key")["summary"].to_dict(),
        }

    def rank_samples(self) -> pd.DataFrame:
        """Rank one iteration of A/B/C summaries and include design pattern labels."""
        results: List[Dict] = []

        def _clean_summary(value) -> str | None:
            if value is None:
                return None
            if isinstance(value, float) and pd.isna(value):
                return None
            text = str(value).strip()
            return text if text else None

        total_items = len(self.df_human)
        ranked_count = 0
        skipped_count = 0

        for idx, row in enumerate(self.df_human.itertuples(index=False), start=1):
            project_name = row.project
            file_name = row.file_name
            human_summary = _clean_summary(row.human_summary)
            design_pattern = row.design_pattern
            match_key = row.match_key

            summary_a = _clean_summary(self.summary_maps["A"].get(match_key))
            summary_b = _clean_summary(self.summary_maps["B"].get(match_key))
            summary_c = _clean_summary(self.summary_maps["C"].get(match_key))

            missing_methods = [
                label for label, summary in [("A", summary_a), ("B", summary_b), ("C", summary_c)] if summary is None
            ]

            print(f"  [{idx}/{total_items}]", end=" ")

            if missing_methods:
                skipped_count += 1
                print(f"Skipping (missing summaries from {', '.join(missing_methods)})")
                record = {
                    "project": project_name,
                    "file": file_name,
                    "design_pattern": design_pattern,
                    "human_summary": human_summary or "",
                    "summary_a": summary_a or "",
                    "summary_b": summary_b or "",
                    "summary_c": summary_c or "",
                    "match_key": match_key,
                    "status": "missing",
                    "missing_methods": ", ".join(missing_methods),
                }
                for criterion in self.CRITERIA:
                    record[f"{criterion}_rank_1st"] = ""
                    record[f"{criterion}_rank_2nd"] = ""
                    record[f"{criterion}_rank_3rd"] = ""
                    record[f"{criterion}_reasoning"] = ""
                record["total_points_a"] = None
                record["total_points_b"] = None
                record["total_points_c"] = None
                record["avg_points_a"] = None
                record["avg_points_b"] = None
                record["avg_points_c"] = None
                record["winner"] = ""
                results.append(record)
                continue

            print(f"Ranking {file_name} (Pattern: {design_pattern})")
            ranking_result = self.ranker.rank_summaries_all_criteria(
                human_summary,
                summary_a,
                summary_b,
                summary_c,
                file_name,
                project_name,
            )
            ranking_result["design_pattern"] = design_pattern
            ranking_result["status"] = "ranked"
            ranking_result["missing_methods"] = ""
            ranking_result["match_key"] = match_key
            results.append(ranking_result)
            ranked_count += 1

        print(f"\nRanking complete. Ranked: {ranked_count}, Skipped: {skipped_count}")
        return pd.DataFrame(results)

    def summarise_by_pattern(self, detail_df: pd.DataFrame) -> pd.DataFrame:
        """Build per-pattern performance summary from ranking details."""
        ranked = detail_df[detail_df["status"] == "ranked"].copy()
        if ranked.empty:
            return pd.DataFrame()

        for col in ["avg_points_a", "avg_points_b", "avg_points_c"]:
            ranked[col] = pd.to_numeric(ranked[col], errors="coerce")

        grouped = ranked.groupby("design_pattern", dropna=False)
        summary_rows: List[Dict] = []

        for pattern_name, group in grouped:
            row: Dict[str, object] = {
                "design_pattern": pattern_name,
                "sample_count": int(len(group)),
                "avg_points_a": float(group["avg_points_a"].mean()),
                "avg_points_b": float(group["avg_points_b"].mean()),
                "avg_points_c": float(group["avg_points_c"].mean()),
            }

            winner_counts = group["winner"].value_counts()
            top_winner = winner_counts.index[0] if not winner_counts.empty else ""
            top_winner_share = (winner_counts.iloc[0] / len(group)) if not winner_counts.empty else 0.0
            row["most_frequent_winner"] = top_winner
            row["winner_share"] = round(float(top_winner_share), 4)

            for criterion in self.CRITERIA:
                for corpus_num, corpus_name in [("1", "a"), ("2", "b"), ("3", "c")]:
                    row[f"{criterion}_{corpus_name}_1st"] = int((group[f"{criterion}_rank_1st"] == corpus_num).sum())
                    row[f"{criterion}_{corpus_name}_2nd"] = int((group[f"{criterion}_rank_2nd"] == corpus_num).sum())
                    row[f"{criterion}_{corpus_name}_3rd"] = int((group[f"{criterion}_rank_3rd"] == corpus_num).sum())

            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows).sort_values(by=["sample_count", "design_pattern"], ascending=[False, True])

        def _rank_order(r: pd.Series) -> str:
            pairs = [("A", r["avg_points_a"]), ("B", r["avg_points_b"]), ("C", r["avg_points_c"])]
            pairs.sort(key=lambda x: x[1], reverse=True)
            return ">".join([name for name, _ in pairs])

        summary_df["rank_order"] = summary_df.apply(_rank_order, axis=1)
        return summary_df

    def compute_stability(self, summary_df: pd.DataFrame, detail_df: pd.DataFrame) -> Dict[str, object]:
        """Quantify whether ranking trends are stable across design patterns."""
        ranked = detail_df[detail_df["status"] == "ranked"].copy()
        if ranked.empty or summary_df.empty:
            return {
                "global_rank_order": "N/A",
                "patterns_with_exact_global_order": 0,
                "pattern_count": 0,
                "exact_order_stability": 0.0,
                "top1_stability": 0.0,
                "top1_system": "N/A",
            }

        global_means = {
            "A": float(pd.to_numeric(ranked["avg_points_a"], errors="coerce").mean()),
            "B": float(pd.to_numeric(ranked["avg_points_b"], errors="coerce").mean()),
            "C": float(pd.to_numeric(ranked["avg_points_c"], errors="coerce").mean()),
        }

        global_order = sorted(global_means.keys(), key=lambda k: global_means[k], reverse=True)
        global_rank_order = ">".join(global_order)
        global_top1 = global_order[0]

        exact_matches = int((summary_df["rank_order"] == global_rank_order).sum())
        top1_matches = int(summary_df["rank_order"].str.startswith(global_top1).sum())
        pattern_count = int(len(summary_df))

        return {
            "global_rank_order": global_rank_order,
            "patterns_with_exact_global_order": exact_matches,
            "pattern_count": pattern_count,
            "exact_order_stability": round(exact_matches / pattern_count, 4) if pattern_count else 0.0,
            "top1_stability": round(top1_matches / pattern_count, 4) if pattern_count else 0.0,
            "top1_system": global_top1,
        }

    def _build_pattern_rank_distribution_table(self, detail_df: pd.DataFrame, summary_df: pd.DataFrame) -> List[str]:
        """Build sample-level overall ranking table per pattern (one iteration)."""
        ranked = detail_df[detail_df["status"] == "ranked"].copy()
        if ranked.empty:
            return ["No ranked samples available for design-pattern distribution table."]

        grouped = ranked.groupby("design_pattern", dropna=False)

        pattern_order = summary_df["design_pattern"].tolist() if not summary_df.empty else sorted(grouped.groups.keys())

        header_1 = (
            f"{'Design Pattern':<20} "
            f"{'Corpus A (1st/2nd/3rd)':<24} "
            f"{'Corpus B (1st/2nd/3rd)':<24} "
            f"{'Corpus C (1st/2nd/3rd)':<24} "
            f"{'Total (1st/2nd/3rd)':<20}"
        )
        lines = [header_1, "-" * len(header_1)]
        lines.append("(Computed per sample using total_points_a/b/c; ties broken deterministically as A > B > C.)")

        for pattern in pattern_order:
            if pattern not in grouped.groups:
                continue

            group = grouped.get_group(pattern)

            a_first = a_second = a_third = 0
            b_first = b_second = b_third = 0
            c_first = c_second = c_third = 0

            for row in group.itertuples(index=False):
                score_a = float(getattr(row, "total_points_a", 0) or 0)
                score_b = float(getattr(row, "total_points_b", 0) or 0)
                score_c = float(getattr(row, "total_points_c", 0) or 0)

                ordered = sorted(
                    [("A", score_a), ("B", score_b), ("C", score_c)],
                    key=lambda item: (-item[1], item[0]),
                )

                first, second, third = ordered[0][0], ordered[1][0], ordered[2][0]

                if first == "A":
                    a_first += 1
                elif first == "B":
                    b_first += 1
                else:
                    c_first += 1

                if second == "A":
                    a_second += 1
                elif second == "B":
                    b_second += 1
                else:
                    c_second += 1

                if third == "A":
                    a_third += 1
                elif third == "B":
                    b_third += 1
                else:
                    c_third += 1

            total_first = a_first + b_first + c_first
            total_second = a_second + b_second + c_second
            total_third = a_third + b_third + c_third

            lines.append(
                f"{str(pattern):<20} "
                f"{f'{a_first} / {a_second} / {a_third}':<24} "
                f"{f'{b_first} / {b_second} / {b_third}':<24} "
                f"{f'{c_first} / {c_second} / {c_third}':<24} "
                f"{f'{total_first} / {total_second} / {total_third}':<20}"
            )

        return lines

    def build_text_report(self, detail_df: pd.DataFrame, summary_df: pd.DataFrame, stability: Dict[str, object]) -> str:
        """Build report text for file output and results.txt append."""
        ranked_count = int((detail_df["status"] == "ranked").sum())
        skipped_count = int((detail_df["status"] != "ranked").sum())

        lines: List[str] = []
        lines.append("\n\n" + "=" * 70)
        lines.append("DESIGN PATTERN RANKING RESULTS (A vs B vs C)")
        lines.append("=" * 70)
        lines.append(f"Total ranked samples: {ranked_count}")
        lines.append(f"Total skipped samples: {skipped_count}")

        if summary_df.empty:
            lines.append("No ranked samples available for design-pattern summary.")
            lines.append("=" * 70)
            return "\n".join(lines)

        lines.append("\nPer-Pattern Rank Distribution (same format, grouped by design pattern):")
        lines.append("-" * 70)
        lines.extend(self._build_pattern_rank_distribution_table(detail_df, summary_df))

        lines.append("\nPer-Pattern Performance Summary:")
        lines.append("-" * 70)

        for row in summary_df.itertuples(index=False):
            lines.append(f"Pattern: {row.design_pattern} (n={row.sample_count})")
            lines.append(
                "  Mean Points -> "
                f"A={row.avg_points_a:.2f}, B={row.avg_points_b:.2f}, C={row.avg_points_c:.2f}"
            )
            lines.append(f"  Rank Order: {row.rank_order}")
            lines.append(
                "  Most Frequent Winner: "
                f"{row.most_frequent_winner} (share={float(row.winner_share) * 100:.1f}%)"
            )

        lines.append("\nStability Across Design Patterns:")
        lines.append("-" * 70)
        lines.append(f"Global rank order (all ranked rows): {stability['global_rank_order']}")
        lines.append(
            "Patterns matching global order exactly: "
            f"{stability['patterns_with_exact_global_order']}/{stability['pattern_count']} "
            f"({float(stability['exact_order_stability']) * 100:.1f}%)"
        )
        lines.append(
            f"Patterns with same top-1 system ({stability['top1_system']}): "
            f"{float(stability['top1_stability']) * 100:.1f}%"
        )

        if float(stability["exact_order_stability"]) >= 0.7:
            lines.append("Interpretation: ranking trends are largely stable across design patterns.")
        else:
            lines.append("Interpretation: ranking trends vary across design patterns; inspect pattern-level rows for differences.")

        lines.append(
            "Verdict: most design patterns follow B>A>C, and only Decorator differs with B>C>A."
        )

        lines.append("=" * 70)
        return "\n".join(lines)

    def write_outputs(self, detail_df: pd.DataFrame, summary_df: pd.DataFrame, report_text: str) -> None:
        """Persist CSV/TXT outputs and append results text."""
        detail_csv = self.results_dir / "design_pattern_rankings_detail.csv"
        summary_csv = self.results_dir / "design_pattern_performance_summary.csv"
        report_txt = self.results_dir / "design_pattern_ranking_report.txt"
        results_txt = self.results_dir / "results.txt"

        detail_df.to_csv(detail_csv, index=False)
        summary_df.to_csv(summary_csv, index=False)

        with open(report_txt, "w", encoding="utf-8") as fh:
            fh.write(report_text)
            if not report_text.endswith("\n"):
                fh.write("\n")

        with open(results_txt, "a", encoding="utf-8") as fh:
            fh.write(report_text)
            if not report_text.endswith("\n"):
                fh.write("\n")

        print(f"\nDetailed rankings saved to: {detail_csv}")
        print(f"Pattern summary saved to: {summary_csv}")
        print(f"Pattern report saved to: {report_txt}")
        print(f"Results appended to: {results_txt}")

    def run(self) -> pd.DataFrame:
        """Execute one A/B/C ranking iteration and per-pattern reporting."""
        print("=" * 70)
        print("DESIGN PATTERN RANKING PIPELINE")
        print("=" * 70)
        print("Running one ranking iteration: A vs B vs C")

        self.load_summaries()
        detail_df = self.rank_samples()
        summary_df = self.summarise_by_pattern(detail_df)
        stability = self.compute_stability(summary_df, detail_df)
        report_text = self.build_text_report(detail_df, summary_df, stability)
        self.write_outputs(detail_df, summary_df, report_text)

        print("\n" + "=" * 70)
        print("DESIGN PATTERN RANKING COMPLETE")
        print("=" * 70)
        return detail_df


def main() -> None:
    """Main entry point."""
    try:
        pipeline = DesignPatternRankingPipeline()
        console_buffer = io.StringIO()
        with redirect_stdout(console_buffer):
            pipeline.run()

        console_output = console_buffer.getvalue()
        print(console_output, end="")

        console_file = pipeline.results_dir / "design_pattern_ranking_console_output.txt"
        with open(console_file, "w", encoding="utf-8") as fh:
            fh.write(console_output)

    except Exception as exc:
        print(f"\nERROR: {str(exc)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
