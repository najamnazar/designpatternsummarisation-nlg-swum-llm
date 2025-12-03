#!/usr/bin/env python3
"""
Summary Evaluation Against Human Summaries
Evaluates NLG, SWUM, and dps_llm (Mixtral) summaries against human-written summaries
using Cosine Similarity and BERTScore with violin plot visualizations.

This module now follows an object-oriented design to encapsulate the data-loading, evaluation,
visualisation, and orchestration responsibilities while retaining the CLI behaviour and outputs.
"""

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from bert_score import score as bert_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import logging as hf_logging

try:
    import requests
except ImportError:
    requests = None

try:
    from dotenv import load_dotenv
    # Load .env file from project root (parent directory of python/)
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(dotenv_path=env_path)
except ImportError:
    pass  # dotenv not installed, will use environment variables directly

# Configure plot style
sns.set_theme(style='whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# Silence noisy transformer weight-init warnings from HF
hf_logging.set_verbosity_error()


def normalize_project_identifier(value: str) -> str:
    """Normalize project identifiers (case-fold, replace spaces, keep alphanumerics and hyphens)."""
    if not isinstance(value, str):
        return ""
    cleaned = value.strip().lower()
    cleaned = re.sub(r"[\s_]+", "-", cleaned)
    cleaned = re.sub(r"[^a-z0-9\-]+", "", cleaned)
    return re.sub(r"-+", "-", cleaned)


def normalize_filename(value: str) -> str:
    """Normalize filenames by lowercasing, trimming paths, and removing extensions."""
    if not isinstance(value, str):
        return ""
    cleaned = value.strip().lower().replace('\\', '/').split('/')[-1]
    return re.sub(r"\.[^.]+$", "", cleaned)


def extract_full_project_path_from_url(url: str, base_project: str) -> str:
    """Extract the full project path from a GitHub URL, falling back to the base project on failure."""
    try:
        if base_project in url:
            after_project = url.split(base_project)[1]
            parts = after_project.strip('/').split('/')
            path_parts = [p for p in parts[:-1] if p and p not in {'blob', 'main'}]
            path_parts = [p.replace('%20', ' ') for p in path_parts]
            if path_parts:
                return f"{base_project}/{'/'.join(path_parts)}"
            return base_project
    except Exception:
        pass
    return base_project


def extract_base_project_name(project_path: str) -> str:
    """Return the top-level project identifier, ignoring nested folders and suffixes."""
    if not isinstance(project_path, str):
        return ""
    cleaned = project_path.strip().replace('\\', '/').replace('"', '')
    if '/' not in cleaned and '_' in cleaned:
        cleaned = cleaned.replace('_', '/', 1)
    segments = cleaned.split('/')
    return segments[0].strip() if segments else cleaned.strip()


class MetricsCalculator:
    """Provides text similarity metrics used throughout the evaluation pipeline."""

    @staticmethod
    def cosine_similarity(text_a: str, text_b: str) -> float:
        vectorizer = TfidfVectorizer()
        try:
            tfidf = vectorizer.fit_transform([text_a, text_b])
            score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0, 0]
            return float(score)
        except ValueError:
            return 0.0

    @staticmethod
    def bert_scores(candidates: List[str], references: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        try:
            precision, recall, f1 = bert_score(
                candidates,
                references,
                lang='en',
                rescale_with_baseline=False,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001 - propagate with context
            raise RuntimeError("BERTScore computation failed") from exc
        return precision.cpu().numpy(), recall.cpu().numpy(), f1.cpu().numpy()


class LLMRanker:
    """Uses Llama3 via OpenRouter API to rank generated summaries against human references."""

    def __init__(self, api_key: Optional[str] = None, model: str = "meta-llama/llama-3.1-70b-instruct") -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self._availability_checked = False
        
        # Don't print warnings here - will be handled when is_available() is called

    def is_available(self) -> bool:
        """Check if the ranker can be used."""
        available = bool(self.api_key and requests)
        
        if not self._availability_checked:
            self._availability_checked = True
            if not self.api_key:
                print("\nINFO: OPENROUTER_API_KEY not found. LLM ranking will be skipped.")
                print("      To enable ranking, set OPENROUTER_API_KEY in .env file or environment.")
            elif requests is None:
                print("\nINFO: 'requests' library not installed. LLM ranking will be skipped.")
                print("      Install with: pip install requests")
        
        return available

    def rank_summaries(
        self,
        human_summary: str,
        nlg_summary: str,
        swum_summary: str,
        llm_summary: str,
        filename: str = "",
    ) -> Dict[str, any]:
        """
        Ask Llama3 to rank the three generated summaries (NLG, SWUM, LLM) against the human reference.
        
        Returns a dict with:
            - ranking: List[str] - Ordered list of methods from best to worst (e.g., ['LLM', 'SWUM', 'NLG'])
            - reasoning: str - LLM's explanation for the ranking
            - error: Optional[str] - Error message if ranking failed
        """
        if not self.is_available():
            return {"ranking": [], "reasoning": "LLM ranking not available", "error": "API key or requests library missing"}

        prompt = self._build_ranking_prompt(human_summary, nlg_summary, swum_summary, llm_summary, filename)
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,  # Low temperature for consistency
                "max_tokens": 512,
            }
            
            response = requests.post(self.base_url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()
            
            # Parse the ranking from the response
            ranking, reasoning = self._parse_ranking_response(content)
            
            return {
                "ranking": ranking,
                "reasoning": reasoning,
                "error": None,
            }
            
        except Exception as exc:
            return {
                "ranking": [],
                "reasoning": "",
                "error": str(exc),
            }

    def _build_ranking_prompt(
        self,
        human_summary: str,
        nlg_summary: str,
        swum_summary: str,
        llm_summary: str,
        filename: str,
    ) -> str:
        """Build the prompt for ranking summaries."""
        file_info = f" for the class '{filename}'" if filename else ""
        
        return f"""You are an expert code documentation evaluator. Given a human-written reference summary and three automatically generated summaries{file_info}, rank the generated summaries from BEST to WORST based on their similarity to the human reference.

Consider:
1. Semantic similarity - Does it capture the same meaning?
2. Completeness - Does it cover the key points?
3. Accuracy - Is the information correct?
4. Clarity - Is it well-expressed?

**Human Reference Summary:**
{human_summary}

**Generated Summary A (NLG):**
{nlg_summary}

**Generated Summary B (SWUM):**
{swum_summary}

**Generated Summary C (LLM):**
{llm_summary}

Provide your ranking in this exact format:
RANKING: [First, Second, Third]
REASONING: Your brief explanation

Example:
RANKING: [LLM, SWUM, NLG]
REASONING: LLM captures all key concepts with proper context. SWUM identifies main patterns but lacks detail. NLG misses the design pattern context."""

    def _parse_ranking_response(self, content: str) -> Tuple[List[str], str]:
        """Parse the LLM's ranking response."""
        ranking = []
        reasoning = ""
        
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith("RANKING:"):
                # Extract ranking list
                ranking_str = line.split("RANKING:", 1)[1].strip()
                # Parse [First, Second, Third] format
                ranking_str = ranking_str.strip('[]')
                ranking = [r.strip() for r in ranking_str.split(',')]
            elif line.strip().startswith("REASONING:"):
                # Get reasoning - may span multiple lines
                reasoning = line.split("REASONING:", 1)[1].strip()
                # Collect remaining lines as part of reasoning
                if i + 1 < len(lines):
                    reasoning += ' ' + ' '.join(lines[i+1:])
                break
        
        # Validate ranking contains expected methods
        valid_methods = {'NLG', 'SWUM', 'LLM'}
        if not ranking or not all(r.upper() in valid_methods for r in ranking):
            # Fallback parsing - look for method names in order
            content_upper = content.upper()
            ranking = []
            for method in ['NLG', 'SWUM', 'LLM']:
                if method in content_upper:
                    idx = content_upper.index(method)
                    ranking.append((idx, method))
            ranking.sort()
            ranking = [m for _, m in ranking]
        
        return ranking, reasoning.strip()


class SummaryDataLoader:
    """Responsible for reading and normalising summary CSV inputs."""

    @staticmethod
    def load_human_summaries(csv_path: Path) -> pd.DataFrame:
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Human summaries file not found: {csv_path}") from exc
        except pd.errors.EmptyDataError as exc:
            raise ValueError(f"Human summaries file is empty: {csv_path}") from exc
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to read human summaries: {csv_path}") from exc

        try:
            df.columns = df.columns.str.strip()
            required_cols = ['Project', 'File Name', 'Human Summary', 'URL']
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                raise ValueError(f"Missing required column(s) {missing} in {csv_path}")

            result = df[['Project', 'File Name', 'Human Summary', 'URL']].copy()
            result.columns = ['base_project', 'filename', 'summary', 'url']
            for column in ['base_project', 'filename', 'summary', 'url']:
                result[column] = result[column].astype(str).str.strip()

            result['project'] = result.apply(
                lambda row: extract_full_project_path_from_url(row['url'], row['base_project']),
                axis=1,
            )
            result = result[result['summary'].str.len() > 0]
            return result[['project', 'base_project', 'filename', 'summary']]
        except Exception:
            raise

    @staticmethod
    def load_generated_summaries(csv_path: Path, summary_col_name: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Generated summaries file not found: {csv_path}") from exc
        except pd.errors.EmptyDataError as exc:
            raise ValueError(f"Generated summaries file is empty: {csv_path}") from exc
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to read generated summaries: {csv_path}") from exc

        try:
            df.columns = df.columns.str.strip()
            project_col = folder_col = filename_col = summary_col = None

            for col in df.columns:
                col_lower = col.lower()
                if col_lower in {'project', 'project name', 'project_name', 'project path', 'project_path'}:
                    project_col = col
                elif col_lower in {'folder', 'folder name', 'folder_name', 'folder path', 'folder_path'}:
                    folder_col = col
                elif col_lower in {'filename', 'file name', 'file_name', 'class', 'class name'}:
                    filename_col = col
                elif col_lower == summary_col_name.lower() or col == summary_col_name:
                    summary_col = col

            if not all([project_col, filename_col, summary_col]):
                available = ', '.join(df.columns)
                raise ValueError(
                    f"Cannot find required columns in {csv_path}. Available columns: {available}"
                )

            columns = [project_col]
            if folder_col:
                columns.append(folder_col)
            columns.extend([filename_col, summary_col])

            result = df[columns].copy()
            rename_map = {project_col: 'project', filename_col: 'filename', summary_col: 'summary'}
            if folder_col:
                rename_map[folder_col] = 'folder'
            result = result.rename(columns=rename_map)

            result['project'] = result['project'].astype(str).str.strip()
            if 'folder' in result.columns:
                result['folder'] = result['folder'].astype(str).str.strip()
            else:
                result['folder'] = ''
            result['filename'] = result['filename'].astype(str).str.strip()
            result['summary'] = result['summary'].astype(str).str.strip()
            result['base_project'] = result['project'].apply(extract_base_project_name)
            result = result[result['summary'].str.len() > 0]
            return result
        except Exception:
            raise


@dataclass
class MethodEvaluationResult:
    """Container for per-method evaluation artefacts."""

    method: str
    metrics: Dict[str, float]
    project_stats: pd.DataFrame
    merged: pd.DataFrame


@dataclass
class EvaluationConfig:
    """CLI configuration mapped into a strongly-typed structure."""

    human_csv: Path
    nlg_csv: Path
    swum_csv: Path
    dps_llm_csv: Path
    output_dir: Path

    def ensure_output_dir(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def method_sources(self) -> List[Tuple[str, Path]]:
        return [
            ('NLG', self.nlg_csv),
            ('SWUM', self.swum_csv),
            ('dps_llm', self.dps_llm_csv),
        ]


class SummaryEvaluator:
    """Handles similarity scoring and persistence for a single method."""

    def __init__(self, human_df: pd.DataFrame, output_dir: Path, metrics: MetricsCalculator) -> None:
        self.human_df = human_df.copy()
        self.output_dir = output_dir
        self.metrics = metrics

    def evaluate_method(
        self,
        method_name: str,
        method_df: pd.DataFrame,
        display_name: Optional[str] = None,
    ) -> Optional[MethodEvaluationResult]:
        friendly_name = display_name or method_name
        print(f"\n{'='*60}")
        print(f"Evaluating {friendly_name} vs Human Summaries")
        print(f"{'='*60}")

        human_df = self.human_df.copy()
        method_df = method_df.copy()

        human_df['normalized_base_project'] = human_df['base_project'].apply(normalize_project_identifier)
        method_df['normalized_base_project'] = method_df['base_project'].apply(normalize_project_identifier)
        human_df['normalized_filename'] = human_df['filename'].apply(normalize_filename)
        method_df['normalized_filename'] = method_df['filename'].apply(normalize_filename)
        human_df['match_key'] = human_df['normalized_base_project'] + '::' + human_df['normalized_filename']
        method_df['match_key'] = method_df['normalized_base_project'] + '::' + method_df['normalized_filename']
        human_df['dup_index'] = human_df.groupby('match_key').cumcount()
        method_df['dup_index'] = method_df.groupby('match_key').cumcount()

        print(f"  Human summaries: {len(human_df)}")
        print(f"  {friendly_name} summaries: {len(method_df)}")

        human_dupes = human_df[human_df.duplicated('match_key', keep=False)]
        method_dupes = method_df[method_df.duplicated('match_key', keep=False)]
        if not human_dupes.empty:
            print(f"  WARNING: Found {len(human_dupes)} duplicate entries in human summaries:")
            for key in human_dupes['match_key'].unique():
                print(f"    - {key}")
        if not method_dupes.empty:
            print(f"  WARNING: Found {len(method_dupes)} duplicate entries in {friendly_name} summaries:")
            for key in method_dupes['match_key'].unique():
                print(f"    - {key}")

        merged = human_df.merge(
            method_df,
            on=['match_key', 'dup_index'],
            how='inner',
            suffixes=('_human', '_method'),
        ).drop(columns=['dup_index'])

        if merged.empty:
            print(f"  ERROR: No matching entries found for {friendly_name}")
            print(f"  Sample human keys: {list(human_df['match_key'].head())}")
            print(f"  Sample {friendly_name} keys: {list(method_df['match_key'].head())}")
            return None

        print(f"  Matched: {len(merged)} (exact 1-to-1 pairs)")

        merged['cosine_similarity'] = merged.apply(
            lambda row: self.metrics.cosine_similarity(row['summary_human'], row['summary_method']),
            axis=1,
        )

        candidates = merged['summary_method'].tolist()
        references = merged['summary_human'].tolist()
        print(f"  Computing BERTScore for {len(candidates)} pairs (this may take 2-3 minutes)...")
        try:
            bert_p, bert_r, bert_f1 = self.metrics.bert_scores(candidates, references)
            print(f"  BERTScore computation complete")
        except RuntimeError as exc:
            print(f"  ERROR: {exc}")
            return None

        merged['bert_precision'] = bert_p
        merged['bert_recall'] = bert_r
        merged['bert_f1'] = bert_f1

        class_csv = self.output_dir / f'{method_name.lower()}_vs_human_class_scores.csv'
        output_df = pd.DataFrame(
            {
                'base_project': merged['base_project_human'],
                'project_path': merged['project_human'],
                'filename': merged['filename_human'],
                'summary_human': merged['summary_human'],
                'summary_method': merged['summary_method'],
                'cosine_similarity': merged['cosine_similarity'],
                'bert_precision': merged['bert_precision'],
                'bert_recall': merged['bert_recall'],
                'bert_f1': merged['bert_f1'],
            }
        )
        output_df.to_csv(class_csv, index=False)
        print(f"  Saved: {class_csv}")

        project_stats = merged.groupby('base_project_human').agg(
            classes=('filename_human', 'count'),
            avg_cosine=('cosine_similarity', 'mean'),
            avg_bert_precision=('bert_precision', 'mean'),
            avg_bert_recall=('bert_recall', 'mean'),
            avg_bert_f1=('bert_f1', 'mean'),
        ).reset_index()
        project_stats.rename(columns={'base_project_human': 'project'}, inplace=True)
        project_stats['combined_score'] = (
            project_stats['avg_cosine'] + project_stats['avg_bert_f1']
        ) / 2.0

        project_csv = self.output_dir / f'{method_name.lower()}_vs_human_project_scores.csv'
        project_stats.sort_values('combined_score', ascending=False).to_csv(project_csv, index=False)
        print(f"  Saved: {project_csv}")

        overall_metrics = {
            'method': method_name,
            'projects_evaluated': int(project_stats.shape[0]),
            'classes_evaluated': int(len(merged)),
            'avg_cosine': float(merged['cosine_similarity'].mean()),
            'avg_bert_precision': float(merged['bert_precision'].mean()),
            'avg_bert_recall': float(merged['bert_recall'].mean()),
            'avg_bert_f1': float(merged['bert_f1'].mean()),
            'combined_score': float(
                (merged['cosine_similarity'].mean() + merged['bert_f1'].mean()) / 2.0
            ),
            'cosine_std': float(merged['cosine_similarity'].std(ddof=0)),
            'bert_f1_std': float(merged['bert_f1'].std(ddof=0)),
        }

        print(f"\n{friendly_name} Results:")
        print(f"  Classes evaluated: {overall_metrics['classes_evaluated']}")
        print(f"  Avg Cosine Similarity: {overall_metrics['avg_cosine']:.4f}")
        print(f"  Avg BERT Precision: {overall_metrics['avg_bert_precision']:.4f}")
        print(f"  Avg BERT Recall: {overall_metrics['avg_bert_recall']:.4f}")
        print(f"  Avg BERT F1: {overall_metrics['avg_bert_f1']:.4f}")
        print(f"  Combined Score: {overall_metrics['combined_score']:.4f}")

        return MethodEvaluationResult(
            method=method_name,
            metrics=overall_metrics,
            project_stats=project_stats,
            merged=merged,
        )


class VisualizationManager:
    """Creates all violin plots for method and LLM comparisons."""

    def __init__(self, metrics: MetricsCalculator) -> None:
        self.metrics = metrics

    def create_visualizations(
        self,
        results: List[MethodEvaluationResult],
        output_dir: Path,
    ) -> None:
        if len(results) < 2:
            print("Insufficient data for comparison visualization (need at least 2 methods)")
            return

        self._create_all_methods_violin(results, output_dir)

    def _create_all_methods_violin(
        self,
        results: List[MethodEvaluationResult],
        output_dir: Path,
    ) -> None:
        combined_data = []
        method_order = []
        for result in results:
            if not result.merged.empty:
                subset = result.merged[['cosine_similarity', 'bert_f1']].copy()
                subset['method'] = result.method
                combined_data.append(subset)
                method_order.append(result.method)

        if not combined_data:
            print("No merged data available for visualization")
            return

        all_data = pd.concat(combined_data, ignore_index=True)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        colors = {
            'NLG': '#e74c3c',
            'SWUM': '#3498db',
            'dps_llm': '#f39c12',
        }

        sns.violinplot(
            data=all_data,
            x='method',
            y='cosine_similarity',
            ax=axes[0],
            order=method_order,
            palette=colors,
            hue='method',
            legend=False,
        )
        axes[0].set_title('Cosine Similarity Distribution by Method', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Method', fontsize=12)
        axes[0].set_ylabel('Cosine Similarity Score', fontsize=12)
        axes[0].grid(axis='y', alpha=0.3)
        for method in method_order:
            method_data = all_data[all_data['method'] == method]
            mean_val = method_data['cosine_similarity'].mean()
            x_pos = method_order.index(method)
            axes[0].plot(
                x_pos,
                mean_val,
                marker='D',
                markersize=8,
                color='darkred',
                zorder=3,
                label='Mean' if method == method_order[0] else '',
            )
        axes[0].legend(loc='upper left')

        sns.violinplot(
            data=all_data,
            x='method',
            y='bert_f1',
            ax=axes[1],
            order=method_order,
            palette=colors,
            hue='method',
            legend=False,
        )
        axes[1].set_title('BERTScore F1 Distribution by Method', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Method', fontsize=12)
        axes[1].set_ylabel('BERTScore F1 Score', fontsize=12)
        axes[1].grid(axis='y', alpha=0.3)
        for method in method_order:
            method_data = all_data[all_data['method'] == method]
            mean_val = method_data['bert_f1'].mean()
            x_pos = method_order.index(method)
            axes[1].plot(
                x_pos,
                mean_val,
                marker='D',
                markersize=8,
                color='darkred',
                zorder=3,
                label='Mean' if method == method_order[0] else '',
            )
        axes[1].legend(loc='upper left')

        plt.tight_layout()
        violin_file = output_dir / 'methods_comparison_violin_plots.png'
        plt.savefig(violin_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {violin_file}")
        plt.close()


class SummaryEvaluationPipeline:
    """Coordinates the end-to-end evaluation process."""

    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config
        self.loader = SummaryDataLoader()
        self.metrics = MetricsCalculator()
        self.visualizer = VisualizationManager(self.metrics)
        self.llm_ranker = LLMRanker()  # Initialize LLM ranker

    def run(self) -> None:
        print(f"\n{'='*60}")
        print("Summary Evaluation Against Human Summaries")
        print(f"{'='*60}")
        print("\nInitializing (loading ML models - this takes 30-60 seconds)...")

        self.config.ensure_output_dir()
        human_df = self._load_human_data()
        if human_df is None:
            return
        print(f"Loaded {len(human_df)} human summaries")

        evaluator = SummaryEvaluator(human_df, self.config.output_dir, self.metrics)
        results: List[MethodEvaluationResult] = []

        for method_name, csv_path in self.config.method_sources():
            display_name = method_name
            if method_name == 'dps_llm':
                display_name = 'dps_llm (Mixtral)'

            if not csv_path.exists():
                print(f"WARNING: {display_name} CSV not found: {csv_path}")
                continue

            try:
                method_df = self.loader.load_generated_summaries(csv_path, 'Summary')
            except Exception as exc:  # noqa: BLE001
                print(f"ERROR loading {display_name} summaries: {exc}")
                continue

            print(f"\nLoading {display_name} summaries from: {csv_path}")
            print(f"Loaded {len(method_df)} {display_name} summaries")

            result = evaluator.evaluate_method(method_name, method_df, display_name)
            if result is None:
                continue

            results.append(result)

        if not results:
            print("\nNo evaluation results generated.")
            return

        # Perform LLM-based ranking if available
        self._perform_llm_ranking(results)

        self._save_overall_comparison(results)
        print("\nCreating violin plot visualizations (30 seconds)...")
        self.visualizer.create_visualizations(results, self.config.output_dir)
        print("Visualizations complete")
        self._write_summary_files(results)
        self._print_completion(results)

    def _load_human_data(self) -> Optional[pd.DataFrame]:
        print(f"\nLoading human summaries from: {self.config.human_csv}")
        try:
            return self.loader.load_human_summaries(self.config.human_csv)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR loading human summaries: {exc}")
            return None

    def _perform_llm_ranking(self, results: List[MethodEvaluationResult]) -> None:
        """Use LLM to rank the three summary methods against human references."""
        if not self.llm_ranker.is_available():
            return

        print(f"\n{'='*60}")
        print("LLM-Based Summary Ranking (Llama3-70B)")
        print(f"{'='*60}")
        print(f"Using model: {self.llm_ranker.model}")
        print(f"API endpoint: {self.llm_ranker.base_url}")
        
        # Find the merged dataframes for each method
        method_dfs = {result.method: result.merged for result in results}
        
        # Check we have all three methods
        if not all(method in method_dfs for method in ['NLG', 'SWUM', 'dps_llm']):
            print("WARNING: Need all three methods (NLG, SWUM, dps_llm) for ranking. Skipping.")
            return

        # Merge all three on the match_key to find common files
        nlg_df = method_dfs['NLG'][['match_key', 'filename_human', 'base_project_human', 'summary_human', 'summary_method']].copy()
        nlg_df.rename(columns={'summary_method': 'summary_nlg'}, inplace=True)
        
        swum_df = method_dfs['SWUM'][['match_key', 'summary_method']].copy()
        swum_df.rename(columns={'summary_method': 'summary_swum'}, inplace=True)
        
        llm_df = method_dfs['dps_llm'][['match_key', 'summary_method']].copy()
        llm_df.rename(columns={'summary_method': 'summary_llm'}, inplace=True)

        # Inner join to get files present in all three methods
        combined = nlg_df.merge(swum_df, on='match_key', how='inner')
        combined = combined.merge(llm_df, on='match_key', how='inner')

        if combined.empty:
            print("WARNING: No common files found across all three methods. Skipping ranking.")
            return

        print(f"Found {len(combined)} files common to all three methods")
        print("Calling Llama3 API for ranking...")

        rankings_data = []
        # Use smaller sample for quick testing - set to 5 for now, can increase to 20 later
        sample_size = min(5, len(combined))  # Rank a sample to avoid excessive API calls
        
        print(f"\nRanking {sample_size} files (this may take 30-60 seconds)...")
        for idx, (_, row) in enumerate(combined.head(sample_size).iterrows()):
            print(f"  [{idx + 1}/{sample_size}] Ranking: {row['filename_human']}...")
            
            ranking_result = self.llm_ranker.rank_summaries(
                human_summary=row['summary_human'],
                nlg_summary=row['summary_nlg'],
                swum_summary=row['summary_swum'],
                llm_summary=row['summary_llm'],
                filename=row['filename_human'],
            )
            
            if ranking_result['error']:
                print(f"    ✗ ERROR: {ranking_result['error']}")
                continue
            
            rankings_data.append({
                'base_project': row['base_project_human'],
                'filename': row['filename_human'],
                'human_summary': row['summary_human'],
                'nlg_summary': row['summary_nlg'],
                'swum_summary': row['summary_swum'],
                'llm_summary': row['summary_llm'],
                'ranking': ', '.join(ranking_result['ranking']),
                'first_place': ranking_result['ranking'][0] if ranking_result['ranking'] else '',
                'second_place': ranking_result['ranking'][1] if len(ranking_result['ranking']) > 1 else '',
                'third_place': ranking_result['ranking'][2] if len(ranking_result['ranking']) > 2 else '',
                'nlg_score': self._calculate_ranking_score('NLG', ranking_result['ranking']),
                'swum_score': self._calculate_ranking_score('SWUM', ranking_result['ranking']),
                'llm_score': self._calculate_ranking_score('LLM', ranking_result['ranking']),
                'reasoning': ranking_result['reasoning'],
            })
            
            print(f"    Result: {', '.join(ranking_result['ranking'])}")

        if not rankings_data:
            print("No rankings were successfully generated.")
            return

        # Save rankings to CSV
        rankings_df = pd.DataFrame(rankings_data)
        ranking_csv = self.config.output_dir / 'llm_summary_rankings.csv'
        rankings_df.to_csv(ranking_csv, index=False)
        print(f"\nSaved LLM rankings: {ranking_csv}")

        # Compute statistics
        first_place_counts = rankings_df['first_place'].value_counts()
        print("\nRanking Results (First Place):")
        for method, count in first_place_counts.items():
            percentage = (count / len(rankings_df)) * 100
            print(f"  {method}: {count}/{len(rankings_df)} ({percentage:.1f}%)")
        
        # Compute average scores
        print("\nAverage Ranking Scores (3=best, 1=worst):")
        for method in ['NLG', 'SWUM', 'LLM']:
            score_col = f'{method.lower()}_score'
            if score_col in rankings_df.columns:
                avg_score = rankings_df[score_col].mean()
                print(f"  {method}: {avg_score:.2f}")

    def _calculate_ranking_score(self, method: str, ranking: List[str]) -> int:
        """Calculate score for a method based on its position in ranking (3=1st, 2=2nd, 1=3rd, 0=not ranked)."""
        try:
            # Normalize method names for comparison
            normalized_ranking = [r.strip().upper() for r in ranking]
            normalized_method = method.strip().upper()
            
            position = normalized_ranking.index(normalized_method)
            # 1st place = 3 points, 2nd = 2 points, 3rd = 1 point
            return 3 - position
        except (ValueError, AttributeError):
            return 0  # Method not found in ranking

    def _save_overall_comparison(self, results: List[MethodEvaluationResult]) -> None:
        overall_df = pd.DataFrame(result.metrics for result in results)
        overall_csv = self.config.output_dir / 'overall_comparison.csv'
        overall_df.to_csv(overall_csv, index=False)
        print(f"\nSaved overall comparison: {overall_csv}")

    def _write_summary_files(self, results: List[MethodEvaluationResult]) -> None:
        summary_file = self.config.output_dir / 'evaluation_summary.txt'
        results_file = self.config.output_dir / 'results.txt'

        try:
            with open(summary_file, 'w', encoding='utf-8') as handle:
                handle.write("=" * 60 + "\n")
                handle.write("EVALUATION SUMMARY: Generated vs Human Summaries\n")
                handle.write("=" * 60 + "\n\n")
                for result in results:
                    method_info = ""
                    if result.method == 'dps_llm':
                        method_info = " (Mixtral-8x22B)"
                    metrics = result.metrics
                    handle.write(f"\n{result.method}{method_info}:\n")
                    handle.write(f"  Classes Evaluated: {metrics['classes_evaluated']}\n")
                    handle.write(
                        f"  Avg Cosine Similarity: {metrics['avg_cosine']:.4f} (±{metrics['cosine_std']:.4f})\n"
                    )
                    handle.write(f"  Avg BERT Precision: {metrics['avg_bert_precision']:.4f}\n")
                    handle.write(f"  Avg BERT Recall: {metrics['avg_bert_recall']:.4f}\n")
                    handle.write(
                        f"  Avg BERT F1: {metrics['avg_bert_f1']:.4f} (±{metrics['bert_f1_std']:.4f})\n"
                    )
                    handle.write(f"  Combined Score: {metrics['combined_score']:.4f}\n")

            formatted_overall = pd.DataFrame(result.metrics for result in results)
            float_cols = [
                'avg_cosine',
                'avg_bert_precision',
                'avg_bert_recall',
                'avg_bert_f1',
                'combined_score',
                'cosine_std',
                'bert_f1_std',
            ]
            for col in float_cols:
                if col in formatted_overall.columns:
                    formatted_overall[col] = formatted_overall[col].map(lambda x: f"{x:.4f}")

            with open(results_file, 'w', encoding='utf-8') as handle:
                handle.write("=" * 60 + "\n")
                handle.write("OVERALL ANALYSIS\n")
                handle.write("=" * 60 + "\n\n")
                handle.write(formatted_overall.to_string(index=False))
                handle.write("\n\nDetailed Metrics by Method:\n")
                for result in results:
                    method_info = ""
                    if result.method == 'dps_llm':
                        method_info = " (Mixtral-8x22B)"
                    metrics = result.metrics
                    handle.write(f"\n{result.method}{method_info}\n")
                    handle.write(f"  Projects Evaluated: {metrics['projects_evaluated']}\n")
                    handle.write(f"  Classes Evaluated: {metrics['classes_evaluated']}\n")
                    handle.write(f"  Avg Cosine Similarity: {metrics['avg_cosine']:.4f}\n")
                    handle.write(f"  Avg BERT Precision: {metrics['avg_bert_precision']:.4f}\n")
                    handle.write(f"  Avg BERT Recall: {metrics['avg_bert_recall']:.4f}\n")
                    handle.write(f"  Avg BERT F1: {metrics['avg_bert_f1']:.4f}\n")
                    handle.write(f"  Combined Score: {metrics['combined_score']:.4f}\n")
        except OSError as exc:
            print(f"ERROR writing summary files: {exc}")
            return

        print(f"Saved summary: {summary_file}")
        print(f"Saved overall analysis: {results_file}")

    def _print_completion(self, results: List[MethodEvaluationResult]) -> None:
        print(f"\n{'='*60}")
        print("EVALUATION COMPLETE!")
        print(f"{'='*60}")
        print(f"\nResults saved to: {self.config.output_dir}")
        print("\nGenerated files:")
        print("  - overall_comparison.csv")
        print("  - evaluation_summary.txt")
        print("  - methods_comparison_violin_plots.png")
        for result in results:
            method = result.method.lower()
            print(f"  - {method}_vs_human_class_scores.csv")
            print(f"  - {method}_vs_human_project_scores.csv")


def parse_arguments(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated summaries against human summaries.")
    parser.add_argument(
        '--human-csv',
        type=Path,
        default=Path('input/DPS_Human_Summaries.csv'),
        help='Path to human summaries CSV file',
    )
    parser.add_argument(
        '--nlg-csv',
        type=Path,
        default=Path('output/summary-output/dps_nlg.csv'),
        help='Path to NLG summaries CSV file',
    )
    parser.add_argument(
        '--swum-csv',
        type=Path,
        default=Path('output/summary-output/swum_summaries.csv'),
        help='Path to SWUM summaries CSV file',
    )
    parser.add_argument(
        '--dps-llm-csv',
        type=Path,
        default=Path('output/summary-output/llm_summaries.csv'),
        help='Path to dps_llm (Mixtral) summaries CSV file',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('evaluation-results'),
        help='Output directory for results',
    )
    return parser.parse_args([] if argv is None else argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_arguments(argv)
    config = EvaluationConfig(
        human_csv=args.human_csv,
        nlg_csv=args.nlg_csv,
        swum_csv=args.swum_csv,
        dps_llm_csv=args.dps_llm_csv,
        output_dir=args.output_dir,
    )
    pipeline = SummaryEvaluationPipeline(config)
    pipeline.run()


if __name__ == '__main__':
    import sys

    main(sys.argv[1:])
