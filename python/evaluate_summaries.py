#!/usr/bin/env python3
"""
Summary Evaluation Against Human Summaries
Evaluates NLG, SWUM, and LLM summaries against human-written summaries
using Cosine Similarity and BERTScore with visualization
"""

import argparse
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import logging as hf_logging

# ML/NLP libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bert_score import score as bert_score

# Configure plot style
sns.set_theme(style='whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# Silence noisy transformer weight-init warnings from HF
hf_logging.set_verbosity_error()


def normalize_project_identifier(value: str) -> str:
    """Normalize project identifiers to align variants like 'Spring Framework' and 'spring-framework'."""
    if not isinstance(value, str):
        return ""
    cleaned = value.strip().lower()
    cleaned = re.sub(r"[\s_]+", "-", cleaned)
    cleaned = re.sub(r"[^a-z0-9\-]+", "", cleaned)
    cleaned = re.sub(r"-+", "-", cleaned)
    return cleaned


def normalize_filename(value: str) -> str:
    """Normalize filenames by lowercasing, stripping extensions, and ignoring path separators."""
    if not isinstance(value, str):
        return ""
    cleaned = value.strip().lower().replace('\\', '/').split('/')[-1]
    cleaned = re.sub(r"\.[^.]+$", "", cleaned)
    return cleaned


def _cosine_similarity_pair(text_a: str, text_b: str) -> float:
    """Compute TF-IDF cosine similarity between two text snippets."""
    vectorizer = TfidfVectorizer()
    try:
        tfidf = vectorizer.fit_transform([text_a, text_b])
        score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0, 0]
        return float(score)
    except ValueError:
        return 0.0


def _bert_per_row(candidates: List[str], references: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute BERTScore precision/recall/F1 arrays for the supplied pairs."""
    precision, recall, f1 = bert_score(
        candidates,
        references,
        lang='en',
        rescale_with_baseline=False,
        verbose=False,
    )
    return (
        precision.cpu().numpy(),
        recall.cpu().numpy(),
        f1.cpu().numpy(),
    )


def extract_full_project_path_from_url(url: str, base_project: str) -> str:
    """
    Extract the full project path from GitHub URL
    Example: https://github.com/.../JamesZBL/adapter/Application.java -> JamesZBL/adapter
    """
    try:
        # Find the base project in the URL and extract everything after it until the filename
        if base_project in url:
            # Split by base_project and get what comes after
            after_project = url.split(base_project)[1]
            # Remove leading slash and get path components before the filename
            parts = after_project.strip('/').split('/')
            # Remove the filename (last part) and join with base project
            path_parts = [p for p in parts[:-1] if p and p != 'blob' and p != 'main']
            # Handle URL encoding (e.g., %20 for spaces)
            path_parts = [p.replace('%20', ' ') for p in path_parts]
            if path_parts:
                return f"{base_project}/{'/'.join(path_parts)}"
            return base_project
    except:
        pass
    return base_project


def extract_base_project_name(project_path: str) -> str:
    """Return the top-level project identifier ignoring nested folders and suffixes."""
    if not isinstance(project_path, str):
        return ""
    cleaned = project_path.strip().replace('\\', '/').replace('"', '')
    # Convert first underscore to a slash when no slash exists to expose the base project
    if '/' not in cleaned and '_' in cleaned:
        cleaned = cleaned.replace('_', '/', 1)
    segments = cleaned.split('/')
    return segments[0].strip() if segments else cleaned.strip()


def load_human_summaries(csv_path: Path) -> pd.DataFrame:
    """Load human summaries with both base project and derived project path information."""
    df = pd.read_csv(csv_path)

    # Normalize column names
    df.columns = df.columns.str.strip()

    # Ensure required columns exist
    required_cols = ['Project', 'File Name', 'Human Summary', 'URL']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Create normalized DataFrame
    result = df[['Project', 'File Name', 'Human Summary', 'URL']].copy()
    result.columns = ['base_project', 'filename', 'summary', 'url']
    result['base_project'] = result['base_project'].astype(str).str.strip()
    result['filename'] = result['filename'].astype(str).str.strip()
    result['summary'] = result['summary'].astype(str).str.strip()
    result['url'] = result['url'].astype(str).str.strip()

    # Extract full project path from URL for reference
    result['project'] = result.apply(
        lambda row: extract_full_project_path_from_url(row['url'], row['base_project']),
        axis=1
    )

    # Remove any rows with empty summaries
    result = result[result['summary'].str.len() > 0]

    # Return with both project identifiers available
    return result[['project', 'base_project', 'filename', 'summary']]


def load_generated_summaries(csv_path: Path, summary_col_name: str) -> pd.DataFrame:
    """Load NLG, SWUM, or LLM summaries from CSV files."""
    df = pd.read_csv(csv_path)

    # Normalize column names
    df.columns = df.columns.str.strip()

    # Find relevant columns
    project_col = None
    folder_col = None
    filename_col = None
    summary_col = None

    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ['project', 'project name', 'project_name', 'project path', 'project_path']:
            project_col = col
        elif col_lower in ['folder', 'folder name', 'folder_name', 'folder path', 'folder_path']:
            folder_col = col
        elif col_lower in ['filename', 'file name', 'file_name', 'class', 'class name']:
            filename_col = col
        elif col_lower == summary_col_name.lower() or col == summary_col_name:
            summary_col = col

    if not all([project_col, filename_col, summary_col]):
        available = ', '.join(df.columns)
        raise ValueError(f"Cannot find required columns in {csv_path}. Available columns: {available}")

    columns = [project_col]
    if folder_col:
        columns.append(folder_col)
    columns.extend([filename_col, summary_col])

    result = df[columns].copy()

    rename_map = {
        project_col: 'project',
        filename_col: 'filename',
        summary_col: 'summary'
    }
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

    # Remove any rows with empty summaries
    result = result[result['summary'].str.len() > 0]

    return result

def evaluate_method_vs_human(method_name: str, human_df: pd.DataFrame, method_df: pd.DataFrame, output_dir: Path) -> Dict[str, float]:
    """Evaluate a single method (NLG/SWUM/LLM) against human summaries using base project + filename matching."""

    print(f"\n{'='*60}")
    print(f"Evaluating {method_name} vs Human Summaries")
    print(f"{'='*60}")

    human_df = human_df.copy()
    method_df = method_df.copy()

    # Create composite key ignoring deep folder structure (match by base project + filename)
    human_df['normalized_base_project'] = human_df['base_project'].apply(normalize_project_identifier)
    method_df['normalized_base_project'] = method_df['base_project'].apply(normalize_project_identifier)

    human_df['normalized_filename'] = human_df['filename'].apply(normalize_filename)
    method_df['normalized_filename'] = method_df['filename'].apply(normalize_filename)

    human_df['match_key'] = human_df['normalized_base_project'] + '::' + human_df['normalized_filename']
    method_df['match_key'] = method_df['normalized_base_project'] + '::' + method_df['normalized_filename']

    human_dupes = human_df[human_df.duplicated('match_key', keep=False)]
    method_dupes = method_df[method_df.duplicated('match_key', keep=False)]

    # Assign positional index so duplicates align deterministically
    human_df['dup_index'] = human_df.groupby('match_key').cumcount()
    method_df['dup_index'] = method_df.groupby('match_key').cumcount()

    print(f"  Human summaries: {len(human_df)}")
    print(f"  {method_name} summaries: {len(method_df)}")

    # Inform about duplicate keys (multiple summaries for the same base project + filename)
    if not human_dupes.empty:
        print(f"  WARNING: Found {len(human_dupes)} duplicate entries in human summaries:")
        for key in human_dupes['match_key'].unique():
            print(f"    - {key}")

    if not method_dupes.empty:
        print(f"  WARNING: Found {len(method_dupes)} duplicate entries in {method_name} summaries:")
        for key in method_dupes['match_key'].unique():
            print(f"    - {key}")

    # Perform 1-to-1 merge on the composite key and duplicate index
    merged = human_df.merge(
        method_df,
        on=['match_key', 'dup_index'],
        how='inner',
        suffixes=('_human', '_method')
    )
    merged = merged.drop(columns=['dup_index'])
    
    if merged.empty:
        print(f"  ERROR: No matching entries found for {method_name}")
        print(f"  Sample human keys: {list(human_df['match_key'].head())}")
        print(f"  Sample {method_name} keys: {list(method_df['match_key'].head())}")
        return {}, pd.DataFrame(), pd.DataFrame()
    
    print(f"  Matched: {len(merged)} (exact 1-to-1 pairs)")
    
    # Calculate cosine similarity for each pair
    merged['cosine_similarity'] = merged.apply(
        lambda row: _cosine_similarity_pair(row['summary_human'], row['summary_method']),
        axis=1
    )
    
    # Calculate BERTScore
    candidates = merged['summary_method'].tolist()
    references = merged['summary_human'].tolist()
    
    print(f"  Computing BERTScore for {len(candidates)} pairs...")
    bert_p, bert_r, bert_f1 = _bert_per_row(candidates, references)
    
    merged['bert_precision'] = bert_p
    merged['bert_recall'] = bert_r
    merged['bert_f1'] = bert_f1
    
    # Save class-level results
    class_csv = output_dir / f'{method_name.lower()}_vs_human_class_scores.csv'
    output_df = pd.DataFrame({
        'base_project': merged['base_project_human'],
        'project_path': merged['project_human'],
        'filename': merged['filename_human'],
        'summary_human': merged['summary_human'],
        'summary_method': merged['summary_method'],
        'cosine_similarity': merged['cosine_similarity'],
        'bert_precision': merged['bert_precision'],
        'bert_recall': merged['bert_recall'],
        'bert_f1': merged['bert_f1']
    })
    output_df.to_csv(class_csv, index=False)
    print(f"  Saved: {class_csv}")
    
    # Aggregate project-level metrics using the base project name
    project_stats = merged.groupby('base_project_human').agg(
        classes=('filename_human', 'count'),
        avg_cosine=('cosine_similarity', 'mean'),
        avg_bert_precision=('bert_precision', 'mean'),
        avg_bert_recall=('bert_recall', 'mean'),
        avg_bert_f1=('bert_f1', 'mean')
    ).reset_index()
    
    project_stats.rename(columns={'base_project_human': 'project'}, inplace=True)
    
    project_stats['combined_score'] = (project_stats['avg_cosine'] + project_stats['avg_bert_f1']) / 2.0
    
    project_csv = output_dir / f'{method_name.lower()}_vs_human_project_scores.csv'
    project_stats.sort_values('combined_score', ascending=False).to_csv(project_csv, index=False)
    print(f"  Saved: {project_csv}")
    
    # Calculate overall metrics
    overall_metrics = {
        'method': method_name,
        'projects_evaluated': int(project_stats.shape[0]),
        'classes_evaluated': int(len(merged)),
        'avg_cosine': float(merged['cosine_similarity'].mean()),
        'avg_bert_precision': float(merged['bert_precision'].mean()),
        'avg_bert_recall': float(merged['bert_recall'].mean()),
        'avg_bert_f1': float(merged['bert_f1'].mean()),
        'combined_score': float((merged['cosine_similarity'].mean() + merged['bert_f1'].mean()) / 2.0),
        'cosine_std': float(merged['cosine_similarity'].std(ddof=0)),
        'bert_f1_std': float(merged['bert_f1'].std(ddof=0))
    }
    
    print(f"\n{method_name} Results:")
    print(f"  Classes evaluated: {overall_metrics['classes_evaluated']}")
    print(f"  Avg Cosine Similarity: {overall_metrics['avg_cosine']:.4f}")
    print(f"  Avg BERT Precision: {overall_metrics['avg_bert_precision']:.4f}")
    print(f"  Avg BERT Recall: {overall_metrics['avg_bert_recall']:.4f}")
    print(f"  Avg BERT F1: {overall_metrics['avg_bert_f1']:.4f}")
    print(f"  Combined Score: {overall_metrics['combined_score']:.4f}")
    
    return overall_metrics, project_stats, merged


def create_comparison_visualizations(all_results: List[Dict], output_dir: Path):
    """Create visualizations comparing all three methods"""
    
    methods_df = pd.DataFrame(all_results)
    
    if methods_df.empty:
        print("No results to visualize")
        return
    
    # 1. Bar chart comparing methods
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(methods_df))
    width = 0.15
    
    metrics = [
        ('avg_cosine', 'Cosine Similarity', 'steelblue'),
        ('avg_bert_precision', 'BERT Precision', 'lightcoral'),
        ('avg_bert_recall', 'BERT Recall', 'lightgreen'),
        ('avg_bert_f1', 'BERT F1', 'gold'),
        ('combined_score', 'Combined Score', 'mediumpurple')
    ]
    
    for i, (metric, label, color) in enumerate(metrics):
        offset = (i - 2) * width
        ax.bar(x + offset, methods_df[metric], width, label=label, color=color)
    
    ax.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score (0-1)', fontsize=12, fontweight='bold')
    ax.set_title('Summary Generation Methods vs Human Summaries', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods_df['method'], fontsize=11)
    ax.legend(fontsize=10, ncol=3)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    chart_file = output_dir / 'methods_comparison.png'
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {chart_file}")
    plt.close()
    
    # 2. Grouped metrics comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Cosine similarity
    axes[0].bar(methods_df['method'], methods_df['avg_cosine'], color='steelblue', alpha=0.7)
    axes[0].set_ylabel('Cosine Similarity', fontsize=11, fontweight='bold')
    axes[0].set_title('Cosine Similarity by Method', fontsize=12, fontweight='bold')
    axes[0].set_ylim(0, 1.0)
    axes[0].grid(axis='y', alpha=0.3)
    
    # BERT F1
    axes[1].bar(methods_df['method'], methods_df['avg_bert_f1'], color='gold', alpha=0.7)
    axes[1].set_ylabel('BERT F1', fontsize=11, fontweight='bold')
    axes[1].set_title('BERT F1 Score by Method', fontsize=12, fontweight='bold')
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    metrics_file = output_dir / 'metrics_comparison.png'
    plt.savefig(metrics_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {metrics_file}")
    plt.close()


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated summaries against human summaries.")
    parser.add_argument(
        '--human-csv',
        type=Path,
        default=Path('input/DPS_Human_Summaries.csv'),
        help='Path to human summaries CSV file'
    )
    parser.add_argument(
        '--nlg-csv',
        type=Path,
        default=Path('output/summary-output/dps_nlg.csv'),
        help='Path to NLG summaries CSV file'
    )
    parser.add_argument(
        '--swum-csv',
        type=Path,
        default=Path('output/summary-output/swum_summaries.csv'),
        help='Path to SWUM summaries CSV file'
    )
    parser.add_argument(
        '--llm-csv',
        type=Path,
        default=Path('output/summary-output/llm_summaries.csv'),
        help='Path to LLM summaries CSV file'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('evaluation-results'),
        help='Output directory for results'
    )
    
    args = parser.parse_args([] if argv is None else argv)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("Summary Evaluation Against Human Summaries")
    print(f"{'='*60}")
    
    # Load human summaries
    print(f"\nLoading human summaries from: {args.human_csv}")
    human_df = load_human_summaries(args.human_csv)
    print(f"Loaded {len(human_df)} human summaries")
    
    all_results = []
    
    # Evaluate NLG
    if args.nlg_csv.exists():
        print(f"\nLoading NLG summaries from: {args.nlg_csv}")
        nlg_df = load_generated_summaries(args.nlg_csv, 'Summary')
        print(f"Loaded {len(nlg_df)} NLG summaries")
        nlg_results, nlg_projects, nlg_merged = evaluate_method_vs_human('NLG', human_df, nlg_df, args.output_dir)
        if nlg_results:
            all_results.append(nlg_results)
    else:
        print(f"WARNING: NLG CSV not found: {args.nlg_csv}")
    
    # Evaluate SWUM
    if args.swum_csv.exists():
        print(f"\nLoading SWUM summaries from: {args.swum_csv}")
        swum_df = load_generated_summaries(args.swum_csv, 'Summary')
        print(f"Loaded {len(swum_df)} SWUM summaries")
        swum_results, swum_projects, swum_merged = evaluate_method_vs_human('SWUM', human_df, swum_df, args.output_dir)
        if swum_results:
            all_results.append(swum_results)
    else:
        print(f"WARNING: SWUM CSV not found: {args.swum_csv}")
    
    # Evaluate LLM
    if args.llm_csv.exists():
        print(f"\nLoading LLM summaries from: {args.llm_csv}")
        llm_df = load_generated_summaries(args.llm_csv, 'Summary')
        print(f"Loaded {len(llm_df)} LLM summaries")
        llm_results, llm_projects, llm_merged = evaluate_method_vs_human('LLM', human_df, llm_df, args.output_dir)
        if llm_results:
            all_results.append(llm_results)
    else:
        print(f"WARNING: LLM CSV not found: {args.llm_csv}")
    
    # Save overall comparison
    if all_results:
        overall_df = pd.DataFrame(all_results)
        overall_csv = args.output_dir / 'overall_comparison.csv'
        overall_df.to_csv(overall_csv, index=False)
        print(f"\nSaved overall comparison: {overall_csv}")
        
        # Create visualizations
        print("\nCreating comparison visualizations...")
        create_comparison_visualizations(all_results, args.output_dir)
        
        # Save summary statistics
        summary_file = args.output_dir / 'evaluation_summary.txt'
        with open(summary_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("EVALUATION SUMMARY: Generated vs Human Summaries\n")
            f.write("="*60 + "\n\n")
            for result in all_results:
                f.write(f"\n{result['method']}:\n")
                f.write(f"  Classes Evaluated: {result['classes_evaluated']}\n")
                f.write(f"  Avg Cosine Similarity: {result['avg_cosine']:.4f} (±{result['cosine_std']:.4f})\n")
                f.write(f"  Avg BERT Precision: {result['avg_bert_precision']:.4f}\n")
                f.write(f"  Avg BERT Recall: {result['avg_bert_recall']:.4f}\n")
                f.write(f"  Avg BERT F1: {result['avg_bert_f1']:.4f} (±{result['bert_f1_std']:.4f})\n")
                f.write(f"  Combined Score: {result['combined_score']:.4f}\n")
        
        results_file = args.output_dir / 'results.txt'
        formatted_overall = overall_df.copy()
        float_cols = [
            'avg_cosine',
            'avg_bert_precision',
            'avg_bert_recall',
            'avg_bert_f1',
            'combined_score',
            'cosine_std',
            'bert_f1_std'
        ]
        for col in float_cols:
            if col in formatted_overall.columns:
                formatted_overall[col] = formatted_overall[col].map(lambda x: f"{x:.4f}")

        with open(results_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("OVERALL ANALYSIS\n")
            f.write("="*60 + "\n\n")
            f.write(formatted_overall.to_string(index=False))
            f.write("\n\nDetailed Metrics by Method:\n")
            for result in all_results:
                f.write(f"\n{result['method']}\n")
                f.write(f"  Projects Evaluated: {result['projects_evaluated']}\n")
                f.write(f"  Classes Evaluated: {result['classes_evaluated']}\n")
                f.write(f"  Avg Cosine Similarity: {result['avg_cosine']:.4f}\n")
                f.write(f"  Avg BERT Precision: {result['avg_bert_precision']:.4f}\n")
                f.write(f"  Avg BERT Recall: {result['avg_bert_recall']:.4f}\n")
                f.write(f"  Avg BERT F1: {result['avg_bert_f1']:.4f}\n")
                f.write(f"  Combined Score: {result['combined_score']:.4f}\n")

        print(f"Saved summary: {summary_file}")
        print(f"Saved overall analysis: {results_file}")
        
        print(f"\n{'='*60}")
        print("EVALUATION COMPLETE!")
        print(f"{'='*60}")
        print(f"\nResults saved to: {args.output_dir}")
        print("\nGenerated files:")
        print("  - overall_comparison.csv")
        print("  - evaluation_summary.txt")
        print("  - methods_comparison.png")
        print("  - metrics_comparison.png")
        for result in all_results:
            method = result['method'].lower()
            print(f"  - {method}_vs_human_class_scores.csv")
            print(f"  - {method}_vs_human_project_scores.csv")
    else:
        print("\nNo evaluation results generated.")


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
