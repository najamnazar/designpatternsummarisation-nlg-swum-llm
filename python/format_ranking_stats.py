"""
format_ranking_stats.py - Format detailed ranking statistics for each iteration

This script reads the b1_b5_ranking_summary.csv and formats the statistics
in a detailed, human-readable format showing the distribution of 1st, 2nd, 
and 3rd place rankings for each criterion and corpus.
"""

import pandas as pd
from pathlib import Path


def format_iteration_stats(iteration_num, row):
    """Format statistics for a single iteration."""
    output = []
    output.append(f"\n{'='*70}")
    output.append(f"ITERATION {iteration_num} (A vs B{iteration_num} vs C)")
    output.append(f"{'='*70}\n")
    
    output.append(f"Ranked: {int(row['ranked_count'])}, Skipped: {int(row['skipped_count'])}\n")
    
    # Criteria to display
    criteria = [
        ('ACCURACY', 'accuracy'),
        ('CONCISENESS', 'conciseness'),
        ('ADEQUACY', 'adequacy'),
        ('CODE_CONTEXT', 'code_context'),
        ('DESIGN_PATTERNS', 'design_patterns')
    ]
    
    # Display per-criterion statistics
    for display_name, key in criteria:
        output.append(f"{display_name}:")
        
        for corpus_name, corpus_key in [('A', 'a'), ('B', 'b'), ('C', 'c')]:
            first = int(row[f'{key}_{corpus_key}_1st'])
            second = int(row[f'{key}_{corpus_key}_2nd'])
            third = int(row[f'{key}_{corpus_key}_3rd'])
            output.append(f"  Corpus {corpus_name}: {first} first, {second} second, {third} third")
        output.append("")
    
    # Overall average points
    output.append("-" * 70)
    output.append("Overall Average Points (3=1st, 2=2nd, 1=3rd per criterion):")
    output.append(f"  Corpus A: {row['avg_points_a']:.2f}")
    output.append(f"  Corpus B: {row['avg_points_b']:.2f}")
    output.append(f"  Corpus C: {row['avg_points_c']:.2f}")
    
    return '\n'.join(output)


def compute_winner_distribution(detail_csv_path, iteration_num):
    """Compute winner distribution for a specific iteration."""
    df = pd.read_csv(detail_csv_path)
    
    # Filter for this iteration and ranked status
    iter_df = df[(df['iteration'] == iteration_num) & (df['status'] == 'ranked')]
    
    if iter_df.empty:
        return "\nNo ranked samples for winner distribution."
    
    winner_counts = iter_df['winner'].value_counts()
    total = len(iter_df)
    
    output = ["\n" + "-" * 70]
    output.append("Winner Distribution:")
    
    for winner, count in winner_counts.items():
        percentage = (count / total) * 100
        output.append(f"  {winner}: {count}/{total} ({percentage:.1f}%)")
    
    return '\n'.join(output)


def main():
    """Main entry point."""
    base_dir = Path(__file__).resolve().parent.parent
    results_dir = base_dir / 'evaluation-results'
    
    summary_csv = results_dir / 'b1_b5_ranking_summary.csv'
    detail_csv = results_dir / 'b1_b5_ranking_detail.csv'
    output_file = results_dir / 'b1_b5_ranking_detailed_stats.txt'
    
    # Read the summary CSV
    df = pd.read_csv(summary_csv)
    
    # Generate formatted output
    output_lines = []
    output_lines.append("="*70)
    output_lines.append("DETAILED RANKING STATISTICS FOR B1-B5 ITERATIONS")
    output_lines.append("="*70)
    
    # Format each iteration
    for idx, row in df.iterrows():
        iteration_num = int(row['iteration'])
        iteration_stats = format_iteration_stats(iteration_num, row)
        output_lines.append(iteration_stats)
        
        # Add winner distribution
        winner_dist = compute_winner_distribution(detail_csv, iteration_num)
        output_lines.append(winner_dist)
        output_lines.append("")
    
    # Compute aggregate statistics across all iterations
    output_lines.append("\n" + "="*70)
    output_lines.append("AGGREGATE STATISTICS ACROSS ALL 5 ITERATIONS")
    output_lines.append("="*70 + "\n")
    
    criteria = ['accuracy', 'conciseness', 'adequacy', 'code_context', 'design_patterns']
    
    for criterion in criteria:
        output_lines.append(f"\n{criterion.upper()}:")
        for corpus in ['a', 'b', 'c']:
            min_1st = df[f'{criterion}_{corpus}_1st'].min()
            max_1st = df[f'{criterion}_{corpus}_1st'].max()
            avg_1st = df[f'{criterion}_{corpus}_1st'].mean()
            
            min_2nd = df[f'{criterion}_{corpus}_2nd'].min()
            max_2nd = df[f'{criterion}_{corpus}_2nd'].max()
            avg_2nd = df[f'{criterion}_{corpus}_2nd'].mean()
            
            min_3rd = df[f'{criterion}_{corpus}_3rd'].min()
            max_3rd = df[f'{criterion}_{corpus}_3rd'].max()
            avg_3rd = df[f'{criterion}_{corpus}_3rd'].mean()
            
            output_lines.append(f"  Corpus {corpus.upper()}:")
            output_lines.append(f"    1st: min={min_1st:.1f}, max={max_1st:.1f}, avg={avg_1st:.1f}")
            output_lines.append(f"    2nd: min={min_2nd:.1f}, max={max_2nd:.1f}, avg={avg_2nd:.1f}")
            output_lines.append(f"    3rd: min={min_3rd:.1f}, max={max_3rd:.1f}, avg={avg_3rd:.1f}")
    
    output_lines.append("\n" + "-" * 70)
    output_lines.append("Overall Average Points - Average across iterations:")
    output_lines.append(f"  Corpus A: {df['avg_points_a'].mean():.2f}")
    output_lines.append(f"  Corpus B: {df['avg_points_b'].mean():.2f}")
    output_lines.append(f"  Corpus C: {df['avg_points_c'].mean():.2f}")
    
    output_lines.append("\n" + "-" * 70)
    output_lines.append("Range of Average Points:")
    output_lines.append(f"  Corpus A: min={df['avg_points_a'].min():.2f}, max={df['avg_points_a'].max():.2f}, avg={df['avg_points_a'].mean():.2f}")
    output_lines.append(f"  Corpus B: min={df['avg_points_b'].min():.2f}, max={df['avg_points_b'].max():.2f}, avg={df['avg_points_b'].mean():.2f}")
    output_lines.append(f"  Corpus C: min={df['avg_points_c'].min():.2f}, max={df['avg_points_c'].max():.2f}, avg={df['avg_points_c'].mean():.2f}")
    
    output_lines.append("\n" + "="*70)
    
    # Write to file
    full_output = '\n'.join(output_lines)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(full_output)
    
    print(f"Detailed statistics written to: {output_file}")
    print("\n" + full_output)


if __name__ == '__main__':
    main()
