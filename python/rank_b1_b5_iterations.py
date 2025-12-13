"""
rank_b1_b5_iterations.py - Multi-iteration ranking for B1-B5 variants

This script runs 5 ranking iterations comparing:
- Iteration 1: A vs B1 vs C
- Iteration 2: A vs B2 vs C
- Iteration 3: A vs B3 vs C
- Iteration 4: A vs B4 vs C
- Iteration 5: A vs B5 vs C

Each iteration ranks summaries using the same 5 criteria as rank_summaries.py:
1. Accuracy
2. Conciseness
3. Adequacy
4. Code Context
5. Design Pattern Recognition

Results are aggregated and appended to b1_b5_vs_human_report.txt
"""

import json
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, List, Tuple

# Import the ranker from existing script
sys.path.insert(0, str(Path(__file__).parent))
from rank_summaries import MultiCriteriaRanker, build_match_key


class B1B5RankingPipeline:
    """Pipeline for ranking B1-B5 iterations against A and C."""
    
    CRITERIA = ['accuracy', 'conciseness', 'adequacy', 'code_context', 'design_patterns']
    
    def __init__(self):
        """Initialize pipeline with directory paths."""
        base_dir = Path(__file__).resolve().parent.parent
        self.base_dir = base_dir
        
        self.output_dir = (base_dir / 'output' / 'summary-output').resolve()
        self.input_dir = (base_dir / 'input').resolve()
        self.results_dir = (base_dir / 'evaluation-results').resolve()
        
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load API key
        env_path = base_dir / '.env'
        load_dotenv(env_path)
        api_key = os.getenv('OPENROUTER_API_KEY')
        
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in .env file")

        api_url = os.getenv('RANK_SUMMARIES_API_URL')
        if not api_url:
            raise ValueError("RANK_SUMMARIES_API_URL not found in .env file")

        model = os.getenv('RANK_SUMMARIES_MODEL')
        if not model:
            raise ValueError("RANK_SUMMARIES_MODEL not found in .env file")

        max_tokens_raw = os.getenv('RANK_SUMMARIES_MAX_TOKENS', '50')
        try:
            max_tokens = int(max_tokens_raw)
        except ValueError as exc:
            raise ValueError("RANK_SUMMARIES_MAX_TOKENS must be an integer") from exc

        prompts_path = self.base_dir / 'src' / 'main' / 'resources' / 'prompts.json'
        if not prompts_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompts_path}")

        with open(prompts_path, 'r', encoding='utf-8') as prompt_file:
            prompts_data = json.load(prompt_file)
        if not isinstance(prompts_data, dict):
            raise ValueError("prompts.json must contain a top-level JSON object")

        ranking_prompts = prompts_data.get('summary_ranking')
        if not isinstance(ranking_prompts, dict):
            raise ValueError("prompts.json is missing the summary_ranking section required by rank_b1_b5_iterations.py")

        self.ranker = MultiCriteriaRanker(
            api_key=api_key,
            api_url=api_url,
            model=model,
            prompts=ranking_prompts,
            max_tokens=max_tokens,
        )
        
    def load_corpus(self, filename: str) -> pd.DataFrame:
        """Load and normalize a corpus CSV file."""
        df = pd.read_csv(self.output_dir / filename)
        df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
        if 'project_name' not in df.columns:
            if 'project' in df.columns:
                df['project_name'] = df['project']
            elif 'projectname' in df.columns:
                df['project_name'] = df['projectname']
            else:
                raise ValueError(f"Missing project column in {filename}")
        df['summary'] = df['summary'].astype(str).str.strip()
        df['match_key'] = df.apply(
            lambda row: build_match_key(row['project_name'], row['file_name']), 
            axis=1
        )
        return df
    
    def load_human_summaries(self) -> pd.DataFrame:
        """Load human reference summaries."""
        df = pd.read_csv(self.input_dir / 'DPS_Human_Summaries.csv')
        df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
        df['human_summary'] = df['human_summary'].astype(str).str.strip()
        df['match_key'] = df.apply(
            lambda row: build_match_key(row['project'], row['file_name']), 
            axis=1
        )
        return df
    
    def rank_iteration(self, iteration_num: int, df_a: pd.DataFrame, 
                      df_b: pd.DataFrame, df_c: pd.DataFrame, 
                      df_human: pd.DataFrame) -> List[Dict]:
        """
        Rank summaries for one iteration (A vs Bi vs C).
        
        Returns:
            List of ranking results for this iteration
        """
        print(f"\n{'='*70}")
        print(f"ITERATION {iteration_num}: Ranking A vs B{iteration_num} vs C")
        print(f"{'='*70}")
        
        # Build lookup maps
        summary_map_a = df_a.drop_duplicates('match_key').set_index('match_key')['summary'].to_dict()
        summary_map_b = df_b.drop_duplicates('match_key').set_index('match_key')['summary'].to_dict()
        summary_map_c = df_c.drop_duplicates('match_key').set_index('match_key')['summary'].to_dict()
        
        results = []
        ranked_count = 0
        skipped_count = 0
        
        for idx, row in enumerate(df_human.itertuples(index=False), start=1):
            project_name = row.project
            file_name = row.file_name
            human_summary = str(row.human_summary).strip()
            match_key = row.match_key
            
            # Get summaries from each corpus
            summary_a = summary_map_a.get(match_key)
            summary_b = summary_map_b.get(match_key)
            summary_c = summary_map_c.get(match_key)
            
            # Check for missing summaries
            missing = []
            if not summary_a:
                missing.append('A')
            if not summary_b:
                missing.append(f'B{iteration_num}')
            if not summary_c:
                missing.append('C')
            
            if missing:
                skipped_count += 1
                print(f"  [{idx}/{len(df_human)}] Skipping {file_name} (missing: {', '.join(missing)})")
                
                # Record empty result
                result = {
                    'iteration': iteration_num,
                    'project': project_name,
                    'file': file_name,
                    'match_key': match_key,
                    'status': 'skipped',
                    'missing_methods': ', '.join(missing)
                }
                
                for criterion in self.CRITERIA:
                    result[f'{criterion}_rank_1st'] = None
                    result[f'{criterion}_rank_2nd'] = None
                    result[f'{criterion}_rank_3rd'] = None
                
                result['total_points_a'] = None
                result['total_points_b'] = None
                result['total_points_c'] = None
                
                results.append(result)
                continue
            
            # Rank this sample
            print(f"  [{idx}/{len(df_human)}] Ranking {file_name} (Project: {project_name})")
            
            ranking_result = self.ranker.rank_summaries_all_criteria(
                human_summary,
                summary_a,
                summary_b,
                summary_c,
                file_name,
                project_name
            )
            
            # Add iteration metadata
            ranking_result['iteration'] = iteration_num
            ranking_result['status'] = 'ranked'
            ranking_result['missing_methods'] = ''
            ranking_result['match_key'] = match_key
            
            results.append(ranking_result)
            ranked_count += 1
        
        print(f"\nIteration {iteration_num} complete: Ranked={ranked_count}, Skipped={skipped_count}")
        return results
    
    def compute_iteration_stats(self, results: List[Dict]) -> Dict:
        """
        Compute min/max/avg statistics for one iteration.
        
        Returns:
            Dict containing statistics for this iteration
        """
        df = pd.DataFrame([r for r in results if r['status'] == 'ranked'])
        
        if df.empty:
            return {
                'iteration': results[0]['iteration'],
                'ranked_count': 0,
                'skipped_count': len(results)
            }
        
        stats = {
            'iteration': results[0]['iteration'],
            'ranked_count': len(df),
            'skipped_count': len([r for r in results if r['status'] != 'ranked'])
        }
        
        # Per-criterion statistics
        for criterion in self.CRITERIA:
            first_col = f'{criterion}_rank_1st'
            second_col = f'{criterion}_rank_2nd'
            third_col = f'{criterion}_rank_3rd'
            
            first_counts = df[first_col].value_counts()
            second_counts = df[second_col].value_counts()
            third_counts = df[third_col].value_counts()
            
            for corpus_num, corpus_name in [('1', 'a'), ('2', 'b'), ('3', 'c')]:
                first_cnt = first_counts.get(corpus_num, 0)
                second_cnt = second_counts.get(corpus_num, 0)
                third_cnt = third_counts.get(corpus_num, 0)
                
                stats[f'{criterion}_{corpus_name}_1st'] = first_cnt
                stats[f'{criterion}_{corpus_name}_2nd'] = second_cnt
                stats[f'{criterion}_{corpus_name}_3rd'] = third_cnt
        
        # Overall points statistics
        stats['avg_points_a'] = df['total_points_a'].mean()
        stats['avg_points_b'] = df['total_points_b'].mean()
        stats['avg_points_c'] = df['total_points_c'].mean()
        
        return stats
    
    def compute_aggregate_stats(self, all_stats: List[Dict]) -> Dict:
        """
        Compute min/max/avg across all iterations.
        
        Returns:
            Dict containing aggregate statistics
        """
        df = pd.DataFrame(all_stats)
        
        aggregate = {
            'total_ranked': df['ranked_count'].sum(),
            'total_skipped': df['skipped_count'].sum()
        }
        
        # Per-criterion aggregates (min, max, avg across iterations)
        for criterion in self.CRITERIA:
            for corpus in ['a', 'b', 'c']:
                for position in ['1st', '2nd', '3rd']:
                    col = f'{criterion}_{corpus}_{position}'
                    if col in df.columns:
                        aggregate[f'{col}_min'] = df[col].min()
                        aggregate[f'{col}_max'] = df[col].max()
                        aggregate[f'{col}_avg'] = df[col].mean()
        
        # Overall points aggregates
        for corpus in ['a', 'b', 'c']:
            col = f'avg_points_{corpus}'
            aggregate[f'{col}_min'] = df[col].min()
            aggregate[f'{col}_max'] = df[col].max()
            aggregate[f'{col}_avg'] = df[col].mean()
        
        return aggregate
    
    def write_report(self, all_results: List[List[Dict]], all_stats: List[Dict], 
                    aggregate_stats: Dict):
        """Write ranking results to text report and append to b1_b5_vs_human_report.txt."""
        
        report_path = self.results_dir / 'b1_b5_vs_human_report.txt'
        
        # Read existing content if file exists
        existing_content = ""
        if report_path.exists():
            with open(report_path, 'r', encoding='utf-8') as f:
                existing_content = f.read()
        
        # Build ranking section
        ranking_section = []
        ranking_section.append("\n\n")
        ranking_section.append("="*70)
        ranking_section.append("\nRANKING RESULTS (A vs B1-B5 vs C)")
        ranking_section.append("\n" + "="*70 + "\n")
        
        # Per-iteration summary
        ranking_section.append("\nPer-Iteration Summary:")
        ranking_section.append("\n" + "-"*70 + "\n")
        
        for stats in all_stats:
            iter_num = stats['iteration']
            ranking_section.append(f"\nIteration {iter_num} (A vs B{iter_num} vs C):")
            ranking_section.append(f"  Ranked: {stats['ranked_count']}, Skipped: {stats['skipped_count']}")
            ranking_section.append(f"  Average Points: A={stats['avg_points_a']:.2f}, B={stats['avg_points_b']:.2f}, C={stats['avg_points_c']:.2f}\n")
            
            ranking_section.append("  Ranking by Criterion:")
            for criterion in self.CRITERIA:
                ranking_section.append(f"    {criterion.upper()}:")
                for corpus in ['a', 'b', 'c']:
                    first = stats.get(f'{criterion}_{corpus}_1st', 0)
                    second = stats.get(f'{criterion}_{corpus}_2nd', 0)
                    third = stats.get(f'{criterion}_{corpus}_3rd', 0)
                    ranking_section.append(f"      Corpus {corpus.upper()}: {first} first, {second} second, {third} third")
        
        # Aggregate statistics
        ranking_section.append("\n" + "-"*70)
        ranking_section.append("\nAggregate Statistics (Min/Max/Avg across all 5 iterations):")
        ranking_section.append("\n" + "-"*70 + "\n")
        
        ranking_section.append("Overall Average Points:")
        for corpus in ['a', 'b', 'c']:
            min_val = aggregate_stats[f'avg_points_{corpus}_min']
            max_val = aggregate_stats[f'avg_points_{corpus}_max']
            avg_val = aggregate_stats[f'avg_points_{corpus}_avg']
            ranking_section.append(f"  Corpus {corpus.upper()}: min={min_val:.2f}, max={max_val:.2f}, avg={avg_val:.2f}")
        
        ranking_section.append("\n\nRanking by Criterion (Min/Max/Avg across iterations):")
        for criterion in self.CRITERIA:
            ranking_section.append(f"\n  {criterion.upper()}:")
            for corpus in ['a', 'b', 'c']:
                ranking_section.append(f"    Corpus {corpus.upper()}:")
                for position in ['1st', '2nd', '3rd']:
                    col = f'{criterion}_{corpus}_{position}'
                    min_val = aggregate_stats.get(f'{col}_min', 0)
                    max_val = aggregate_stats.get(f'{col}_max', 0)
                    avg_val = aggregate_stats.get(f'{col}_avg', 0)
                    ranking_section.append(f"      {position}: min={min_val:.1f}, max={max_val:.1f}, avg={avg_val:.1f}")
        
        ranking_section.append("\n" + "="*70 + "\n")
        
        # Write combined content
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(existing_content)
            f.write(''.join(ranking_section))
        
        print(f"\nRanking results appended to: {report_path}")
    
    def run(self):
        """Execute the complete 5-iteration ranking pipeline."""
        print("="*70)
        print("B1-B5 MULTI-ITERATION RANKING PIPELINE")
        print("="*70)
        
        # Load fixed corpora (A and C)
        print("\nLoading fixed corpora...")
        df_a = self.load_corpus('A.csv')
        df_c = self.load_corpus('C.csv')
        df_human = self.load_human_summaries()
        
        print(f"  Loaded A.csv: {len(df_a)} entries")
        print(f"  Loaded C.csv: {len(df_c)} entries")
        print(f"  Loaded Human Summaries: {len(df_human)} entries")
        
        # Run 5 iterations
        all_results = []
        all_stats = []
        
        for iteration in range(1, 6):
            # Load Bi corpus for this iteration
            bi_filename = f'B{iteration}.csv'
            print(f"\nLoading {bi_filename}...")
            df_bi = self.load_corpus(bi_filename)
            print(f"  Loaded {bi_filename}: {len(df_bi)} entries")
            
            # Rank this iteration
            results = self.rank_iteration(iteration, df_a, df_bi, df_c, df_human)
            all_results.append(results)
            
            # Compute iteration statistics
            stats = self.compute_iteration_stats(results)
            all_stats.append(stats)
        
        # Compute aggregate statistics
        print("\n" + "="*70)
        print("COMPUTING AGGREGATE STATISTICS")
        print("="*70)
        aggregate_stats = self.compute_aggregate_stats(all_stats)
        
        # Save detailed results to CSV
        all_results_flat = []
        for iteration_results in all_results:
            all_results_flat.extend(iteration_results)
        
        results_df = pd.DataFrame(all_results_flat)
        detail_csv_path = self.results_dir / 'b1_b5_ranking_detail.csv'
        results_df.to_csv(detail_csv_path, index=False)
        print(f"\nDetailed results saved to: {detail_csv_path}")
        
        # Save summary statistics to CSV
        stats_df = pd.DataFrame(all_stats)
        summary_csv_path = self.results_dir / 'b1_b5_ranking_summary.csv'
        stats_df.to_csv(summary_csv_path, index=False)
        print(f"Summary statistics saved to: {summary_csv_path}")
        
        # Write report and append to existing b1_b5_vs_human_report.txt
        self.write_report(all_results, all_stats, aggregate_stats)
        
        print("\n" + "="*70)
        print("RANKING PIPELINE COMPLETE")
        print("="*70)


def main():
    """Main entry point."""
    try:
        pipeline = B1B5RankingPipeline()
        pipeline.run()
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
