# Design Pattern Summarizer

An automated tool for generating natural language summaries of design pattern implementations in Java code. Compares three different summary generation methods (NLG, SWUM, and LLM) against human-written summaries, with AI-powered ranking.

## Overview

This project analyzes Java implementations of common design patterns and generates descriptive summaries using three different approaches:

1. **NLG (Natural Language Generation)** - Template-based summary generation using code structure analysis
2. **SWUM (Software Word Use Model)** - Linguistic analysis of identifier names and code structure
3. **LLM (Large Language Model)** - AI-powered contextual summary generation using Mixtral-8x22B via OpenRouter

The generated summaries are evaluated against human-written summaries using:
- **Automated Metrics**: Cosine Similarity and BERTScore for quantitative assessment
- **LLM Ranking**: Llama3-70B ranks the three methods to determine which produces summaries closest to human references
 - **Multi-Criteria Ranking**: A Python pipeline evaluates summaries across five criteria with deterministic prompts

## Key Features

- **Automated Summary Generation**: Three different methods for creating class-level summaries
- **Design Pattern Detection**: Identifies common design patterns (Factory, Singleton, Observer, Visitor, Decorator, Facade, Adapter, Memento, Abstract Factory)
- **Comprehensive Evaluation**: Measures quality using Cosine Similarity and BERTScore metrics
- **LLM-Based Ranking**: Uses Llama3-70B to rank which method produces summaries most similar to human references
- **Multi-Project Support**: Analyzes implementations from multiple repositories
- **Visualization**: Generates comparison charts and detailed reports

## Project Structure

```
designpatternsummariser/
├── src/                          # Java source code
│   ├── common/                   # Shared utilities and pattern detection
│   ├── dps_app/                  # NLG summary generation
│   ├── dps_llm/                  # LLM-based summary generation (OpenRouter)
│   └── dps_swum/                 # SWUM-based summary generation
├── python/                       # Python evaluation scripts
│   ├── evaluate_summaries.py    # Main evaluation + LLM ranking script
│   ├── rank_summaries.py        # Multi-criteria ranking across A/B/C vs human
│   ├── evaluate_nlg_additional_metrics.py  # Additional MT metrics
│   └── visualize_comparative_results.py    # Visualization script
├── input/                        # Input code repositories
│   ├── AbdurRKhalid/            # Design pattern examples
│   ├── JamesZBL/                # Additional implementations
│   ├── spring-framework/        # Real-world patterns
│   └── DPS_Human_Summaries.csv  # Human-written summaries
├── output/                       # Generated summaries
│   ├── json-output/             # Parsed code structure (JSON)
│   └── summary-output/          # Generated summaries (CSV)
│       ├── dps_nlg.csv          # NLG summaries
│       ├── swum_summaries.csv   # SWUM summaries
│       └── llm_summaries.csv    # LLM summaries (OpenRouter)
├── evaluation-results/           # Evaluation outputs
│   ├── results.txt              # Detailed evaluation report
│   ├── evaluation_summary.txt   # Quick summary
│   ├── overall_comparison.csv   # Method comparison
│   ├── llm_summary_rankings.csv # LLM-based ranking results
│   ├── multi_criteria_rankings.csv # Per-file ranks for all criteria
│   └── ranking_console_output.txt # Full transcript of ranking run
│   ├── *_vs_human_*.csv         # Class & project scores
│   └── *.png                    # Visualization charts
├── pom.xml                       # Maven configuration
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Installation

### Prerequisites

- **Java**: JDK 11 or higher
- **Maven**: 3.6 or higher
- **Python**: 3.8 or higher
- **API Key**: OpenRouter API key (for LLM summaries)

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/najamnazar/designpatternsummariser.git
   cd designpatternsummariser
   ```

2. **Install Java dependencies**:
   ```bash
   mvn clean install
   ```

3. **Set up Python environment**:
   ```bash
   # Windows
   setup_python.bat
   
   # PowerShell
   .\setup_python.ps1
   ```

4. **Configure API Key** (for LLM summaries):
   Create a `.env` file in the root directory:
   ```
   OPENROUTER_API_KEY=your_api_key_here
   OPENROUTER_MODEL=mistralai/mixtral-8x22b-instruct
   OPENROUTER_MAX_TOKENS=256
   OPENROUTER_TEMPERATURE=0.2
   ```

## Usage

### Generate Summaries

Run all summary generation methods:

```bash
# Generate NLG summaries
mvn exec:java@dps-app

# Generate SWUM summaries
mvn exec:java@swum-pipeline

# Generate LLM summaries (OpenRouter - requires API key)
mvn exec:java@llm-summaries
```

### Evaluate Summaries

Compare generated summaries against human summaries with automated metrics and LLM ranking:

```bash
python python/evaluate_summaries.py
```

This script will:
1. Compute Cosine Similarity and BERTScore for each method vs human summaries
2. Use Llama3-70B to rank which method produces the most human-like summaries
3. Generate visualizations and detailed reports

Results will be saved in `evaluation-results/`.

### Multi-Criteria LLM Ranking (Python)

Evaluate each summary set (A/B/C) against human references across five criteria using deterministic LLM prompts. This produces per-criterion ranks and aggregate counts.

Prerequisites:
- `.env` in repo root with `OPENROUTER_API_KEY` and model settings
- Human summaries: `input/DPS_Human_Summaries.csv` (150 entries)
- Generated summary CSVs placed under `output/summary-output/` as A/B/C sets

Run (Windows PowerShell):
```powershell
# Ensure dependencies
python -m pip install -r requirements.txt

# Execute full corpus ranking (150 entries)
python python/rank_summaries.py

# Optional: limit to first 50 for a quick check
python python/rank_summaries.py --limit 50
```

Environment (.env):
```
OPENROUTER_API_KEY=sk-...
OPENROUTER_MODEL=meta-llama/llama-3.1-70b-instruct
OPENROUTER_TEMPERATURE=0
OPENROUTER_MAX_TOKENS=128
```

Criteria evaluated:
- Accuracy, Conciseness, Adequacy, Code Context, Design Patterns

Outputs:
- `evaluation-results/multi_criteria_rankings.csv` — per-file ranks, criteria columns, winner
- `evaluation-results/ranking_console_output.txt` — full console transcript with aggregate statistics

Aggregate example (from 150 entries):
- Accuracy winners: `B 139`, `A 1`, `C 9`
- Adequacy winners: `B 142`, `A 7`, `C 0`
- Overall winners: `B 132 / 149`, `C 5 / 149`, `A 5 / 149`

### Quick Start (Windows PowerShell)

Use these commands to install deps, configure the API, and run the ranking end-to-end.

```powershell
# 1) Clone and enter
git clone https://github.com/najamnazar/designpatternsummariser.git
cd designpatternsummariser

# 2) (Optional) Build Java parts
mvn -v
mvn clean install

# 3) Python deps
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 4) Configure OpenRouter
@"
OPENROUTER_API_KEY=sk-your-key
OPENROUTER_MODEL=meta-llama/llama-3.1-70b-instruct
OPENROUTER_TEMPERATURE=0
OPENROUTER_MAX_TOKENS=128
"@ | Set-Content -Encoding UTF8 .env

# 5) Optional quick test on 50 entries
python python/rank_summaries.py --limit 50

# 6) Full 150-entry ranking
python python/rank_summaries.py

# 7) View outputs
Get-Content evaluation-results/ranking_console_output.txt -Tail 100 | Out-String
```

### Maven Execution IDs

The project defines these execution targets:

- `@dps-app` - Run NLG summary generation
- `@swum-pipeline` - Run SWUM summary generation
- `@llm-summaries` - Run LLM summary generation

## Evaluation Results

### Performance Summary

Based on evaluation against 150 human-written summaries:

#### Automated Metrics

| Method | Cosine Similarity | BERT F1 | Combined Score | Rank |
|--------|------------------|---------|----------------|------|
| **LLM** | **0.3210** | 0.8622 | **0.5916** | 🥇 1st |
| **SWUM** | 0.2486 | **0.8642** | 0.5564 | 🥈 2nd |
| **NLG** | 0.1628 | 0.8423 | 0.5025 | 🥉 3rd |

#### LLM Ranking Results

Llama3-70B was asked to rank which summaries are most similar to human references:

- **Sample Size**: 20 files evaluated
- **Ranking Methodology**: Direct comparison with detailed reasoning
- **Results**: See `evaluation-results/llm_summary_rankings.csv`

### Key Findings

1. **LLM is Best Overall**: Achieves highest combined score (0.5916) with superior lexical alignment (cosine similarity nearly 2x that of NLG)

2. **SWUM Has Highest Precision**: BERT precision of 0.8889 indicates most accurate information extraction

3. **All Methods Capture Semantics**: BERT F1 scores (0.84-0.86) show all methods understand code meaning, despite vocabulary differences

4. **LLM Ranking Provides Qualitative Assessment**: Llama3-70B ranking adds human-like judgment to complement automated metrics

5. **Multi-Criteria Ranking Adds Robustness**: Deterministic prompts across five criteria reduce bias and provide granular insights. Full run artifacts are saved as CSV and transcript for auditability.

5. **Consistent Across Projects**: Relative performance remains stable across different codebases

### Detailed Results

See `evaluation-results/results.txt` for comprehensive analysis including:
- Per-class BERT scores
- Top/bottom performing classes
- Project-level breakdowns
- Comparative visualizations
- Methodology details

## Design Patterns Supported

The tool recognizes and generates summaries for:

- **Creational**: Factory Method, Abstract Factory, Singleton
- **Structural**: Adapter, Decorator, Facade
- **Behavioral**: Observer, Visitor, Memento

## Metrics Explained

### Cosine Similarity (0-1)
Measures lexical overlap between generated and human summaries using TF-IDF vectors. Higher values indicate more similar word choice and phrasing.

### BERT Score
Uses contextual embeddings to measure semantic similarity:
- **Precision**: Accuracy of generated content
- **Recall**: Coverage of human-written content  
- **F1**: Harmonic mean of precision and recall

### Combined Score
Average of Cosine Similarity and BERT F1, providing a balanced measure of lexical and semantic quality.

## Input Data

### Code Repositories

- **AbdurRKhalid**: Educational design pattern examples
- **JamesZBL**: Additional pattern implementations
- **Spring Framework**: Real-world enterprise patterns from Spring's codebase

### Human Summaries

150 manually written summaries in `input/DPS_Human_Summaries.csv`:
- Project name
- File name
- Design pattern type
- GitHub URL
- Human-written summary

## Output Files

### Generated Summaries (CSV)

Each method produces a CSV with:
- Project name/path
- Filename
- Generated summary

### Evaluation Results

- **Class-level scores**: Individual file comparisons with all metrics
- **Project-level scores**: Aggregated statistics per project
- **Overall comparison**: Cross-method performance summary
- **Visualizations**: Bar charts and metric comparisons
- **Detailed report**: `results.txt` with comprehensive analysis

## Configuration

### LLM Settings (.env)

```
OPENROUTER_API_KEY=sk-...           # Required
OPENROUTER_MODEL=mistralai/...      # Default: mixtral-8x22b-instruct
OPENROUTER_MAX_TOKENS=256           # Default: 256
OPENROUTER_TEMPERATURE=0.2          # Default: 0.2 (deterministic)
LLM_PROJECT_LIMIT=                  # Optional: limit projects processed
```

### Python Dependencies

Key packages in `requirements.txt`:
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `scikit-learn` - TF-IDF and cosine similarity
- `bert-score` - Semantic similarity evaluation
- `matplotlib`, `seaborn` - Visualizations
- `transformers` - BERT model access

## Development

### Building the Project

```bash
mvn clean compile
```

### Running Tests

```bash
mvn test
```

### Code Structure

- **Common utilities** (`common/`): Shared parsing, pattern detection
- **NLG pipeline** (`dps_app/`): Template-based generation
- **SWUM pipeline** (`dps_swum/`): Identifier analysis and linguistic modeling
- **LLM pipeline** (`dps_llm/`): API integration and prompt engineering

## Research Context

This project supports research in automated code documentation and design pattern understanding. The evaluation methodology compares automated approaches against human judgment to assess:

- **Readability**: How natural do generated summaries sound?
- **Accuracy**: Do summaries correctly describe code behavior?
- **Completeness**: Are key concepts and relationships captured?
- **Usefulness**: Would developers find these summaries helpful?

## Limitations

1. **Language Support**: Currently Java only
2. **Pattern Coverage**: Limited to 9 common patterns
3. **LLM Dependency**: Requires API key and internet connection
4. **Context Window**: LLM summaries limited to 256 tokens
5. **Evaluation Dataset**: Human summaries from specific projects only

## Future Work

- Expand to additional programming languages (Python, C#)
- Support more design patterns
- Fine-tune LLM for code summarization
- Develop hybrid approaches combining method strengths
- Create larger evaluation dataset
- Add interactive web interface

## Contributing

Contributions welcome! Areas of interest:

- Additional design pattern detectors
- Alternative summary generation methods
- Expanded evaluation metrics
- Support for more languages
- Improved visualization

## License

[Your License Here - e.g., MIT]

## Citation

If you use this work in research, please cite:

```
[Add citation information]
```

## Acknowledgments

- Human summaries provided by domain experts
- Design pattern examples from open-source repositories
- Evaluation methodology based on established NLG metrics
- OpenRouter for LLM API access

## Contact

[Add contact information]

## References

- Design Patterns: Elements of Reusable Object-Oriented Software (Gang of Four)
- SWUM: Software Word Use Model
- BERTScore: Evaluating Text Generation with BERT
- OpenRouter API Documentation

---

**Last Updated**: November 30, 2025  
**Version**: 1.0  
**Status**: Active Development
