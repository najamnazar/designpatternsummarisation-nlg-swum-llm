# Design Pattern Summariser

An automated design pattern summarisation platform that generates, ranks, and evaluates natural-language summaries for Java design-pattern implementations. Three summarisation strategies - template-driven NLG, identifier-aware SWUM, and OpenRouter-hosted LLMs - are benchmarked against 150 human-written references using automated metrics and LLM judgement.

## Overview

The pipeline analyses curated Java projects that exemplify the Gang-of-Four patterns and produces class-level summaries via:

1. **NLG (Natural Language Generation)** - deterministic templates informed by AST and design-pattern cues.
2. **SWUM (Software Word Use Model)** - linguistic modelling of identifier semantics plus lightweight static analysis.
3. **LLM (Large Language Model)** - Mixtral-8x22B (and swap-in OpenRouter models) prompted with code context for richer prose.

Outputs are cross-checked against expert summaries through cosine similarity, BERTScore, and LLM-facing ranking pipelines, ensuring quantitative and qualitative coverage.

## Capabilities

- Automated detection of nine core design patterns across several open-source corpora.
- Parallel generation of class summaries using Java-based NLG, SWUM, and LLM executables.
- Python evaluation suite covering cosine similarity, BERT precision or recall or F1, multi-project aggregations, and prompt-engineered rankings.
- Multi-criteria, rubric-driven LLM judgements (accuracy, conciseness, adequacy, code context, pattern clarity).
- Rich artefact trail (`evaluation-results/`) with CSVs, transcripts, visualisations, and narrative reports.

## Repository Layout

```
designpatternsummarisation-nlg-swum-llm/
├── src/main/java/
│   ├── common/              # Shared AST tooling and detection heuristics
│   ├── dps_app/             # NLG generator (Maven exec: dps-app)
│   ├── dps_swum/            # SWUM-based summariser (exec: swum-pipeline)
│   └── dps_llm/             # OpenRouter client and prompt orchestration
├── python/                  # Evaluation and ranking scripts
│   ├── evaluate_summaries.py
│   ├── rank_summaries.py
│   ├── evaluate_b1_b5.py
│   ├── analyze_conciseness.py
│   └── summary_length_stats.py (plus LLM prompt variant)
├── input/
│   ├── DPS_Human_Summaries.csv
│   ├── AbdurRKhalid/ (pattern exemplars)
│   ├── JamesZBL/
│   └── spring-framework/ (selected Spring modules)
├── output/
│   ├── json-output/         # Parsed code structure
│   └── summary-output/      # Per-method CSVs and experiment shards
├── evaluation-results/      # Metrics, rankings, iteration logs
├── pom.xml
├── requirements.txt
└── README.md
```

## Prerequisites

- **Java**: JDK 11+ (tested up to JDK 21).
- **Maven**: 3.6+ with the `exec-maven-plugin` enabled.
- **Python**: 3.10+ recommended for `transformers` support.
- **OpenRouter API key**: required for LLM generation and ranking workflows.

## Quick Start

```powershell
git clone https://github.com/najamnazar/designpatternsummarisation-nlg-swum-llm.git
cd designpatternsummarisation-nlg-swum-llm

# Java build (all pipelines)
mvn clean install

# Python environment
python -m venv .venv
.\\.venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# OpenRouter credentials
"OPENROUTER_API_KEY=sk-your-key`nOPENROUTER_MODEL=mistralai/mixtral-8x22b-instruct`nOPENROUTER_TEMPERATURE=0.2`nOPENROUTER_MAX_TOKENS=256" | Set-Content -Encoding UTF8 .env
```

## Summary Generation (Java)

```powershell
# Template-driven NLG summaries
mvn exec:java@dps-app

# Identifier-aware SWUM summaries
mvn exec:java@swum-pipeline

# LLM summaries via OpenRouter (requires .env)
mvn exec:java@llm-summaries
```

Each execution iterates over the curated repositories listed under `input/`, detects design-pattern roles, and exports consolidated CSVs to `output/summary-output/` (one per method or experiment iteration).

## Evaluation Pipelines (Python)

### Automated metrics and LLM tie-breaker

```powershell
python python/evaluate_summaries.py
```

The script aligns each generated summary with its human reference, computes cosine similarity and BERTScore (precision, recall, F1), and queries Llama3-70B via OpenRouter to adjudicate ties. Results appear under `evaluation-results/` as:

- `results.txt` - narrative analysis with per-pattern insights.
- `overall_comparison.csv` - aggregate metrics per method.
- `*_vs_human_*.csv` - project and class level breakdowns.

### Multi-criteria deterministic ranking

```powershell
python python/rank_summaries.py --limit 150        # Full corpus
python python/rank_summaries.py --limit 50         # Smoke test
```

Inputs expected:

- `output/summary-output/A.csv` - NLG corpus.
- `output/summary-output/B.csv` - LLM corpus.
- `output/summary-output/C.csv` - SWUM corpus.
- `input/DPS_Human_Summaries.csv` - gold references.

The script issues five rubric-aligned prompts per file (accuracy, conciseness, adequacy, code context, pattern articulation) using zero-temperature settings for determinism. Artefacts include:

- `evaluation-results/multi_criteria_rankings.csv` - per-file rubric scores and winners.
- `evaluation-results/ranking_console_output.txt` - reproducibility log.

### Iteration tracking and ablations

- `python/rank_b1_b5_iterations.py` - compares five prompt or parameter iterations.
- `python/evaluate_iterations.py` - longitudinal metrics to detect drift.
- `evaluation-results/iteration_*.txt` - preserved LLM outputs for auditing.

## Configuration

### Environment variables (.env)

```
OPENROUTER_API_KEY=sk-xxxxxxxxxxxxxxxx
OPENROUTER_MODEL=mistralai/mixtral-8x22b-instruct
OPENROUTER_TEMPERATURE=0.2
OPENROUTER_MAX_TOKENS=256
LLM_PROJECT_LIMIT=
OPENROUTER_BASE_URL=
```

### Python dependencies

Key packages from `requirements.txt`:

- `pandas`, `numpy`, `scikit-learn` - tabular processing and cosine similarity.
- `bert-score`, `transformers`, `torch` - semantic similarity computation.
- `matplotlib`, `seaborn` - visualisation helpers.
- `openai`, `tenacity`, `python-dotenv` - OpenRouter client plumbing and resilience.

## Input Data

- **Human references**: 150 manually curated summaries stored in `input/DPS_Human_Summaries.csv` with project, file, pattern label, GitHub URL, and prose description.
- **Source repositories**: AbdurRKhalid, JamesZBL, and selected Spring Framework modules copied into `input/` for offline processing.
- **SWUM dictionaries**: bundled within `src/main/resources/` to keep linguistic cues stable across runs.

## Outputs and Reporting

- `output/json-output/` - intermediate AST or metadata captures for reproducibility.
- `output/summary-output/*.csv` - per-method summaries plus experiment tags.
- `evaluation-results/overall_comparison.csv` - headline metrics table.
- `evaluation-results/b1_b5_ranking_detail.csv` - deep dive into LLM adjudication.
- `evaluation-results/conciseness_by_pattern.txt` - textual reports for qualitative review.
- Visual assets (`*.png`) for presentation decks or papers.

## Metrics

- **Cosine similarity** - TF-IDF vectors across summaries (0 <= s <= 1).
- **BERTScore** - contextual embeddings measuring semantic recall, precision, and F1.
- **Combined score** - arithmetic mean of cosine similarity and BERT F1 to balance lexical versus semantic fidelity.
- **LLM ranking** - qualitative ordering from Llama3-70B with reasoning strings stored for audit.

## Design Patterns Covered

- **Creational**: Factory Method, Abstract Factory, Singleton.
- **Structural**: Adapter, Decorator, Facade.
- **Behavioural**: Observer, Visitor, Memento.

## Development Workflow

- `mvn clean compile` - Java compile and SWUM resource generation.
- `mvn test` - unit tests ensuring pattern detection remains stable.
- `python -m pytest` (if enabled) - validation for new evaluation utilities.
- `python/summary_length_stats_llm_prompts.py` - monitors verbosity drift between experiments.

## Research Context

The project underpins studies on automated code documentation, assessing whether hybrid approaches (symbolic plus neural) can rival expert-authored summaries. The artefacts provided enable replication, alternative LLM plug-ins, and human-in-the-loop verification.

## Limitations

1. Java-only coverage; Kotlin or C# sources are presently out of scope.
2. Nine design patterns supported; composite or state patterns require additional heuristics.
3. LLM pipelines depend on third-party availability and cost control.
4. Context windows are capped (default 256 tokens) to ensure deterministic latency.
5. Human references originate from a limited project set, which may bias lexical style.

## Roadmap

- Extend AST traversals to support additional behavioural patterns (Command, Strategy).
- Experiment with fine-tuned, self-hosted LLMs to reduce API dependence.
- Broaden evaluation metrics (MoverScore, BLEURT) for cross-study comparability.
- Build a lightweight web front-end for interactive exploration of summaries and rankings.

## Contributing

Contributions are welcome. Please open an issue describing planned work (for example new pattern detectors, evaluation scripts, or visualisations) before submitting a pull request to avoid duplicated effort.

## Licence

Released under the [BSD 2-Clause Licence](LICENSE). See the licence file for full terms.

## Citation


## Contact

For research collaborations or dataset questions, please raise a GitHub issue.

**Last updated**: 16 January 2026 - Status: Active research prototype.
