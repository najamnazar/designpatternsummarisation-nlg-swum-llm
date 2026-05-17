# Design Pattern Summariser

An automated design pattern summarisation platform that generates, ranks, and evaluates natural-language summaries for Java design-pattern implementations. Three summarisation strategies - template-driven NLG, identifier-aware SWUM, and OpenRouter-hosted LLMs - are benchmarked against 150 human-written references using automated metrics and LLM judgement.

## Overview

The pipeline analyses curated Java projects that exemplify the Gang-of-Four patterns and produces class-level summaries via:

1. **NLG (Natural Language Generation)** - deterministic templates informed by AST and design-pattern cues.
2. **SWUM (Software Word Use Model)** - linguistic modelling of identifier semantics plus lightweight static analysis.
3. **LLM (Large Language Model)** - Mixtral-8x22B (and swap-in OpenRouter models) prompted with code context for richer prose.

Outputs are cross-checked against expert summaries through cosine similarity, BERTScore, and LLM-facing ranking pipelines.

## Repository Layout

```text
designpatternsummarisation-nlg-swum-llm/
├── src/main/java/
│   ├── common/              # Shared AST tooling and detection heuristics
│   ├── dps_nlg/             # NLG generator (Maven exec: dps-app)
│   ├── dps_swum/            # SWUM-based summariser (exec: swum-pipeline)
│   └── dps_llm/             # OpenRouter client and prompt orchestration
├── python/                  # Evaluation and ranking scripts
├── input/                   # Human references and Java datasets
├── output/                  # Generated summaries and JSON intermediates
├── evaluation-results/      # Metrics and ranking reports
├── pom.xml
├── requirements.txt
└── README.md
```

## Prerequisites

- **Java**: JDK 11+ (tested up to JDK 21).
- **Maven**: 3.6+.
- **Python**: 3.10+.
- **OpenRouter API key**: required for LLM generation and ranking workflows.

## Quick Start (Console)

### 1) Clone and build Java

```bash
Clone the repository.
cd designpatternsummarisation-nlg-swum-llm
mvn clean install
```

### 2) Create and activate Python virtual environment

Linux and macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows Command Prompt:

```bat
py -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3) Create `.env` in project root

Use this exact template:

```text
OPENROUTER_API_KEY=sk-your-key
OPENROUTER_MODEL=mistralai/mixtral-8x22b-instruct
OPENROUTER_TEMPERATURE=0.2
OPENROUTER_MAX_TOKENS=256
LLM_PROJECT_LIMIT=
OPENROUTER_BASE_URL=
```

## Run Pipelines From Console

The Maven commands below are the same on Windows, Linux, and macOS.

### Java summary generation

```bash
# NLG summaries
mvn exec:java@dps-app

# SWUM summaries
mvn exec:java@swum-pipeline

# LLM summaries via OpenRouter
mvn exec:java@llm-summaries
```

### Python evaluation and ranking

```bash
# Core automated metrics
python python/evaluate_summaries.py

# Multi-criteria ranking
python python/rank_summaries.py --limit 150
python python/rank_summaries.py --limit 50

# Iteration analysis
python python/rank_b1_b5_iterations.py
python python/evaluate_iterations.py
```

On Windows you can use `py` instead of `python` if preferred.

## Run From IDE (VS Code, IntelliJ, Eclipse)

### Java main files

Run any of these main classes from your IDE run configuration:

- `src/main/java/dps_nlg/Application.java`
- `src/main/java/dps_swum/swum/SWUMEvaluationPipeline.java`
- `src/main/java/dps_llm/DpsLlmApplication.java`

### Python scripts

Open and run these files directly from your IDE:

- `python/evaluate_summaries.py`
- `python/rank_summaries.py`
- `python/evaluate_b1_b5.py`
- `python/rank_design_patterns.py`
- `python/evaluate_nlg_additional_metrics.py`

In VS Code, use Run Python File. In IntelliJ or Eclipse, run the Java class containing `main(...)`.

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
- `requests`, `python-dotenv`, `scipy` - API calls, environment loading, statistical tests.
- `nltk`, `rouge-score`, `frugalscore` - additional text-quality metrics.

## Input Data

- **Human references**: 150 manually curated summaries in `input/DPS_Human_Summaries.csv`.
- **Source repositories**: curated Java projects under `input/`.
- **SWUM dictionaries**: bundled in `src/main/resources/`.

## Outputs

- `output/json-output/` - intermediate AST or metadata captures.
- `output/summary-output/*.csv` - generated summaries.
- `evaluation-results/*.csv` and `evaluation-results/*.txt` - evaluation outputs.

## Design Patterns Covered

- **Creational**: Factory Method, Abstract Factory, Singleton.
- **Structural**: Adapter, Decorator, Facade.
- **Behavioural**: Observer, Visitor, Memento.

## Development Workflow

- `mvn clean compile` - compile Java sources.
- `mvn test` - run Java tests.
- `python -m pytest` (if enabled) - run Python tests.

## Licence

Released under the [BSD 2-Clause Licence](LICENSE).

## Contact

For research collaborations or dataset questions, please open a GitHub issue.

**Last updated**: 17 May 2026 - Status: Active research prototype.
