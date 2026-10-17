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
│   ├── dps_nlg/             # NLG generator (Maven exec: nlg)
│   ├── dps_swum/            # SWUM-based summariser (exec: swum)
│   └── dps_llm/             # OpenRouter client and prompt orchestration
├── python/                  # Evaluation and ranking scripts
├── resources/               # Shared prompts (prompts.json) for Java and Python
├── archive/                 # Superseded scripts and duplicate Spring module folders, kept for provenance
├── input/
│   ├── corpus/              # Java datasets: <Project>/<Pattern>/*.java
│   └── ground-truth/        # Human references (DPS_Human_Summaries.csv)
├── output/                  # Generated summaries and JSON intermediates
├── evaluation-results/      # Metrics and ranking reports
├── final-results/           # Per-RQ tables (CSV + LaTeX), regenerated from the raw artefacts
├── pom.xml
├── requirements.txt
└── README.md
```

## Prerequisites

- **Java**: JDK 11+ (tested up to JDK 21).
- **Maven**: 3.6+.
- **Python**: 3.10+.
- **OpenRouter API key**: needed only for LLM summary generation and the LLM-judge ranking
  scripts. The NLG and SWUM pipelines, the similarity metrics, the statistical tests and the
  final tables all run offline.

## Where to run the commands from

**Every command in this README is run from the repository root**, the
`designpatternsummarisation-nlg-swum-llm/` folder that contains `pom.xml`, `.env` and the
`python/` folder. Do not `cd` into `python/` or `src/`:

- the Java pipelines read `input/corpus/` and write `output/` and `evaluation-results/` using
  paths relative to the current folder;
- several Python scripts (`evaluate_summaries.py`, `spearman_sensitivity_analysis.py`) default to paths such as `output/summary-output/...`
  relative to the current folder.

To check you are in the right place:

```bash
ls pom.xml python/ input/corpus/    # all three should be listed
```

## Setup (one time)

### 1) Clone and build Java

```bash
git clone https://github.com/najamnazar/designpatternsummarisation-nlg-swum-llm.git
cd designpatternsummarisation-nlg-swum-llm
mvn clean install        # compiles and runs the JUnit tests
```

### 2) Create and activate a Python virtual environment

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

Activate the environment again in every new terminal before running the Python scripts.

### 3) Create `.env` in the repository root

Both the Java pipelines and the Python scripts read `.env` from the repository root. It is
listed in `.gitignore`, so your key is not committed.

Use this template. `OPENROUTER_API_URL` is required by both the Java LLM pipeline and the
Python ranking scripts; values may optionally be wrapped in quotes.

```text
# Java LLM summary generation (mvn exec:java@llm-gpt / mvn exec:java@llm-mistral)
OPENROUTER_API_KEY=sk-your-key
OPENROUTER_API_URL=https://openrouter.ai/api/v1/chat/completions
OPENROUTER_MAX_COMPLETION_TOKENS=512
OPENROUTER_TEMPERATURE=0
# OpenRouter model ids; the --gpt / --mistral flag picks which one a run uses
GPT_MODEL=openai/your-gpt-model-id
MISTRAL_MODEL=mistralai/your-mistral-model-id
LLM_PROJECT_LIMIT=
# Optional: run other prompts from resources/prompts.json, e.g. SENIOR_ANALYST_20_WORDS=output/summary-output/llm_summaries_20.csv
LLM_PROMPT_ALIASES=

# Python ranking script (rank_summaries_four_way.py); model and max tokens are required, no defaults in code
RANK_SUMMARIES_MODEL=your-judge-model-id
RANK_SUMMARIES_MAX_TOKENS=50
# Optional: separate endpoint for ranking; defaults to OPENROUTER_API_URL
RANK_SUMMARIES_API_URL=
```

## Running the pipeline

Run the steps in this order; each one reads what the previous one wrote. All generated
summaries and evaluation results are already committed, so you can start at any step: for
example, step 4 alone rebuilds the paper tables from the committed results.

The Maven commands are the same on Windows, Linux and macOS. On Windows you can use `py`
instead of `python`.

### Step 1: Generate summaries (Java)

```bash
# 1a. NLG summaries          -> output/summary-output/nlg_summaries.csv, output/json-output/nlg/
mvn exec:java@nlg

# 1b. SWUM summaries         -> output/summary-output/swum_summaries.csv, output/json-output/swum/
#     Reads output/json-output/nlg/, so run 1a first.
mvn exec:java@swum

# 1c. LLM summaries (calls OpenRouter; needs .env). Pick the model with a flag:
mvn exec:java@llm-gpt        # GPT_MODEL     -> output/summary-output/llm_summaries_gpt.csv
mvn exec:java@llm-mistral    # MISTRAL_MODEL -> output/summary-output/llm_summaries_mistral.csv

# 1d. Optional: the same two models with the non-concise 50-word prompt
mvn exec:java@llm-gpt-nonconcise       # -> output/summary-output/llm_summaries_nonconcise_gpt.csv
mvn exec:java@llm-mistral-nonconcise   # -> output/summary-output/llm_summaries_nonconcise_mistral.csv

# 1e. If an LLM run stopped part-way (e.g. 429 rate limits), resume it: the classes already
#     in that CSV are skipped without calling the API, and only the missing ones are generated
mvn exec:java@llm-gpt-resume
mvn exec:java@llm-mistral-resume
mvn exec:java@llm-gpt-nonconcise-resume
mvn exec:java@llm-mistral-nonconcise-resume

# 1f. Optional: summary-length variants (20, 40, 60 or 80 words), one prompt per command
mvn exec:java@llm-gpt-20       # SENIOR_ANALYST_20_WORDS -> output/summary-output/llm_summaries_gpt_20.csv
mvn exec:java@llm-gpt-40       # SENIOR_ANALYST_40_WORDS -> output/summary-output/llm_summaries_gpt_40.csv
mvn exec:java@llm-gpt-60       # SENIOR_ANALYST_60_WORDS -> output/summary-output/llm_summaries_gpt_60.csv
mvn exec:java@llm-gpt-80       # SENIOR_ANALYST_80_WORDS -> output/summary-output/llm_summaries_gpt_80.csv
mvn exec:java@llm-mistral-20   # the same four for Mistral   -> output/summary-output/llm_summaries_mistral_20.csv, ...
mvn exec:java@llm-mistral-40
mvn exec:java@llm-mistral-60
mvn exec:java@llm-mistral-80
#     Each has a resume version too, e.g. mvn exec:java@llm-gpt-20-resume

# Optional: print the prompts available in resources/prompts.json
mvn exec:java@prompt-demo
```

The `llm-*` executions are shortcuts for passing the flags to `DpsLlmApplication` yourself;
each pair below is equivalent:

```bash
mvn exec:java@llm-gpt
mvn exec:java@llm-summaries -Dexec.args="--gpt"

mvn exec:java@llm-mistral-nonconcise
mvn exec:java@llm-summaries -Dexec.args="--mistral --nonconcise"

mvn exec:java@llm-mistral-resume
mvn exec:java@llm-summaries -Dexec.args="--mistral --resume"

mvn exec:java@llm-gpt-20
mvn exec:java@llm-summaries -Dexec.args="--gpt" -Dllm.prompt.aliases=SENIOR_ANALYST_20_WORDS
```

Exactly one of `--gpt` or `--mistral` is required (`-gpt` and `-mistral` also work); without
one the program prints its usage and exits. The flag selects the model id from `GPT_MODEL` or
`MISTRAL_MODEL` in `.env`. The optional `--nonconcise` flag switches the prompt from
`SENIOR_ANALYST_50_WORDS` to `SENIOR_ANALYST_50_WORDS_NON_CONCISE` (both in
`resources/prompts.json`). Each model and prompt combination writes its own CSV, so no run
overwrites another, nor the canonical Mixtral summaries in `llm_summaries.csv` and
`llm_summaries_nonconcise.csv`. Set `LLM_SUMMARY_PATH` in `.env` to write
somewhere else.

The length variants (`llm-<model>-20`, `-40`, `-60`, `-80`) set the JVM property
`llm.prompt.aliases` to the matching `SENIOR_ANALYST_<n>_WORDS` prompt in
`resources/prompts.json`; nothing in `.env` needs to change. Leave `LLM_PROMPT_ALIASES` in
`.env` empty for these runs, because a value there takes precedence over the property.

A normal run empties its CSV first. With `--resume` it keeps the rows already there, skips
those classes (no API call, so no cost), and appends the missing ones. A class whose request
fails still gets no row, so repeat the resume command until the run reports
`Failed summaries: 0` (and `Already in CSV` + `Summaries generated` = 150).

The ranking script reads `nlg_summaries.csv`, `llm_summaries_gpt.csv`, `llm_summaries_mistral.csv`
and `swum_summaries.csv` directly; the `A.csv` / `B.csv` / `C.csv` copies and the five `B1.csv` to
`B5.csv` runs are no longer used. The scripts that worked on them (`rank_b1_b5_iterations.py`,
`evaluate_b1_b5.py`, `evaluate_iterations.py`, `rank_design_patterns.py` and the three-way
`rank_summaries.py`) are kept in `archive/` for reference.

### Step 2: Automated similarity metrics (offline)

```bash
# Cosine similarity + BERTScore for NLG, SWUM, GPT and Mistral (concise and non-concise) vs the
# human references; a missing CSV is skipped. Mixtral is only scored when passed with --dps-llm-csv.
python python/evaluate_summaries.py
```

The first run of a BERTScore script downloads the scoring model from Hugging Face, so it needs
internet access once.

`evaluate_summaries.py` compares NLG, SWUM, GPT and Mistral with the human summaries and writes
`nlg_vs_human_*.csv`, `swum_vs_human_*.csv`, `llm (gpt)_vs_human_*.csv`, `llm (mistral)_vs_human_*.csv`,
`llm (gpt non-concise)_vs_human_*.csv` and `llm (mistral non-concise)_vs_human_*.csv`, plus
`overall_comparison.csv`, `evaluation_summary.txt`, `results.txt` and the violin plot (NLG, SWUM,
GPT, Mistral). Mixtral is not scored by default; add
`--dps-llm-csv output/summary-output/llm_summaries.csv` to include it. Its score files keep their
names (`build_final_results.py` reads them). Override the inputs with `--gpt-csv`, `--mistral-csv`,
`--gpt-nonconcise-csv` and `--mistral-nonconcise-csv`, or leave them out with `--skip-gpt-mistral`.
The script warns when a CSV matches fewer than all 150 human summaries (resume that LLM run
first), and when a skipped method's old score files are still in the output folder.

To score another LLM corpus (for example a length-controlled one) without overwriting the
canonical Mixtral scores, write it to its own folder:

```bash
python python/evaluate_summaries.py \
  --dps-llm-csv output/summary-output/llm_summaries_20.csv \
  --skip-gpt-mistral \
  --output-dir evaluation-results/length-20
```

### Step 3: LLM-judge ranking (needs API access)

```bash
# Rank NLG vs GPT vs Mistral vs SWUM on 5 criteria, one iteration -> evaluation-results/four_way_ranking_*.csv
python python/rank_summaries_four_way.py

# Cheap trial on the first 3 classes (15 judge calls)
python python/rank_summaries_four_way.py --limit 3
```

For every class the judge sees the human summary and the four generated summaries, in the fixed
order 1 = NLG, 2 = GPT, 3 = Mistral, 4 = SWUM, and ranks them on accuracy, conciseness, adequacy,
code context and design pattern recognition (prompts: `summary_ranking` in
`resources/prompts.json`, with `{summary_a}`..`{summary_d}` for summaries 1..4). Because all four are ranked together, GPT and Mistral are compared
with each other as well as with NLG and SWUM. Points per criterion: 1st = 4, 2nd = 3, 3rd = 2,
4th = 1. A reply that is not a full ranking of the four is recorded as invalid and earns no points.

The judge model and the max tokens per call come only from `.env` (`RANK_SUMMARIES_MODEL`,
`RANK_SUMMARIES_MAX_TOKENS`); the script stops if either is missing. Outputs:
`four_way_ranking_detail.csv` (per class, saved after every class), `four_way_ranking_summary.csv`
(1st..4th counts, mean rank and mean points per criterion), `four_way_ranking_head_to_head.csv`
(how often each method of a pair was ranked higher) and a block in `results.txt` with
plain-text and LaTeX tables, replaced on every re-run.

### Step 4: Statistical tests and paper tables (offline)

```bash
# Paired Wilcoxon tests + rank-biserial effect sizes: NLG and SWUM vs each of GPT, GPT non-concise,
# Mistral and Mistral non-concise (run after evaluate_summaries.py, which rewrites results.txt)
python python/calculate_wilcoxon_tests.py

# Friedman tests (NLG, GPT, Mistral, SWUM) across the 5 ranking criteria, with post-hoc comparisons
# (run after rank_summaries_four_way.py; reads four_way_ranking_detail.csv)
python python/friedman_test.py

# Spearman criterion-to-total sensitivity analysis
python python/spearman_sensitivity_analysis.py --append-to evaluation-results/results.txt

# Paper-ready tables (CSV + LaTeX), grouped by research question -> final-results/
python python/build_final_results.py
```

### Full run, copy and paste

From the repository root, with the virtual environment active and `.env` filled in:

```bash
mvn clean install
mvn exec:java@nlg
mvn exec:java@swum
mvn exec:java@llm-gpt
mvn exec:java@llm-mistral
mvn exec:java@llm-gpt-nonconcise
mvn exec:java@llm-mistral-nonconcise
python python/evaluate_summaries.py
python python/rank_summaries_four_way.py
python python/calculate_wilcoxon_tests.py
python python/friedman_test.py
python python/spearman_sensitivity_analysis.py --append-to evaluation-results/results.txt
python python/build_final_results.py
```

### Script reference

| Script / command | API key | Reads | Writes (under `evaluation-results/` unless noted) |
| --- | --- | --- | --- |
| `mvn exec:java@nlg` | no | `input/corpus/` | `output/summary-output/nlg_summaries.csv` |
| `mvn exec:java@swum` | no | `output/json-output/nlg/` | `output/summary-output/swum_summaries.csv`, `pipeline_summary.txt` |
| `mvn exec:java@llm-gpt` / `@llm-mistral` | yes | `input/corpus/`, `resources/prompts.json` | `output/summary-output/llm_summaries_{gpt,mistral}.csv` |
| `mvn exec:java@llm-gpt-nonconcise` / `@llm-mistral-nonconcise` | yes | `input/corpus/`, `resources/prompts.json` | `output/summary-output/llm_summaries_nonconcise_{gpt,mistral}.csv` |
| `mvn exec:java@llm-<model>[-nonconcise]-resume` | yes (missing classes only) | the same, plus the existing CSV | appends to the same CSV |
| `mvn exec:java@llm-<model>-{20,40,60,80}[-resume]` | yes | `input/corpus/`, `resources/prompts.json` | `output/summary-output/llm_summaries_<model>_<n>.csv` |
| `evaluate_summaries.py` | no | NLG, SWUM, GPT, Mistral (+ non-concise) CSVs + human CSV; Mixtral via `--dps-llm-csv` | `*_vs_human_{class,project}_scores.csv`, `overall_comparison.csv`, `evaluation_summary.txt`, violin plot |
| `rank_summaries_four_way.py` | yes | NLG, GPT, Mistral, SWUM CSVs + human CSV, `resources/prompts.json` | `four_way_ranking_{detail,summary,head_to_head}.csv`, block in `results.txt` |
| `calculate_wilcoxon_tests.py` | no | `*_vs_human_class_scores.csv` | block in `results.txt` |
| `friedman_test.py` | no | `four_way_ranking_detail.csv` | block in `results.txt` |
| `spearman_sensitivity_analysis.py` | no | `multi_criteria_rankings.csv` | console; `results.txt` with `--append-to` |
| `build_final_results.py` | no | all of the above | `final-results/` |

`python/project_paths.py` is a shared module, not a script.
`spearman_sensitivity_analysis.py` and `build_final_results.py` still expect the old three-way
outputs (`multi_criteria_rankings.csv`, `A.csv`..`C.csv`, `b1_b5_*`) and need updating before they
can use the four-way ranking.

## Run From IDE (VS Code, IntelliJ, Eclipse)

Set the run configuration's **working directory to the repository root** (the default in VS
Code when the repository folder is opened as the workspace; `$PROJECT_DIR$` in IntelliJ).
Running with `python/` or `src/` as the working directory makes the relative paths above fail.

### Java main classes

- `src/main/java/dps_nlg/Application.java` (NLG)
- `src/main/java/dps_swum/swum/SWUMEvaluationPipeline.java` (SWUM)
- `src/main/java/dps_llm/DpsLlmApplication.java` (LLM): set the program argument to `--gpt` or `--mistral`

### Python scripts

Any script in the table above can be opened and run directly (in VS Code: Run Python File),
with the `.venv` interpreter selected.

## Configuration

### Environment variables (.env)

See the template in [Setup step 3](#3-create-env-in-the-repository-root). The Java code reads
the same `.env` file as Python (`OPENROUTER_API_URL`, not `OPENROUTER_BASE_URL`), and retries
rate-limit (429) and server (5xx) errors up to four times with exponential backoff.

`OPENROUTER_MODEL` is no longer read: the model comes from `GPT_MODEL` or `MISTRAL_MODEL`,
chosen by the `--gpt` / `--mistral` flag. `OPENROUTER_MAX_TOKENS` is still accepted when
`OPENROUTER_MAX_COMPLETION_TOKENS` is not set (default 256). The ranking scripts use their own
judge model, `RANK_SUMMARIES_MODEL`.

### Python dependencies

Key packages from `requirements.txt`:

- `pandas`, `numpy`, `scikit-learn` - tabular processing and cosine similarity.
- `bert-score`, `transformers`, `torch` - semantic similarity computation.
- `matplotlib`, `seaborn` - visualisation helpers.
- `requests`, `python-dotenv`, `scipy` - API calls, environment loading, statistical tests.

## Input Data

- **Human references**: 150 manually curated summaries in `input/ground-truth/DPS_Human_Summaries.csv`.
- **Source repositories**: curated Java projects under `input/corpus/<Project>/<Pattern>/`
  (`AbdurRKhalid`, `JamesZBL`, `SpringFramework`), one folder per design pattern.
- These paths are defined once: `common.utils.ProjectPaths` (Java) and `python/project_paths.py`
  (Python). The NLG and LLM pipelines scan `input/corpus/`, and project names are taken relative
  to it, so a new project goes in as `input/corpus/<Project>/<Pattern>/`.

### Pattern naming convention

Every project uses the same nine pattern folders, written in PascalCase with no spaces or
hyphens: `AbstractFactory`, `Adapter`, `Decorator`, `Facade`, `FactoryMethod`, `Memento`,
`Observer`, `Singleton`, `Visitor`. The same form is used for the `Design Pattern` column of
`input/ground-truth/DPS_Human_Summaries.csv`, the `Folder Name` column of every summary CSV, JSON file names
and JSON pattern keys (see `common.designpatternidentifier.PatternNames`). Inside summary
sentences the readable form is used instead ("Abstract Factory pattern"), matching the human
references. The `URL` column of the human CSV, and the `project_path` column derived from it,
keep the original repositories' own folder names (e.g. `JamesZBL/abstract-factory`).

Project names follow the same rule: `AbdurRKhalid`, `JamesZBL` and `SpringFramework` (formerly
`spring-framework` in `input/` and the outputs, and `Spring-Framework` in the human CSV). This
applies to the input folder, the `Project` column of the human and summary CSVs, JSON file names
and the project columns of the evaluation tables. Python matchers ignore case and punctuation,
so older labels still match.

`archive/spring-framework-modules/` holds the ten Spring module folders (`spring-aop`, ...) that
used to sit in the Spring project folder (now `input/corpus/SpringFramework/`); all 33 files in them were byte-identical copies of files
in the Spring pattern folders, so they were never summarised separately.
- **Prompts**: `resources/prompts.json`, shared by the Java LLM pipeline (on the
  classpath via `pom.xml`) and the Python ranking scripts.

## Outputs

- `output/json-output/` - intermediate AST or metadata captures.
- `output/summary-output/*.csv` - generated summaries.
- `evaluation-results/*.csv` and `evaluation-results/*.txt` - evaluation outputs.
- `final-results/` - the tables reported in the paper, grouped by research
  question (`corpus/`, `rq1/` ... `rq4/`), each as a `.csv` and a paste-ready
  `.tex`. Regenerate with `python python/build_final_results.py` (from the repository root).
  `final-results/README.md` indexes the tables and lists what still needs a
  generation run; `final-results/ERRATA.md` records what changed against the
  submitted manuscript and why.

### Which LLM run is canonical

The probabilistic arm is the Mixtral run in
`output/summary-output/llm_summaries.csv` (identical to `B.csv`), scored as
`evaluation-results/llm (mixtral)_vs_human_class_scores.csv`. Every LLM number
in the paper comes from that run. `calculate_wilcoxon_tests.py` now tests the GPT and
Mistral runs (concise and non-concise) by default; `--llm-csv` runs the old single-arm test
on one file instead, e.g. `--llm-csv "llm (mixtral)_vs_human_class_scores.csv"`. It refuses
to run if the arms do not cover the same files, rather than silently testing a smaller set.

### Matching files across corpora

Twenty files share a (project, filename) pair with another file - `Test.java`
appears seven times under AbdurRKhalid and `Application.java` nine times under
JamesZBL - so a project-plus-filename key collapses different classes onto one
entry. Every matcher here keys on project + pattern folder + filename and
asserts that the resulting keys are unique.

## Design Patterns Covered

- **Creational**: Factory Method, Abstract Factory, Singleton.
- **Structural**: Adapter, Decorator, Facade.
- **Behavioural**: Observer, Visitor, Memento.

## Development Workflow

- `mvn clean compile` - compile Java sources.
- `mvn test` - run Java unit tests (JUnit 5, under `src/test/java`).

## Licence

Released under the [BSD 2-Clause Licence](LICENSE).

## Contact

For research collaborations or dataset questions, please open a GitHub issue.

**Last updated**: 14 October 2026 - Status: Active research prototype.
