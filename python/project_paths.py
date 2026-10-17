"""Shared repository paths for the Python scripts.

CHANGE (input layout): the input folder is now split into

    input/
    ├── corpus/          Java sources: corpus/<Project>/<Pattern>/*.java
    └── ground-truth/    DPS_Human_Summaries.csv

Scripts used to build ``input/DPS_Human_Summaries.csv`` themselves; they now import
these constants so the layout is defined once. The Java pipelines use the same
paths from ``common.utils.ProjectPaths``.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = REPO_ROOT / "input"
CORPUS_DIR = INPUT_DIR / "corpus"
GROUND_TRUTH_DIR = INPUT_DIR / "ground-truth"
HUMAN_SUMMARIES_CSV = GROUND_TRUTH_DIR / "DPS_Human_Summaries.csv"
