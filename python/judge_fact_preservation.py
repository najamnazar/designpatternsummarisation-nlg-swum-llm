#!/usr/bin/env python3
"""
judge_fact_preservation.py - Design-pattern fact preservation of DPS summaries (LLM judge)

For every evaluated class, a list of atomic facts about its design-pattern role and
relationships is extracted from the DPS JSON (output/json-output/nlg/<Project>_<Pattern>.json).
An LLM judge then labels each fact PRESERVED, OMITTED or INCORRECT for each of the four
summaries of that class (NLG, GPT, Mistral, SWUM). The class list and the summaries are the
same as in rank_summaries_four_way.py.

Facts (each states exactly one thing):
    detector  role       the pattern role the DPS detector assigned to the class, and the
                         classes it lists directly under it ("X is a concrete factory for Y")
    parser    kind       interface / abstract class
    parser    inheritance  "X extends Y", "X implements Y"
    parser    constructor  private constructor; constructor parameters whose type is another
                         class of the same pattern instance
    parser    field      fields whose type is another class of the same pattern instance
                         (or the class itself, as in Singleton)
    parser    method     declared or overridden methods (overridden first)
    parser    call       method calls the parser resolved to another class
Counts, line numbers and body line types are not used: summaries do not state them, so they
would only add OMITTED labels to every method. Facts of a type beyond its cap are left out.

Every fact is also reported under the reviewer's information categories:
    pattern role     role
    responsibility   method, constructor, field
    relationship     inheritance, call
    identity         kind
and every role fact is marked "same pattern" or "other pattern": whether the pattern the detector
names is the pattern folder the class comes from (the detector also reports patterns it finds
in other folders, e.g. Observer roles among the Abstract Factory classes).

Corpora (--corpora):
    default      the four default summaries (NLG, GPT, Mistral, SWUM): the four-way comparison
    word-limits  those four plus the GPT and Mistral summaries written to 20/40/60/80-word limits
                 (output/summary-output/llm_summaries_{gpt,mistral}_{20,40,60,80}.csv). The facts
                 and the prompt are the same, so the default replies are reused from the cache and
                 only the eight word-limit corpora are new calls. Results go to separate files.

Evidence quotes: the judge quotes the summary for each label. evidence_found records whether the
quote is in the summary verbatim. evidence_match refines it: "verbatim"; "fragments" when the
judge joined parts of the summary with "..." and every part is verbatim; "not_found" otherwise.
"fragments" is lenient: each part is matched on its own, and a part can be a single word, so it
says the words are in the summary, not that they support the label; that needs a manual check.
A quote that is not verbatim is not necessarily wrong (the judge may combine or reword passages),
so the preserved rate counting only verbatim quotes, and only verbatim or fragment quotes, is
reported as a sensitivity check, not as an error count.

Configuration comes from .env; the code has no fallback values for it:
    OPENROUTER_API_KEY                   API key (not needed for --dry-run)
    OPENROUTER_API_URL                   endpoint, unless FACT_JUDGE_API_URL is set
    FACT_JUDGE_API_URL                   optional endpoint for this task only
    FACT_JUDGE_MODEL                     judge model id
    FACT_JUDGE_TEMPERATURE               sampling temperature of the judge
    FACT_JUDGE_TOKENS_PER_FACT           completion tokens allowed per fact
    FACT_JUDGE_MAX_TOKENS                ceiling on completion tokens per call
    FACT_JUDGE_MAX_ATTEMPTS              attempts per call when the API request fails
    FACT_JUDGE_REQUEST_TIMEOUT_SECONDS   timeout of one API request
    FACT_JUDGE_RETRY_BACKOFF_SECONDS     wait before a failed request is sent again (doubles each time)
    FACT_JUDGE_MAX_ROLE_FACTS            caps on the number of facts per class, by type
    FACT_JUDGE_MAX_FIELD_FACTS
    FACT_JUDGE_MAX_CONSTRUCTOR_FACTS
    FACT_JUDGE_MAX_METHOD_FACTS
    FACT_JUDGE_MAX_CALL_FACTS
Completion tokens per call = min(FACT_JUDGE_MAX_TOKENS, FACT_JUDGE_TOKENS_PER_FACT x facts).
Prompt: resources/prompts.json, section "fact_preservation", key "judge"
({class_name}, {facts}, {summary}).

Outputs (evaluation-results/):
    fact_preservation_facts.csv     the extracted facts, one row per fact
    fact_preservation_raw.jsonl     every judge reply, appended as it arrives (the resume cache, shared
                                    by both corpus sets)
    fact_preservation_verdicts.csv  one row per fact and method: label, evidence, justification,
                                    evidence match, summary length in words
    fact_preservation_detail.csv    one row per class and method: validity and label counts
    fact_preservation_summary.csv   per method and fact scope: label counts, rates, mean summary length
    fact_preservation_results.csv   every figure of the results.txt block, one row each
    results.txt                     delimited block (replaced on re-run), plain-text and LaTeX tables
With --corpora word-limits the four per-run tables are written as fact_preservation_word_limits_*.csv
instead, and results.txt gets a separate block; the default outputs are left untouched.

Usage:
    python python/judge_fact_preservation.py --dry-run         extract and check the facts only, no API calls
    python python/judge_fact_preservation.py --limit 3         judge the first 3 classes (trial run)
    python python/judge_fact_preservation.py                   judge every class; replies already in
                                                               fact_preservation_raw.jsonl are reused
    python python/judge_fact_preservation.py --reparse         no API calls: rebuild all outputs from the saved replies
    python python/judge_fact_preservation.py --retry-invalid   as a normal run, and ask once more for each
                                                               saved reply that is not valid
    python python/judge_fact_preservation.py --corpora word-limits [--dry-run | --reparse | ...]
                                                               the same, for the 20/40/60/80-word corpora
A saved reply is reused only for exactly the same request (prompt, model, temperature and max
tokens), so changing the prompt, the facts or a setting in .env asks the judge again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv

from project_paths import REPO_ROOT
# The class list, the summaries and the match keys are shared with the four-way ranking,
# so both evaluations judge exactly the same 150 classes and summaries.
from rank_summaries_four_way import METHODS, Method, build_match_key, load_human_summaries, load_method_summaries

# --------------------------------------------------------------------------- locations

ENV_FILE = REPO_ROOT / ".env"
PROMPTS_JSON = REPO_ROOT / "resources" / "prompts.json"
PROMPT_SECTION = "fact_preservation"
PROMPT_KEY = "judge"
PROMPT_PLACEHOLDERS = ("class_name", "facts", "summary")

# The detector block is identical in the llm/, nlg/ and swum/ folders; nlg/ also holds the parser data.
FACT_SOURCE_DIR = REPO_ROOT / "output" / "json-output" / "nlg"
EVAL_DIR = REPO_ROOT / "evaluation-results"
FACTS_CSV = EVAL_DIR / "fact_preservation_facts.csv"
RAW_JSONL = EVAL_DIR / "fact_preservation_raw.jsonl"
VERDICTS_CSV = EVAL_DIR / "fact_preservation_verdicts.csv"
DETAIL_CSV = EVAL_DIR / "fact_preservation_detail.csv"
SUMMARY_CSV = EVAL_DIR / "fact_preservation_summary.csv"
RESULTS_CSV = EVAL_DIR / "fact_preservation_results.csv"
RESULTS_TXT = EVAL_DIR / "results.txt"

# Markers delimiting this script's block inside results.txt, so a re-run replaces it.
BLOCK_START = "<<<BEGIN FACT PRESERVATION>>>"
BLOCK_END = "<<<END FACT PRESERVATION>>>"

# The GPT and Mistral summaries generated with an explicit target length in words.
WORD_LIMITS = (20, 40, 60, 80)
WORD_LIMIT_METHODS: Tuple[Method, ...] = tuple(
    Method(f"{family} {words}w", f"{slug}_{words}", f"llm_summaries_{slug}_{words}.csv")
    for slug, family in (("gpt", "GPT"), ("mistral", "Mistral"))
    for words in WORD_LIMITS
)


@dataclass(frozen=True)
class CorpusSet:
    """The summaries judged in one run and where that run's outputs go."""
    name: str
    title: str
    methods: Tuple[Method, ...]
    verdicts_csv: Path
    detail_csv: Path
    summary_csv: Path
    results_csv: Path
    block_start: str
    block_end: str


CORPUS_SETS: Dict[str, CorpusSet] = {
    "default": CorpusSet(
        "default", "DESIGN-PATTERN FACT PRESERVATION (LLM judge against facts extracted from the DPS JSON)",
        tuple(METHODS), VERDICTS_CSV, DETAIL_CSV, SUMMARY_CSV, RESULTS_CSV, BLOCK_START, BLOCK_END),
    # The default corpora are included so the word-limit corpora are compared with NLG and SWUM on the
    # same classes; their replies are already cached, so only the eight word-limit corpora cost calls.
    "word-limits": CorpusSet(
        "word-limits", "DESIGN-PATTERN FACT PRESERVATION UNDER 20/40/60/80-WORD LIMITS (LLM judge)",
        tuple(METHODS) + WORD_LIMIT_METHODS,
        EVAL_DIR / "fact_preservation_word_limits_verdicts.csv",
        EVAL_DIR / "fact_preservation_word_limits_detail.csv",
        EVAL_DIR / "fact_preservation_word_limits_summary.csv",
        EVAL_DIR / "fact_preservation_word_limits_results.csv",
        "<<<BEGIN FACT PRESERVATION WORD LIMITS>>>", "<<<END FACT PRESERVATION WORD LIMITS>>>"),
}

# --------------------------------------------------------------------------- fixed vocabulary

# The three labels the prompt defines; the reply is rejected if it uses anything else.
LABELS = ("PRESERVED", "OMITTED", "INCORRECT")

# Fact types in the order they are listed to the judge, with the source they come from.
FACT_TYPES: Tuple[Tuple[str, str], ...] = (
    ("role", "detector"),
    ("kind", "parser"),
    ("inheritance", "parser"),
    ("constructor", "parser"),
    ("field", "parser"),
    ("method", "parser"),
    ("call", "parser"),
)
SOURCES = ("detector", "parser")

# The reviewer's information categories (pattern names sit inside the role facts).
FACT_CATEGORIES = {
    "role": "pattern role",
    "method": "responsibility",
    "constructor": "responsibility",
    "field": "responsibility",
    "inheritance": "relationship",
    "call": "relationship",
    "kind": "identity",
}
CATEGORIES = ("pattern role", "responsibility", "relationship", "identity")

# Whether a role fact names the pattern of the folder its class comes from.
ROLE_PATTERNS = ("same pattern", "other pattern")

# Evidence-match values, from strictest to loosest; "" when the judge gave no quote.
EVIDENCE_MATCHES = ("verbatim", "fragments", "not_found")
ELLIPSIS = re.compile(r"\.{3,}|…")

# Top-level keys of a DPS JSON file that are not a pattern section of per-class parser data.
NON_CLASS_KEYS = {"final_summary", "design_pattern", "summary_NLG"}

# The detector names the role of every class except the one each pattern instance is keyed
# on ({"AbstractFactory": {"GUIFactory": {...}}}). This names that class's role in GoF terms.
ANCHOR_ROLES = {
    "AbstractFactory": "abstract factory",
    "FactoryMethod": "creator",
    "Adapter": "adapter",
    "Decorator": "decorator",
    "Observer": "observer interface",
    "Visitor": "element interface",
    "Facade": "facade",
    "Memento": "originator",
}
UNKNOWN_ANCHOR_ROLE = "central participant"


# =========================================================================== configuration

class EnvReader:
    """Reads and validates settings from .env. A missing or malformed value is an error, never a default."""

    def __init__(self, env_file: Path) -> None:
        if not env_file.exists():
            raise FileNotFoundError(f"{env_file} not found")
        # override=True: the values in .env win over variables already set in the shell.
        load_dotenv(env_file, override=True)

    @staticmethod
    def optional_text(name: str) -> str:
        return (os.getenv(name) or "").strip()

    def text(self, name: str) -> str:
        value = self.optional_text(name)
        if not value:
            raise ValueError(f"{name} is not set in .env")
        return value

    def integer(self, name: str, minimum: int) -> int:
        raw = self.text(name)
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
        if value < minimum:
            raise ValueError(f"{name} must be at least {minimum}, got {value}")
        return value

    def number(self, name: str, minimum: float, maximum: float) -> float:
        raw = self.text(name)
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError(f"{name} must be a number, got {raw!r}") from exc
        if not minimum <= value <= maximum:
            raise ValueError(f"{name} must be between {minimum} and {maximum}, got {value}")
        return value


@dataclass(frozen=True)
class FactCaps:
    """Maximum number of facts of each capped type kept per class."""
    role: int
    field: int
    constructor: int
    method: int
    call: int

    def describe(self) -> str:
        return (f"role {self.role}, field {self.field}, constructor {self.constructor}, "
                f"method {self.method}, call {self.call}")


@dataclass(frozen=True)
class FactJudgeSettings:
    """Every setting of this evaluation, read from .env."""
    api_key: str
    api_url: str
    model: str
    temperature: float
    tokens_per_fact: int
    max_tokens: int
    max_attempts: int
    request_timeout: int
    retry_backoff: float
    caps: FactCaps

    @classmethod
    def from_env(cls, env_file: Path, need_api: bool) -> "FactJudgeSettings":
        """Load the settings. need_api=False (dry run) skips the API key and endpoint."""
        env = EnvReader(env_file)
        caps = FactCaps(
            role=env.integer("FACT_JUDGE_MAX_ROLE_FACTS", 0),
            field=env.integer("FACT_JUDGE_MAX_FIELD_FACTS", 0),
            constructor=env.integer("FACT_JUDGE_MAX_CONSTRUCTOR_FACTS", 0),
            method=env.integer("FACT_JUDGE_MAX_METHOD_FACTS", 0),
            call=env.integer("FACT_JUDGE_MAX_CALL_FACTS", 0),
        )
        settings = cls(
            api_key=env.text("OPENROUTER_API_KEY") if need_api else "",
            api_url=(env.optional_text("FACT_JUDGE_API_URL") or env.text("OPENROUTER_API_URL")) if need_api else "",
            model=env.text("FACT_JUDGE_MODEL"),
            temperature=env.number("FACT_JUDGE_TEMPERATURE", 0.0, 2.0),
            tokens_per_fact=env.integer("FACT_JUDGE_TOKENS_PER_FACT", 1),
            max_tokens=env.integer("FACT_JUDGE_MAX_TOKENS", 1),
            max_attempts=env.integer("FACT_JUDGE_MAX_ATTEMPTS", 1),
            request_timeout=env.integer("FACT_JUDGE_REQUEST_TIMEOUT_SECONDS", 1),
            retry_backoff=env.number("FACT_JUDGE_RETRY_BACKOFF_SECONDS", 0.0, 3600.0),
            caps=caps,
        )
        if settings.max_tokens < settings.tokens_per_fact:
            raise ValueError("FACT_JUDGE_MAX_TOKENS must be at least FACT_JUDGE_TOKENS_PER_FACT")
        return settings

    def token_budget(self, n_facts: int) -> int:
        """Completion tokens for a call that judges n_facts facts."""
        return min(self.max_tokens, self.tokens_per_fact * n_facts)

    def fits(self, n_facts: int) -> bool:
        """False when the ceiling is below tokens_per_fact x n_facts, so the reply may be cut off."""
        return self.tokens_per_fact * n_facts <= self.max_tokens

    def describe(self) -> List[str]:
        return [
            f"Judge model: {self.model}",
            f"Temperature: {self.temperature}",
            f"Completion tokens: {self.tokens_per_fact} per fact, at most {self.max_tokens} per call",
            f"Fact caps per class: {self.caps.describe()}",
        ]


class PromptTemplate:
    """The fact-judge prompt from prompts.json, checked before any call is made."""

    def __init__(self, text: str) -> None:
        self.text = text

    @classmethod
    def load(cls, path: Path, section: str, key: str) -> "PromptTemplate":
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        block = data.get(section) if isinstance(data, dict) else None
        if not isinstance(block, dict) or not isinstance(block.get(key), str) or not block[key].strip():
            raise ValueError(f"prompts.json has no prompt at {section}.{key}")
        template = cls(block[key])
        template.validate()
        return template

    def validate(self) -> None:
        absent = [p for p in PROMPT_PLACEHOLDERS if "{" + p + "}" not in self.text]
        if absent:
            raise ValueError(f"prompts.json {PROMPT_SECTION}.{PROMPT_KEY} is missing placeholders: {absent}")
        # A trial fill catches single braces in the JSON example, which str.format cannot read.
        try:
            self.text.format(**{p: "" for p in PROMPT_PLACEHOLDERS})
        except (KeyError, IndexError, ValueError) as exc:
            raise ValueError(
                f"prompts.json {PROMPT_SECTION}.{PROMPT_KEY} cannot be filled ({exc!r}); "
                "literal braces must be doubled ({{ }})"
            ) from exc

    def render(self, class_name: str, facts: str, summary: str) -> str:
        return self.text.format(class_name=class_name, facts=facts, summary=summary)


# =========================================================================== facts

@dataclass(frozen=True)
class Fact:
    fact_id: int
    fact_type: str
    source: str
    text: str
    role_pattern: str = ""  # role facts only: "same pattern" or "other pattern"

    @property
    def category(self) -> str:
        return FACT_CATEGORIES[self.fact_type]


@dataclass
class ClassFacts:
    """The facts of one evaluated class."""
    project: str
    design_pattern: str
    file_name: str
    class_name: str
    match_key: str
    facts: List[Fact] = field(default_factory=list)
    # Facts left out by a cap, per type (reported, so the caps' effect is visible).
    dropped: Counter = field(default_factory=Counter)

    @property
    def label(self) -> str:
        return f"{self.project}/{self.design_pattern}/{self.file_name}"

    def numbered_list(self) -> str:
        """The facts as the judge sees them: '1. ...' one per line."""
        return "\n".join(f"{f.fact_id}. {f.text}" for f in self.facts)


@dataclass(frozen=True)
class RoleAssignment:
    """One role from the detector: subject plays role (for parent) in pattern."""
    pattern: str
    subject: str
    role: str
    parent: Optional[str]


def humanise_pattern(pattern: str) -> str:
    """'AbstractFactory' -> 'Abstract Factory'."""
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", pattern)


def humanise_role(role: str) -> str:
    """'concrete_factory' -> 'concrete factory'."""
    return role.replace("_", " ").strip()


def with_article(phrase: str) -> str:
    return ("an " if phrase[:1].lower() in "aeiou" else "a ") + phrase


class DetectorRoleReader:
    """Turns the detector's nested design_pattern block into flat role assignments.

    Layouts the detector writes:
        {"Singleton": ["A", "B"]}                              A, B are singletons
        {"P": {"Anchor": {"role": ["A", "B"]}}}               A, B play role for Anchor
        {"P": {"Anchor": {"role": "A"}}}                      A plays role for Anchor
        {"P": {"Anchor": {"role": {"A": ["B"]}}}}              A plays role for Anchor,
                                                                B is a concrete role for A
        {"P": {"Anchor": {"role": {"A": {"role2": [...]}}}}}   nested roles under A
    """

    def read(self, block: object) -> List[RoleAssignment]:
        assignments: List[RoleAssignment] = []
        for entry in block if isinstance(block, list) else [block]:
            if not isinstance(entry, dict):
                continue
            for pattern, instances in entry.items():
                if isinstance(instances, list):
                    assignments += [RoleAssignment(pattern, name, pattern.lower(), None)
                                    for name in instances if isinstance(name, str) and name]
                elif isinstance(instances, dict):
                    for anchor, roles in instances.items():
                        anchor_role = ANCHOR_ROLES.get(pattern, UNKNOWN_ANCHOR_ROLE)
                        assignments.append(RoleAssignment(pattern, anchor, anchor_role, None))
                        self._read_roles(pattern, anchor, roles, assignments)
        # The detector sometimes repeats a class ("CarpenterOperation" three times); keep one of each.
        return list(dict.fromkeys(assignments))

    def _read_roles(self, pattern: str, parent: str, roles: object, out: List[RoleAssignment]) -> None:
        if not isinstance(roles, dict):
            return
        for role_key, members in roles.items():
            role = humanise_role(role_key)
            if isinstance(members, str) and members:
                out.append(RoleAssignment(pattern, members, role, parent))
            elif isinstance(members, list):
                out += [RoleAssignment(pattern, m, role, parent) for m in members if isinstance(m, str) and m]
            elif isinstance(members, dict):
                for name, below in members.items():
                    out.append(RoleAssignment(pattern, name, role, parent))
                    if isinstance(below, list):
                        concrete = self._concrete(role)
                        out += [RoleAssignment(pattern, m, concrete, name) for m in below if isinstance(m, str) and m]
                    elif isinstance(below, dict):
                        self._read_roles(pattern, name, below, out)

    @staticmethod
    def _concrete(role: str) -> str:
        """The role of classes listed under an abstract participant ('abstract product' -> 'concrete product')."""
        if role.startswith("abstract "):
            return "concrete " + role[len("abstract "):]
        return role if role.startswith("concrete ") else "concrete " + role


class ClassFactBuilder:
    """Builds the atomic facts of one class from its detector roles and its parser data."""

    def __init__(self, caps: FactCaps) -> None:
        self.caps = caps

    def build(self, target: ClassFacts, parsed: Dict[str, object], peers: Iterable[str],
              roles: Sequence[RoleAssignment]) -> None:
        name = target.class_name
        peer_names = set(peers)
        role_facts = self._role_facts(name, roles)
        # The text names the pattern, so equal texts always have the same pattern.
        role_pattern_of = {text: self._role_pattern(pattern, target.design_pattern) for text, pattern in role_facts}
        candidates: Dict[str, List[str]] = {
            "role": [text for text, _ in role_facts],
            "kind": self._kind_facts(name, parsed),
            "inheritance": self._inheritance_facts(name, parsed),
            "constructor": self._constructor_facts(name, parsed, peer_names),
            "field": self._field_facts(name, parsed, peer_names),
            "method": self._method_facts(name, parsed),
            "call": self._call_facts(name, parsed),
        }
        caps = {"role": self.caps.role, "constructor": self.caps.constructor, "field": self.caps.field,
                "method": self.caps.method, "call": self.caps.call}

        next_id = 1
        for fact_type, source in FACT_TYPES:
            texts = list(dict.fromkeys(candidates[fact_type]))  # de-duplicated, order kept
            limit = caps.get(fact_type)
            if limit is not None and len(texts) > limit:
                target.dropped[fact_type] += len(texts) - limit
                texts = texts[:limit]
            for text in texts:
                target.facts.append(Fact(next_id, fact_type, source, text, role_pattern_of.get(text, "")))
                next_id += 1

    # ----- detector

    @staticmethod
    def _role_facts(name: str, roles: Sequence[RoleAssignment]) -> List[Tuple[str, str]]:
        """(text, detector pattern): the class's own roles first, then the classes the detector lists directly under it."""
        key = name.lower()
        own, children = [], []
        for r in roles:
            pattern = humanise_pattern(r.pattern)
            if r.subject.lower() == key:
                if r.parent is None:
                    own.append((f"{name} is the {r.role} in the {pattern} pattern.", r.pattern))
                else:
                    own.append((f"{name} is {with_article(r.role)} for {r.parent} in the {pattern} pattern.", r.pattern))
            elif r.parent is not None and r.parent.lower() == key:
                children.append((f"{r.subject} is {with_article(r.role)} for {name} in the {pattern} pattern.", r.pattern))
        return own + children

    @staticmethod
    def _role_pattern(detector_pattern: str, folder_pattern: str) -> str:
        """'same pattern' when the detector's pattern is the class's pattern folder, else 'other pattern'."""
        def norm(s: str) -> str:
            return re.sub(r"[^a-z]", "", s.lower())
        return ROLE_PATTERNS[0] if norm(detector_pattern) == norm(folder_pattern) else ROLE_PATTERNS[1]

    # ----- parser

    @staticmethod
    def _class_detail(parsed: Dict[str, object]) -> Dict[str, object]:
        details = parsed.get("CLASSORINTERFACEDETAIL") or [{}]
        return details[0] if isinstance(details, list) and details else {}

    def _kind_facts(self, name: str, parsed: Dict[str, object]) -> List[str]:
        detail = self._class_detail(parsed)
        facts = []
        if detail.get("ISINTERFACEORNOT"):
            facts.append(f"{name} is an interface.")
        if detail.get("ISABSTRACTORNOT"):
            facts.append(f"{name} is an abstract class.")
        return facts

    def _inheritance_facts(self, name: str, parsed: Dict[str, object]) -> List[str]:
        detail = self._class_detail(parsed)
        facts = [f"{name} extends {_simple_type(t)}." for t in detail.get("EXTENDSFROM") or []]
        facts += [f"{name} implements {_simple_type(t)}." for t in detail.get("IMPLEMENTSFROM") or []]
        return facts

    @staticmethod
    def _constructor_facts(name: str, parsed: Dict[str, object], peers: set) -> List[str]:
        facts = []
        for ctor in parsed.get("CONSTRUCTORDETAIL") or []:
            # A private constructor is the Singleton cue; it is listed before the parameters.
            if "private" in [m.lower() for m in ctor.get("CONSTRUCTORMODIFIER") or []]:
                facts.insert(0, f"{name} has a private constructor.")
            for param in ctor.get("CONSTRUCTORPARAMETER") or []:
                param_type = param.get("PARAMETERTYPE") or ""
                if _refers_to(param_type, peers):
                    facts.append(f"{name}'s constructor takes a parameter of type {_clean_type(param_type)}.")
        return facts

    @staticmethod
    def _field_facts(name: str, parsed: Dict[str, object], peers: set) -> List[str]:
        facts = []
        for fld in parsed.get("FIELDDETAIL") or []:
            field_type = fld.get("FIELDCOMMONTYPE") or fld.get("FIELDDATATYPE") or ""
            if not _refers_to(field_type, peers):
                continue
            is_static = "static" in [m.lower() for m in fld.get("FIELDMODIFIERTYPE") or []]
            for declaration in fld.get("FIELDDECLARATION") or []:
                # "INSTANCE = new Director()" -> "INSTANCE"
                field_name = str(declaration).split("=")[0].strip()
                kind = "static field" if is_static else "field"
                facts.append(f"{name} has a {kind} {field_name} of type {_clean_type(field_type)}.")
        return facts

    @staticmethod
    def _method_facts(name: str, parsed: Dict[str, object]) -> List[str]:
        """Overridden methods first (they carry the pattern's interface), then the rest in source order."""
        overridden, declared = [], []
        for method in parsed.get("METHODDETAIL") or []:
            method_name = method.get("METHODNAME")
            if not method_name:
                continue
            annotations = " ".join(method.get("METHODOVERRIDE") or [])
            if "@Override" in annotations:
                overridden.append(f"{name} overrides method {method_name}.")
            else:
                declared.append(f"{name} defines method {method_name}.")
        return overridden + declared

    @staticmethod
    def _call_facts(name: str, parsed: Dict[str, object]) -> List[str]:
        """Only calls the parser resolved; an empty list does not mean the method calls nothing."""
        facts = []
        for method in parsed.get("METHODDETAIL") or []:
            for call in method.get("OUTGOINGMETHOD") or []:
                callee_class, callee = call.get("CALLEECLASS"), call.get("CALLEEMETHODNAME")
                if callee_class and callee:
                    facts.append(f"Method {method.get('METHODNAME')} of {name} calls {callee} of {callee_class}.")
        return facts


def _clean_type(type_name: str) -> str:
    """Drop type annotations: 'String @Nullable []' -> 'String[]'."""
    return re.sub(r"\s+", "", re.sub(r"@\w+(\([^)]*\))?", "", str(type_name)))


def _simple_type(type_name: str) -> str:
    """Drop generic arguments and package qualifiers: 'java.util.List<Foo>' -> 'List'."""
    base = _clean_type(type_name).split("<")[0]
    return base.split(".")[-1]


def _refers_to(type_name: str, names: set) -> bool:
    """True if a type mentions one of the named classes ('List<Observer>' refers to Observer)."""
    return bool(set(re.findall(r"[A-Za-z_]\w*", _clean_type(type_name))) & names)


class FactRepository:
    """Indexes the DPS JSON files and builds the facts of any evaluated class."""

    def __init__(self, source_dir: Path, builder: ClassFactBuilder) -> None:
        if not source_dir.is_dir():
            raise FileNotFoundError(f"{source_dir} not found; run the DPS pipeline first.")
        self.builder = builder
        # match key -> (parsed class data, names of the classes in the same section, detector roles, class name)
        self._index: Dict[str, Tuple[Dict[str, object], List[str], List[RoleAssignment], str]] = {}
        self.files_read = 0
        self.files_without_roles: List[str] = []
        self._load(source_dir)

    def _load(self, source_dir: Path) -> None:
        reader = DetectorRoleReader()
        files = sorted(source_dir.glob("*.json"))
        if not files:
            raise FileNotFoundError(f"No JSON files in {source_dir}")
        for path in files:
            project = path.stem.split("_", 1)[0]
            data = json.loads(path.read_text(encoding="utf-8"))
            roles = reader.read(data.get("design_pattern") or [])
            if not roles:
                self.files_without_roles.append(path.name)
            self.files_read += 1
            for pattern, classes in data.items():
                if pattern in NON_CLASS_KEYS or not isinstance(classes, dict):
                    continue
                peers = list(classes)
                for class_name, parsed in classes.items():
                    key = build_match_key(project, class_name, pattern)
                    if key in self._index:
                        raise ValueError(f"{path.name}: class {class_name} appears twice for {project}/{pattern}")
                    self._index[key] = (parsed, peers, roles, class_name)

    def facts_for(self, project: str, design_pattern: str, file_name: str, match_key: str) -> Optional[ClassFacts]:
        """The facts of one class, or None when the JSON has no data for it."""
        entry = self._index.get(match_key)
        if entry is None:
            return None
        parsed, peers, roles, class_name = entry
        target = ClassFacts(project, design_pattern, file_name, class_name, match_key)
        self.builder.build(target, parsed, peers, roles)
        return target


# =========================================================================== judge

@dataclass(frozen=True)
class JudgeRequest:
    """One call: a class's facts against one method's summary."""
    match_key: str
    method: str
    prompt: str
    max_tokens: int
    key: str  # hash of everything that determines the reply; identifies it in the cache


@dataclass(frozen=True)
class FactVerdict:
    fact_id: int
    label: str
    evidence: str
    justification: str
    evidence_found: Optional[bool]  # quote found verbatim in the summary; None when there is no quote
    evidence_match: str = ""  # "verbatim", "fragments" or "not_found"; "" when there is no quote


@dataclass(frozen=True)
class ParsedReply:
    valid: bool
    problem: str
    verdicts: Tuple[FactVerdict, ...] = ()

    def count(self, label: str) -> int:
        return sum(v.label == label for v in self.verdicts)


class OpenRouterClient:
    """Sends one chat request; retries only failed requests, never a reply the parser rejects."""

    def __init__(self, settings: FactJudgeSettings) -> None:
        self.settings = settings
        self.session = requests.Session()

    def complete(self, prompt: str, max_tokens: int) -> Dict[str, object]:
        """Return {content, finish_reason, usage, error}; error is empty when a reply arrived."""
        payload = {
            "model": self.settings.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.settings.temperature,
            "max_tokens": max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.settings.api_key}", "Content-Type": "application/json"}
        last_error = ""
        for attempt in range(1, self.settings.max_attempts + 1):
            try:
                response = self.session.post(self.settings.api_url, headers=headers, json=payload,
                                             timeout=self.settings.request_timeout)
                response.raise_for_status()
                body = response.json()
                choice = body["choices"][0]
                return {
                    "content": (choice["message"]["content"] or "").strip(),
                    "finish_reason": choice.get("finish_reason") or "",
                    "usage": body.get("usage") or {},
                    "error": "",
                }
            except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self.settings.max_attempts:
                    time.sleep(self.settings.retry_backoff * 2 ** (attempt - 1))
        return {"content": "", "finish_reason": "", "usage": {}, "error": last_error}


class ReplyParser:
    """Reads the judge's JSON reply and checks it against the facts it was given."""

    def parse(self, raw: str, facts: ClassFacts, summary: str) -> ParsedReply:
        if not raw:
            return ParsedReply(False, "empty reply")
        data = self._load_json(raw)
        if data is None:
            return ParsedReply(False, "reply is not valid JSON (cut off or extra text)")
        entries = data.get("facts") if isinstance(data, dict) else data
        if not isinstance(entries, list):
            return ParsedReply(False, 'no "facts" list in the reply')

        expected = [f.fact_id for f in facts.facts]
        verdicts: Dict[int, FactVerdict] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                return ParsedReply(False, "a facts entry is not an object")
            try:
                fact_id = int(entry.get("id"))
            except (TypeError, ValueError):
                return ParsedReply(False, f"fact id {entry.get('id')!r} is not a number")
            label = str(entry.get("label") or "").strip().upper()
            if label not in LABELS:
                return ParsedReply(False, f"fact {fact_id}: unknown label {entry.get('label')!r}")
            if fact_id in verdicts:
                return ParsedReply(False, f"fact {fact_id} is labelled twice")
            if fact_id not in expected:
                return ParsedReply(False, f"fact {fact_id} was not asked")
            evidence = "" if entry.get("evidence") is None else str(entry.get("evidence")).strip()
            if evidence.lower() == "null":
                evidence = ""
            verdicts[fact_id] = FactVerdict(
                fact_id=fact_id,
                label=label,
                evidence=evidence,
                justification=str(entry.get("justification") or "").strip(),
                evidence_found=self._quote_in(evidence, summary) if evidence else None,
                evidence_match=self._match(evidence, summary),
            )
        missing = [i for i in expected if i not in verdicts]
        if missing:
            return ParsedReply(False, f"no label for fact(s) {missing}")
        return ParsedReply(True, "", tuple(verdicts[i] for i in expected))

    @staticmethod
    def _load_json(raw: str) -> Optional[object]:
        """Parse the reply, tolerating code fences or text around the JSON object."""
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
        for candidate in (text, text[text.find("{"): text.rfind("}") + 1]):
            if not candidate:
                continue
            try:
                return json.loads(candidate)
            except ValueError:
                continue
        return None

    @staticmethod
    def _quote_in(quote: str, summary: str) -> bool:
        """Is the quote in the summary, ignoring case, spacing, backticks and surrounding quotes?"""
        def norm(s: str) -> str:
            return re.sub(r"\s+", " ", s.replace("`", "")).strip().strip("\"'.").lower()
        return norm(quote) in norm(summary)

    @classmethod
    def _match(cls, quote: str, summary: str) -> str:
        """'verbatim', 'fragments' (parts joined with '...', each verbatim) or 'not_found'; '' for no quote."""
        if not quote:
            return ""
        if cls._quote_in(quote, summary):
            return "verbatim"
        if ELLIPSIS.search(quote):
            parts = [p.strip(" ,;:") for p in ELLIPSIS.split(quote)]
            parts = [p for p in parts if p.strip("\"'.")]
            if parts and all(cls._quote_in(p, summary) for p in parts):
                return "fragments"
        return "not_found"


class JudgementStore:
    """Append-only JSONL of every judge reply; the resume cache and the record of the run."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._latest: Dict[str, Dict[str, object]] = {}
        self._retried: set = set()
        if path.exists():
            for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    print(f"  WARNING: {path.name} line {line_no} is not valid JSON and is ignored")
                    continue
                self._remember(record)

    def _remember(self, record: Dict[str, object]) -> None:
        key = str(record.get("request_key"))
        self._latest[key] = record  # a later reply for the same request replaces an earlier one
        if record.get("retry"):
            self._retried.add(key)

    def __len__(self) -> int:
        return len(self._latest)

    def latest(self, key: str) -> Optional[Dict[str, object]]:
        return self._latest.get(key)

    def was_retried(self, key: str) -> bool:
        return key in self._retried

    def append(self, record: Dict[str, object]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._remember(record)


class FactJudge:
    """Builds requests from the prompt template and sends them to the judge model."""

    def __init__(self, settings: FactJudgeSettings, template: PromptTemplate,
                 client: Optional[OpenRouterClient]) -> None:
        self.settings = settings
        self.template = template
        self.client = client

    def request_for(self, facts: ClassFacts, method: Method, summary: str) -> JudgeRequest:
        prompt = self.template.render(facts.class_name, facts.numbered_list(), summary)
        max_tokens = self.settings.token_budget(len(facts.facts))
        identity = json.dumps({"model": self.settings.model, "temperature": self.settings.temperature,
                               "max_tokens": max_tokens, "prompt": prompt}, sort_keys=True)
        key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return JudgeRequest(facts.match_key, method.slug, prompt, max_tokens, key)

    def ask(self, request: JudgeRequest, retry: bool) -> Dict[str, object]:
        if self.client is None:
            raise RuntimeError("the judge was created without an API client")
        reply = self.client.complete(request.prompt, request.max_tokens)
        return {
            "request_key": request.key,
            "match_key": request.match_key,
            "method": request.method,
            "model": self.settings.model,
            "temperature": self.settings.temperature,
            "max_tokens": request.max_tokens,
            "retry": retry,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "raw": reply["content"],
            "finish_reason": reply["finish_reason"],
            "usage": reply["usage"],
            "error": reply["error"],
        }


# =========================================================================== evaluation items

@dataclass
class EvaluationItem:
    """One evaluated class with its summaries and facts, or the reason it cannot be judged."""
    project: str
    design_pattern: str
    file_name: str
    match_key: str
    summaries: Dict[str, str]
    facts: Optional[ClassFacts]
    status: str  # "judge", "missing summary", "no JSON data" or "no facts"

    @property
    def label(self) -> str:
        return f"{self.project}/{self.design_pattern}/{self.file_name}"

    def words(self, method: Method) -> int:
        """Length of the method's summary in whitespace-separated words, as in summary_length_stats.py."""
        return len(self.summaries.get(method.slug, "").split())


@dataclass
class Judgement:
    """The outcome of one class and method, as used in the outputs."""
    item: EvaluationItem
    method: Method
    request: JudgeRequest
    record: Optional[Dict[str, object]]
    parsed: Optional[ParsedReply]
    origin: str  # "new", "cached", "re-asked" or "not judged"

    @property
    def valid(self) -> bool:
        return self.parsed is not None and self.parsed.valid

    @property
    def truncated(self) -> bool:
        return bool(self.record) and self.record.get("finish_reason") == "length"


# =========================================================================== results

class ResultBuilder:
    """Turns the judgements into the output tables and the results.txt block.

    Rates compare the methods on the same classes: only classes for which every method's
    judgement is valid are counted, so one invalid reply cannot favour a method.
    """

    FACT_COLUMNS = ["project", "design_pattern", "file", "class_name", "fact_id", "fact_type", "source",
                    "category", "role_pattern", "fact_text"]
    VERDICT_COLUMNS = ["project", "design_pattern", "file", "method", "method_slug", "summary_words", "fact_id",
                       "fact_type", "source", "category", "role_pattern", "fact_text", "label", "evidence",
                       "evidence_found", "evidence_match", "justification", "in_comparison"]

    def __init__(self, judgements: List[Judgement], items: List[EvaluationItem],
                 settings: FactJudgeSettings, corpus_set: CorpusSet) -> None:
        self.judgements = judgements
        self.items = items
        self.settings = settings
        self.corpus_set = corpus_set
        self.methods = corpus_set.methods
        by_class: Dict[str, List[Judgement]] = {}
        for j in judgements:
            by_class.setdefault(j.item.match_key, []).append(j)
        self.common_keys = {k for k, js in by_class.items()
                            if len(js) == len(self.methods) and all(j.valid for j in js)}

    # ----- tables

    def facts_table(self) -> pd.DataFrame:
        rows = []
        for item in self.items:
            if item.facts is None:
                continue
            for f in item.facts.facts:
                rows.append({"project": item.project, "design_pattern": item.design_pattern,
                             "file": item.file_name, "class_name": item.facts.class_name,
                             "fact_id": f.fact_id, "fact_type": f.fact_type, "source": f.source,
                             "category": f.category, "role_pattern": f.role_pattern, "fact_text": f.text})
        return pd.DataFrame(rows, columns=self.FACT_COLUMNS)

    def verdicts_table(self) -> pd.DataFrame:
        rows = []
        for j in self.judgements:
            if not j.valid:
                continue
            facts = {f.fact_id: f for f in j.item.facts.facts}
            for v in j.parsed.verdicts:
                fact = facts[v.fact_id]
                rows.append({"project": j.item.project, "design_pattern": j.item.design_pattern,
                             "file": j.item.file_name, "method": j.method.label, "method_slug": j.method.slug,
                             "summary_words": j.item.words(j.method),
                             "fact_id": v.fact_id, "fact_type": fact.fact_type, "source": fact.source,
                             "category": fact.category, "role_pattern": fact.role_pattern,
                             "fact_text": fact.text, "label": v.label, "evidence": v.evidence,
                             "evidence_found": v.evidence_found, "evidence_match": v.evidence_match,
                             "justification": v.justification,
                             "in_comparison": j.item.match_key in self.common_keys})
        return pd.DataFrame(rows, columns=self.VERDICT_COLUMNS)

    def detail_table(self) -> pd.DataFrame:
        rows = []
        for item in self.items:
            for m in self.methods:
                row = {"project": item.project, "design_pattern": item.design_pattern, "file": item.file_name,
                       "method": m.label, "status": item.status,
                       "summary_words": item.words(m) if item.summaries.get(m.slug) else None,
                       "n_facts": len(item.facts.facts) if item.facts else 0}
                j = next((x for x in self.judgements if x.item is item and x.method is m), None)
                if j is not None:
                    row.update({
                        "origin": j.origin,
                        "valid": j.valid,
                        "problem": j.parsed.problem if j.parsed else (j.record or {}).get("error", "not judged"),
                        "truncated": j.truncated,
                        "max_tokens": j.request.max_tokens,
                        **{label.lower(): j.parsed.count(label) if j.valid else None for label in LABELS},
                        "in_comparison": item.match_key in self.common_keys,
                        "raw": (j.record or {}).get("raw", ""),
                    })
                rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def scopes() -> List[Tuple[str, str, Optional[str]]]:
        """(scope, value, verdicts column): all facts, each source, category, fact type, and role pattern."""
        scopes: List[Tuple[str, str, Optional[str]]] = [("all", "", None)]
        scopes += [("source", s, "source") for s in SOURCES]
        scopes += [("category", c, "category") for c in CATEGORIES]
        scopes += [("fact_type", t, "fact_type") for t, _ in FACT_TYPES]
        scopes += [("role_pattern", p, "role_pattern") for p in ROLE_PATTERNS]
        return scopes

    def summary_table(self, verdicts: pd.DataFrame) -> pd.DataFrame:
        """Per method and scope: label counts and rates, the quote-match sensitivity rates and mean length."""
        compared = verdicts[verdicts["in_comparison"]] if not verdicts.empty else verdicts
        class_columns = ["project", "design_pattern", "file"]
        rows = []
        for scope, value, column in self.scopes():
            subset = compared if column is None or compared.empty else compared[compared[column] == value]
            for m in self.methods:
                part = subset[subset["method_slug"] == m.slug] if not subset.empty else subset
                n = len(part)
                classes = part.groupby(class_columns).ngroups if n else 0
                row = {"scope": scope, "value": value or "all facts", "method": m.label,
                       "classes": int(classes), "facts": n,
                       # Mean over the classes in scope, each summary counted once.
                       "mean_words": round(float(part.drop_duplicates(class_columns)["summary_words"].mean()), 1)
                       if n else None}
                for label in LABELS:
                    count = int((part["label"] == label).sum()) if n else 0
                    row[label.lower()] = count
                    row[f"{label.lower()}_rate"] = round(count / n, 4) if n else None
                # Macro rate: the preserved share of each class's facts, averaged over classes.
                if n:
                    per_class = part.groupby(class_columns)["label"].apply(lambda s: (s == "PRESERVED").mean())
                    row["macro_preserved_rate"] = round(float(per_class.mean()), 4)
                else:
                    row["macro_preserved_rate"] = None
                # Sensitivity check: PRESERVED counted only when its quote is verbatim, or verbatim in parts.
                preserved = part[part["label"] == "PRESERVED"] if n else part
                verbatim = int((preserved["evidence_match"] == "verbatim").sum()) if n else 0
                supported = int(preserved["evidence_match"].isin(["verbatim", "fragments"]).sum()) if n else 0
                row["preserved_verbatim_quote"] = verbatim
                row["preserved_verbatim_quote_rate"] = round(verbatim / n, 4) if n else None
                row["preserved_verbatim_or_fragments"] = supported
                row["preserved_verbatim_or_fragments_rate"] = round(supported / n, 4) if n else None
                rows.append(row)
        return pd.DataFrame(rows)

    # ----- results block

    def judgement_counts(self) -> Dict[str, Dict[str, int]]:
        counts: Dict[str, Dict[str, int]] = {}
        for m in self.methods:
            js = [j for j in self.judgements if j.method is m]
            counts[m.slug] = {
                "asked": sum(j.record is not None for j in js),
                "valid": sum(j.valid for j in js),
                "truncated": sum(j.truncated for j in js),
                "retried": sum(j.origin == "re-asked" for j in js),
            }
        return counts

    def evidence_counts(self, verdicts: pd.DataFrame) -> Dict[str, Dict[str, int]]:
        """Per method (classes compared, any label): quotes not verbatim, and how many of those are verbatim parts."""
        counts: Dict[str, Dict[str, int]] = {}
        compared = verdicts[verdicts["in_comparison"]] if not verdicts.empty else verdicts
        for m in self.methods:
            part = compared[compared["method_slug"] == m.slug] if not compared.empty else compared
            counts[m.slug] = {
                "evidence_not_found": int((part["evidence_found"] == False).sum()) if len(part) else 0,  # noqa: E712
                "evidence_fragments": int((part["evidence_match"] == "fragments").sum()) if len(part) else 0,
                "evidence_not_found_in_parts": int((part["evidence_match"] == "not_found").sum()) if len(part) else 0,
            }
        return counts

    @staticmethod
    def _pct(value: object) -> str:
        return "-" if value is None or pd.isna(value) else f"{100 * float(value):.1f}%"

    @staticmethod
    def _words(value: object) -> str:
        return "-" if value is None or pd.isna(value) else f"{float(value):.1f}"

    @staticmethod
    def _row_of(summary: pd.DataFrame, scope: str, value: str, method: Method) -> pd.Series:
        return summary[(summary["scope"] == scope) & (summary["value"] == value)
                       & (summary["method"] == method.label)].iloc[0]

    def format_block(self, summary: pd.DataFrame, verdicts: pd.DataFrame, generated_at: str) -> str:
        body = (self._default_body(summary, verdicts) if self.corpus_set.name == "default"
                else self._word_limit_body(summary, verdicts))
        lines = [self.corpus_set.block_start, *self._header(generated_at), *body, self.corpus_set.block_end]
        return "\n".join(lines) + "\n"

    def _header(self, generated_at: str) -> List[str]:
        facts = self.facts_table()
        counts = self.judgement_counts()
        judged_items = [i for i in self.items if i.status == "judge"]
        skipped = Counter(i.status for i in self.items if i.status != "judge")
        roles = facts[facts["fact_type"] == "role"]
        return [
            "=" * 78,
            self.corpus_set.title,
            "=" * 78,
            f"Generated: {generated_at}",
            *self.settings.describe(),
            f"Prompt: resources/prompts.json [{PROMPT_SECTION}.{PROMPT_KEY}]",
            f"Fact source: {FACT_SOURCE_DIR.relative_to(REPO_ROOT)}",
            f"Classes: {len(self.items)} evaluated, {len(judged_items)} with facts"
            + (f" (not judged: {', '.join(f'{k} {v}' for k, v in sorted(skipped.items()))})" if skipped else ""),
            f"Facts: {len(facts)} in total; " + ", ".join(
                f"{s} {int((facts['source'] == s).sum())}" for s in SOURCES) if not facts.empty else "Facts: 0",
            f"Role facts naming the class's own pattern / another pattern: "
            + " / ".join(str(int((roles["role_pattern"] == p).sum())) for p in ROLE_PATTERNS),
            "Judgements (asked / valid / cut off / re-asked): " + "; ".join(
                f"{m.label} {counts[m.slug]['asked']}/{counts[m.slug]['valid']}/"
                f"{counts[m.slug]['truncated']}/{counts[m.slug]['retried']}" for m in self.methods),
            f"Classes compared (valid judgement for all {len(self.methods)} methods): {len(self.common_keys)}",
        ]

    def _evidence_lines(self, summary: pd.DataFrame, verdicts: pd.DataFrame) -> List[str]:
        evidence = self.evidence_counts(verdicts)
        return [
            "",
            "EVIDENCE QUOTES (classes compared only). A quote that is not verbatim is not necessarily wrong:",
            "the judge may join or reword passages. The two rates below are a sensitivity check, not an error count.",
            "Quotes not found verbatim in the summary: "
            + ", ".join(f"{m.label} {evidence[m.slug]['evidence_not_found']}" for m in self.methods),
            "  of which parts joined with '...', each verbatim: "
            + ", ".join(f"{m.label} {evidence[m.slug]['evidence_fragments']}" for m in self.methods),
            "Preserved rate, counting only verbatim quotes: "
            + ", ".join(f"{m.label} {self._pct(self._row_of(summary, 'all', 'all facts', m)['preserved_verbatim_quote_rate'])}"
                        for m in self.methods),
            "Preserved rate, counting verbatim quotes and verbatim parts: "
            + ", ".join(f"{m.label} {self._pct(self._row_of(summary, 'all', 'all facts', m)['preserved_verbatim_or_fragments_rate'])}"
                        for m in self.methods),
        ]

    def _default_body(self, summary: pd.DataFrame, verdicts: pd.DataFrame) -> List[str]:
        pct, row_of = self._pct, lambda scope, value, m: self._row_of(summary, scope, value, m)
        lines = [
            "",
            "ALL FACTS (classes compared only)",
            f"{'Method':<20}{'Mean words':>11}{'Facts':>7}{'Preserved':>11}{'Omitted':>10}{'Incorrect':>11}{'Macro pres.':>13}",
        ]
        for m in self.methods:
            r = row_of("all", "all facts", m)
            lines.append(f"{m.label:<20}{self._words(r['mean_words']):>11}{int(r['facts']):>7}{pct(r['preserved_rate']):>11}"
                         f"{pct(r['omitted_rate']):>10}{pct(r['incorrect_rate']):>11}{pct(r['macro_preserved_rate']):>13}")

        lines += ["", "PRESERVED RATE BY FACT SOURCE, CATEGORY, TYPE AND ROLE PATTERN (facts per method in brackets)",
                  f"{'Scope':<16}" + "".join(f"{m.label:>22}" for m in self.methods)]
        for scope, value, _ in self.scopes()[1:]:
            cells = []
            for m in self.methods:
                r = row_of(scope, value, m)
                cells.append(f"{pct(r['preserved_rate'])} ({int(r['facts'])})")
            lines.append(f"{value:<16}" + "".join(f"{c:>22}" for c in cells))

        lines += self._evidence_lines(summary, verdicts)
        lines += ["", r"\begin{table}[htbp]", r"  \centering",
                  r"  \caption{Design-pattern fact preservation (LLM judge; "
                  + f"{len(self.common_keys)} classes" + r").}",
                  r"  \label{tab:fact-preservation}",
                  r"  \begin{tabular}{lrrrrrr}", r"    \toprule",
                  r"    Method & Mean words & Facts & Preserved & Omitted & Incorrect & Macro preserved \\",
                  r"    \midrule"]
        for m in self.methods:
            r = row_of("all", "all facts", m)
            cells = [pct(r[c]).replace("%", r"\%") for c in
                     ("preserved_rate", "omitted_rate", "incorrect_rate", "macro_preserved_rate")]
            lines.append(f"    {m.label.replace('_', chr(92) + '_')} & {self._words(r['mean_words'])} & "
                         f"{int(r['facts'])} & " + " & ".join(cells) + r" \\")
        lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
        return lines

    def _word_limit_body(self, summary: pd.DataFrame, verdicts: pd.DataFrame) -> List[str]:
        """One row per corpus: with twelve corpora, the corpora go down the page, not across it."""
        pct, row_of = self._pct, lambda scope, value, m: self._row_of(summary, scope, value, m)
        lines = ["", "PRESERVED RATE BY CORPUS AND CATEGORY (classes compared only)",
                 f"{'Corpus':<20}{'Mean words':>11}{'All':>8}" + "".join(f"{c:>16}" for c in CATEGORIES)
                 + f"{'Incorrect':>11}"]
        for m in self.methods:
            r = row_of("all", "all facts", m)
            lines.append(f"{m.label:<20}{self._words(r['mean_words']):>11}{pct(r['preserved_rate']):>8}"
                         + "".join(f"{pct(row_of('category', c, m)['preserved_rate']):>16}" for c in CATEGORIES)
                         + f"{pct(r['incorrect_rate']):>11}")

        types = [t for t, _ in FACT_TYPES]
        lines += ["", "PRESERVED RATE BY CORPUS AND FACT TYPE (classes compared only; role split by role pattern)",
                  f"{'Corpus':<20}" + "".join(f"{t:>13}" for t in types) + "".join(f"{p:>15}" for p in ROLE_PATTERNS)]
        for m in self.methods:
            lines.append(f"{m.label:<20}" + "".join(f"{pct(row_of('fact_type', t, m)['preserved_rate']):>13}" for t in types)
                         + "".join(f"{pct(row_of('role_pattern', p, m)['preserved_rate']):>15}" for p in ROLE_PATTERNS))
        first = self.methods[0]
        lines.append("Facts per corpus: " + ", ".join(
            f"{v} {int(row_of(s, v, first)['facts'])}" for s, v, _ in self.scopes()
            if s in ("all", "category", "fact_type", "role_pattern") and v) + f", all {int(row_of('all', 'all facts', first)['facts'])}")

        lines += self._evidence_lines(summary, verdicts)
        lines += ["", r"\begin{table}[htbp]", r"  \centering",
                  r"  \caption{Design-pattern fact preservation by category under word limits (LLM judge; "
                  + f"{len(self.common_keys)} classes" + r").}",
                  r"  \label{tab:fact-preservation-word-limits}",
                  r"  \begin{tabular}{lrrrrrr}", r"    \toprule",
                  r"    Corpus & Mean words & All & Pattern role & Responsibility & Relationship & Identity \\",
                  r"    \midrule"]
        for m in self.methods:
            r = row_of("all", "all facts", m)
            cells = [pct(r["preserved_rate"])] + [pct(row_of("category", c, m)["preserved_rate"]) for c in CATEGORIES]
            lines.append(f"    {m.label.replace('_', chr(92) + '_')} & {self._words(r['mean_words'])} & "
                         + " & ".join(c.replace("%", r"\%") for c in cells) + r" \\")
        lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
        return lines

    def results_table(self, summary: pd.DataFrame, verdicts: pd.DataFrame, generated_at: str) -> pd.DataFrame:
        """Every figure of the results block as one long table (section, scope, method, metric, value)."""
        rows: List[Dict[str, object]] = []

        def add(section: str, metric: str, value: object, scope: str = "", method: str = "") -> None:
            rows.append({"section": section, "scope": scope, "method": method, "metric": metric, "value": value})

        add("run", "generated_at", generated_at)
        add("run", "corpora", self.corpus_set.name)
        add("run", "judge_model", self.settings.model)
        add("run", "temperature", self.settings.temperature)
        add("run", "tokens_per_fact", self.settings.tokens_per_fact)
        add("run", "max_tokens", self.settings.max_tokens)
        add("run", "fact_caps", self.settings.caps.describe())
        add("run", "classes_evaluated", len(self.items))
        add("run", "classes_compared", len(self.common_keys))
        evidence = self.evidence_counts(verdicts)
        for m in self.methods:
            for metric, value in self.judgement_counts()[m.slug].items():
                add("judgements", metric, value, method=m.label)
            for metric, value in evidence[m.slug].items():
                add("judgements", metric, value, method=m.label)
        for _, r in summary.iterrows():
            for metric in ("facts", "mean_words", "preserved", "omitted", "incorrect", "preserved_rate", "omitted_rate",
                           "incorrect_rate", "macro_preserved_rate", "preserved_verbatim_quote_rate",
                           "preserved_verbatim_or_fragments_rate"):
                add("preservation", metric, r[metric], scope=f"{r['scope']}:{r['value']}", method=r["method"])
        return pd.DataFrame(rows)


def write_block(results_txt: Path, section: str, block_start: str, block_end: str) -> None:
    """Write the section into results.txt, replacing any previous block with the same markers."""
    existing = results_txt.read_text(encoding="utf-8") if results_txt.exists() else ""
    start, end = existing.find(block_start), existing.find(block_end)
    if start != -1 and end != -1:
        updated = existing[:start] + section + existing[end + len(block_end) + 1:]
    else:
        separator = "" if existing.endswith("\n\n") or not existing else "\n\n"
        updated = existing + separator + section
    results_txt.write_text(updated, encoding="utf-8")


# =========================================================================== pipeline

class FactPreservationEvaluation:
    """Runs the evaluation: load inputs, extract facts, judge, and write the outputs."""

    def __init__(self, mode: str, limit: Optional[int], corpus_set: CorpusSet) -> None:
        self.mode = mode  # "dry-run", "run", "reparse" or "retry-invalid"
        self.limit = limit
        self.corpus_set = corpus_set
        self.methods = corpus_set.methods
        self.settings = FactJudgeSettings.from_env(ENV_FILE, need_api=mode in ("run", "retry-invalid"))
        self.template = PromptTemplate.load(PROMPTS_JSON, PROMPT_SECTION, PROMPT_KEY)
        self.repository = FactRepository(FACT_SOURCE_DIR, ClassFactBuilder(self.settings.caps))
        client = OpenRouterClient(self.settings) if mode in ("run", "retry-invalid") else None
        self.judge = FactJudge(self.settings, self.template, client)
        self.parser = ReplyParser()

    # ----- inputs

    def load_items(self) -> List[EvaluationItem]:
        human = load_human_summaries()
        summaries = {m.slug: load_method_summaries(m) for m in self.methods}
        if self.limit is not None:
            human = human.head(self.limit)
        items = []
        for row in human.itertuples(index=False):
            texts = {m.slug: summaries[m.slug].get(row.match_key, "") for m in self.methods}
            facts = self.repository.facts_for(row.project, row.design_pattern, row.file_name, row.match_key)
            if any(not t for t in texts.values()):
                status = "missing summary"
            elif facts is None:
                status = "no JSON data"
            elif not facts.facts:
                status = "no facts"
            else:
                status = "judge"
            items.append(EvaluationItem(row.project, row.design_pattern, row.file_name, row.match_key,
                                        texts, facts, status))
        return items

    # ----- console

    @staticmethod
    def banner(title: str) -> None:
        print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)

    def report_facts(self, items: List[EvaluationItem]) -> None:
        """Fact extraction checks: coverage, facts per type, caps applied and token budget."""
        self.banner("FACT EXTRACTION")
        print(f"JSON files read: {self.repository.files_read} from {FACT_SOURCE_DIR.relative_to(REPO_ROOT)}")
        print(f"JSON files with no detector roles: {len(self.repository.files_without_roles)}")
        status = Counter(i.status for i in items)
        print(f"Classes: {len(items)}; with facts: {status['judge']}"
              + "".join(f"; {k}: {v}" for k, v in sorted(status.items()) if k != "judge"))
        for item in items:
            if item.status != "judge":
                print(f"  NOT JUDGED ({item.status}): {item.label}")

        judged = [i for i in items if i.status == "judge"]
        if not judged:
            return
        per_type = Counter(f.fact_type for i in judged for f in i.facts.facts)
        dropped = Counter()
        for i in judged:
            dropped.update(i.facts.dropped)
        sizes = [len(i.facts.facts) for i in judged]
        print(f"Facts: {sum(sizes)} (per class: mean {sum(sizes) / len(sizes):.1f}, min {min(sizes)}, max {max(sizes)})")
        print(f"{'Fact type':<14}{'source':<10}{'facts':>7}{'classes':>9}{'left out by cap':>17}")
        for fact_type, source in FACT_TYPES:
            classes = sum(any(f.fact_type == fact_type for f in i.facts.facts) for i in judged)
            print(f"{fact_type:<14}{source:<10}{per_type[fact_type]:>7}{classes:>9}{dropped[fact_type]:>17}")
        no_role = [i for i in judged if not any(f.source == "detector" for f in i.facts.facts)]
        print(f"Classes without a detector role fact: {len(no_role)}")
        role_patterns = Counter(f.role_pattern for i in judged for f in i.facts.facts if f.fact_type == "role")
        print("Role facts naming the class's own pattern / another pattern: "
              + " / ".join(str(role_patterns[p]) for p in ROLE_PATTERNS))
        print(f"Classes with only method facts: "
              f"{sum(all(f.fact_type == 'method' for f in i.facts.facts) for i in judged)}")
        over = [i for i in judged if not self.settings.fits(len(i.facts.facts))]
        budget = sum(self.settings.token_budget(len(i.facts.facts)) for i in judged)
        print(f"Completion-token ceiling for all {len(judged) * len(self.methods)} calls: {budget * len(self.methods)}")
        if over:
            print(f"  WARNING: {len(over)} classes need more than FACT_JUDGE_MAX_TOKENS; their replies may be cut off:")
            for i in over:
                print(f"    {i.label} ({len(i.facts.facts)} facts)")

    # ----- judging

    def judge_all(self, items: List[EvaluationItem]) -> List[Judgement]:
        store = JudgementStore(RAW_JSONL)
        judged = [i for i in items if i.status == "judge"]
        self.banner({"run": "JUDGING", "retry-invalid": "JUDGING (RE-ASKING INVALID REPLIES)",
                     "reparse": "RE-SCORING SAVED REPLIES (NO API CALLS)"}[self.mode])
        for line in self.settings.describe():
            print(line)
        print(f"Saved replies in {RAW_JSONL.name}: {len(store)}")
        print(f"Classes to judge: {len(judged)} x {len(self.methods)} methods = {len(judged) * len(self.methods)} judgements")

        judgements: List[Judgement] = []
        calls = 0
        for index, item in enumerate(judged, start=1):
            print(f"\n[{index}/{len(judged)}] {item.label} - {len(item.facts.facts)} facts")
            for m in self.methods:
                judgement = self._judge_one(store, item, m)
                calls += judgement.origin in ("new", "re-asked")
                judgements.append(judgement)
                print("    " + self._describe(judgement))
        print(f"\nAPI calls made: {calls}")
        return judgements

    def _judge_one(self, store: JudgementStore, item: EvaluationItem, method: Method) -> Judgement:
        summary = item.summaries[method.slug]
        request = self.judge.request_for(item.facts, method, summary)
        record = store.latest(request.key)
        # A failed API call is not a reply: it is asked again in any mode that makes calls.
        has_reply = record is not None and not record.get("error")

        if has_reply:
            parsed = self.parser.parse(str(record.get("raw") or ""), item.facts, summary)
            origin = "re-asked" if store.was_retried(request.key) else "cached"
            # Invalid replies are asked again only on request, and only once, so re-asking
            # cannot favour whichever method's replies the model happens to format cleanly.
            if parsed.valid or self.mode != "retry-invalid" or store.was_retried(request.key):
                return Judgement(item, method, request, record, parsed, origin)
            retry = True
        elif self.mode == "reparse":
            return Judgement(item, method, request, record, None, "not judged")
        else:
            retry = False

        record = self.judge.ask(request, retry=retry)
        store.append(record)
        parsed = None if record["error"] else self.parser.parse(str(record["raw"]), item.facts, summary)
        return Judgement(item, method, request, record, parsed, "re-asked" if retry else "new")

    @staticmethod
    def _describe(j: Judgement) -> str:
        name = f"{j.method.label:<18}"
        if j.parsed is None:
            reason = (j.record or {}).get("error") or "no saved reply"
            return f"{name} NOT JUDGED  ({reason})"
        if not j.parsed.valid:
            cut = ", reply cut off at max_tokens" if j.truncated else ""
            return f"{name} INVALID     ({j.parsed.problem}{cut}) [{j.origin}]"
        p = j.parsed
        missing_quotes = sum(v.evidence_found is False for v in p.verdicts)
        note = f", {missing_quotes} quote(s) not in summary" if missing_quotes else ""
        return (f"{name} preserved {p.count('PRESERVED'):>2}  omitted {p.count('OMITTED'):>2}  "
                f"incorrect {p.count('INCORRECT'):>2}  [{j.origin}{note}]")

    # ----- outputs

    def write_outputs(self, items: List[EvaluationItem], judgements: List[Judgement]) -> None:
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        builder = ResultBuilder(judgements, items, self.settings, self.corpus_set)
        verdicts = builder.verdicts_table()
        summary = builder.summary_table(verdicts)
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cs = self.corpus_set
        verdicts.to_csv(cs.verdicts_csv, index=False)
        builder.detail_table().to_csv(cs.detail_csv, index=False)
        summary.to_csv(cs.summary_csv, index=False)
        builder.results_table(summary, verdicts, generated_at).to_csv(cs.results_csv, index=False)
        section = builder.format_block(summary, verdicts, generated_at)
        write_block(RESULTS_TXT, section, cs.block_start, cs.block_end)

        print("\n" + section)
        for label, path in [("Verdicts", cs.verdicts_csv), ("Detail", cs.detail_csv), ("Summary", cs.summary_csv),
                            ("Results table", cs.results_csv), ("Results block", RESULTS_TXT),
                            ("Raw replies", RAW_JSONL)]:
            print(f"{label} written to: {path}")

    def execute(self) -> None:
        self.banner("DESIGN-PATTERN FACT PRESERVATION")
        print(f"Mode: {self.mode}" + (f" (first {self.limit} classes)" if self.limit is not None else ""))
        print(f"Corpora: {self.corpus_set.name} ({', '.join(m.label for m in self.methods)})")
        items = self.load_items()
        self.report_facts(items)

        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        # The facts do not depend on the corpora, so both corpus sets write the same facts file.
        ResultBuilder([], items, self.settings, self.corpus_set).facts_table().to_csv(FACTS_CSV, index=False)
        print(f"\nFacts written to: {FACTS_CSV}")
        if self.mode == "dry-run":
            print("Dry run: no API calls made and no results written. Check the facts file, then run without --dry-run.")
            return

        judgements = self.judge_all(items)
        self.write_outputs(items, judgements)


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Judge how well the NLG, GPT, Mistral and SWUM summaries preserve the design-pattern "
                    "facts extracted from the DPS JSON."
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Use only the first N classes (for a cheap trial run). Default: all.")
    parser.add_argument("--corpora", choices=sorted(CORPUS_SETS), default="default",
                        help="default: the four default summaries. word-limits: those four plus the GPT and Mistral "
                             "20/40/60/80-word summaries, written to fact_preservation_word_limits_*.csv. Default: default.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="Extract the facts, print the checks and write fact_preservation_facts.csv; no API calls.")
    mode.add_argument("--reparse", action="store_true",
                      help="No API calls: score the replies saved in fact_preservation_raw.jsonl again and "
                           "rewrite all outputs.")
    mode.add_argument("--retry-invalid", action="store_true",
                      help="As a normal run, and ask once more for each saved reply that is not valid.")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_arguments(argv)
    mode = "dry-run" if args.dry_run else "reparse" if args.reparse else "retry-invalid" if args.retry_invalid else "run"
    try:
        FactPreservationEvaluation(mode, args.limit, CORPUS_SETS[args.corpora]).execute()
    except Exception as exc:
        print(f"\nERROR: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
