"""Question bank + rubric loader (used by T1.3, T1.6, T1.7).

Rule from the proposal: only approved items may be asked. Items in data/rubric without
`approved_by` are skipped. If data/rubric has no files yet, the UNAPPROVED sample bank in
samples/rubric is used and every report is marked as a sample.
"""
from __future__ import annotations

import itertools
import json
import random
from dataclasses import dataclass, field

from coach.profile import PROJECT_ROOT, Config, resolve

SAMPLE_DIR = PROJECT_ROOT / "samples" / "rubric"


@dataclass
class Criterion:
    id: str
    name: str
    description: str
    key_points: list[str]
    anchors: dict[int, list[str]]   # score level -> example answers
    correction: str
    follow_up: dict[str, str] | None = None   # approved follow-up question per language, asked when weak


@dataclass
class Question:
    id: str
    text: dict[str, str]            # language -> question text
    criteria: list[Criterion]
    approved_by: str | None = None
    bank: str = ""


@dataclass
class Bank:
    questions: list[Question] = field(default_factory=list)
    is_sample: bool = False
    source: str = ""


def _parse(data: dict) -> list[Question]:
    out = []
    for q in data["questions"]:
        crits = [Criterion(c["id"], c["name"], c["description"], c.get("key_points", []),
                           {int(k): v for k, v in c["anchors"].items()}, c["correction"], c.get("follow_up"))
                 for c in q["criteria"]]
        out.append(Question(q["id"], q["text"], crits, q.get("approved_by"), data["bank"]))
    return out


def load_bank(cfg: Config, program: str) -> Bank:
    """Common questions + the program's technical questions."""
    approved_dir = resolve(cfg, cfg.paths.rubric_dir)
    files = sorted(approved_dir.glob("*.json")) if approved_dir.exists() else []
    is_sample = not files
    if is_sample:
        files = sorted(SAMPLE_DIR.glob("*.json"))
    bank = Bank(is_sample=is_sample, source=str((SAMPLE_DIR if is_sample else approved_dir)))
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        if program not in data.get("programs", []):
            continue
        for q in _parse(data):
            if is_sample or q.approved_by:
                bank.questions.append(q)
    # common (behavioral) first, then technical
    bank.questions.sort(key=lambda q: q.bank != "common")
    return bank


def programs(cfg: Config) -> list[str]:
    """Courses in scope (session.programs). The robot never asks the student which one."""
    return list(cfg.session.get("programs") or ["CpE", "ECE"])


def load_session_bank(cfg: Config) -> Bank:
    """Every askable question for a session: common questions + the technical questions of every
    program in scope. Which ones are asked is chosen by pick_questions()."""
    bank = None
    for p in programs(cfg):
        b = load_bank(cfg, p)
        if bank is None:
            bank = Bank(is_sample=b.is_sample, source=b.source)
        for q in b.questions:
            if q.id not in {x.id for x in bank.questions}:
                bank.questions.append(q)
    return bank or Bank()


def pick_questions(bank: Bank, n: int, vary: bool = True, rng: random.Random | None = None) -> list[Question]:
    """Common questions first, then technical questions taking turns between the programs
    (CpE, ECE, CpE, ...), so every session covers both courses. vary=True rotates which technical
    questions are used from session to session."""
    rng = rng or random.Random()
    common = [q for q in bank.questions if q.bank == "common"]
    by_program: dict[str, list[Question]] = {}
    for q in bank.questions:
        if q.bank != "common":
            by_program.setdefault(q.bank, []).append(q)
    if vary:
        for qs in by_program.values():
            rng.shuffle(qs)
    technical = [q for group in itertools.zip_longest(*by_program.values()) for q in group if q]
    n_tech = min(len(technical), max(len(by_program), n - len(common)))   # at least one per program
    return (common[: n - n_tech] + technical[:n_tech])[:n]


def all_questions(cfg: Config) -> list[Question]:
    """Every askable question in scope (for pre-synthesizing audio)."""
    return load_session_bank(cfg).questions
