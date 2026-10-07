"""Question bank + rubric loader (used by T1.3, T1.6, T1.7).

Rule from the proposal: only approved items may be asked. Items in data/rubric without
`approved_by` are skipped. If data/rubric has no files yet, the UNAPPROVED sample bank in
samples/rubric is used and every report is marked as a sample.
"""
from __future__ import annotations

import json
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
                           {int(k): v for k, v in c["anchors"].items()}, c["correction"])
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


def all_questions(cfg: Config) -> list[Question]:
    """Every askable question across programs (for pre-synthesizing audio)."""
    seen: dict[str, Question] = {}
    for program in ("CpE", "EE", "ECE"):
        for q in load_bank(cfg, program).questions:
            seen[q.id] = q
    return list(seen.values())
