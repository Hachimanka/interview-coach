"""Synthetic interview data (G3): record format and quality filter. No torch: importable anywhere.

Each record is one interviewer call in EXACTLY the runtime format (coach.interviewer.interviewer
build_messages), plus the teacher's answer as the last assistant message:

    {"task": "question", "persona": "nervous", "messages": [system, examples..., user, assistant]}

validate() runs the same guard the robot uses at runtime, so the student model only learns from
replies that the robot would have been allowed to say.

PRIVACY: this data is made up by a teacher model from personas. Never put real student recordings or
transcripts here unless consent and ethics approval cover it (RA 10173); data/ is never committed.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from coach.interviewer.guard import check_turn
from coach.interviewer.interviewer import build_messages

PERSONAS = {
    "strong": "a confident student who gives specific, structured answers with real technical details and results",
    "average": "an average student who answers correctly but briefly and without many specifics",
    "weak": "a student who gives short, vague answers and sometimes goes off topic",
    "nervous": "a nervous student who hesitates, uses fillers like um and uh, and sometimes says they are nervous",
    "taglish": "a student who mixes English and Filipino (Taglish), for example 'Yung project namin is an IoT system'",
    "rambling": "a student who talks a lot, repeats themselves and takes long to reach the point",
}

PROJECTS = [
    "an RFID attendance system with an ESP32 and a web dashboard", "a line-following robot with an Arduino",
    "a solar-powered weather station that sends data over WiFi", "a smart water meter with a flow sensor and MQTT",
    "an automatic plant watering system with a soil moisture sensor", "a parking space detector with ultrasonic sensors",
    "a PCB for a small audio amplifier designed in KiCad", "a mobile app that controls home lights through Bluetooth",
    "an FPGA traffic light controller written in Verilog", "a face-recognition door lock with a Raspberry Pi",
]

TASK = re.compile(r"^TASK: (\w+)", re.M)
KINDS = {"QUESTION": "question", "FOLLOW_UP": "follow_up", "COACH": "coach"}
NAME = re.compile(r"^NAME: (.+)$", re.M)


def make_record(user: str, answer: str, persona: str, max_words: int = 40) -> dict:
    task = TASK.search(user).group(1)
    return {"task": task.lower(), "persona": persona,
            "messages": build_messages(user, max_words) + [{"role": "assistant", "content": answer.strip()}]}


def _section(user: str, label: str) -> str:
    m = re.search(rf"^{label}:\s*(.*?)(?=^\w[\w ]*:|\Z)", user, re.M | re.S)
    return m.group(1).strip() if m else ""


def validate(record: dict) -> tuple[bool, str]:
    """(ok, reason): the runtime guard on the record's answer, with the same context the robot would use."""
    msgs = record.get("messages") or []
    if len(msgs) < 3 or msgs[-1]["role"] != "assistant" or msgs[-2]["role"] != "user":
        return False, "bad message layout"
    user, answer = msgs[-2]["content"], msgs[-1]["content"]
    m = TASK.search(user)
    if not m:
        return False, "no TASK line"
    task = m.group(1)
    if task == "FACTS":
        lines = [x for x in answer.splitlines() if x.strip()]
        ok = 1 <= len(lines) <= 3 and all(x.strip().startswith("- ") for x in lines)
        return ok, "ok" if ok else "facts must be 1-3 lines starting with '- '"
    kind = KINDS.get(task)
    if kind is None:
        return False, f"unknown task {task}"
    nm = NAME.search(user)
    name = nm.group(1).strip() if nm and not nm.group(1).startswith("(") else None
    memory = " ".join([_section(user, "NOTES"), _section(user, "STUDENT JUST SAID"), _section(user, "CALLBACK"),
                       name or ""])
    return check_turn(answer, kind, user, name, allow_praise=kind == "coach",
                      memory=None if kind == "coach" else memory)


def write_jsonl(records, path: str | Path) -> tuple[int, int]:
    """Write only valid records; returns (kept, dropped)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kept = dropped = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            ok, why = validate(r)
            if ok:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                kept += 1
            else:
                dropped += 1
    return kept, dropped


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
