"""Plain-language correction from the weakest criterion and its missed key points (T1.6)."""
from __future__ import annotations

from coach.scoring.rubric import Question

SHORT_ANSWER = "Your answer was very short. Give a fuller answer with a specific example from your studies or projects."


def correction_for(question: Question, criteria_results: list[dict], use_key_points: bool = False) -> dict:
    """Pick the lowest-scoring criterion and turn its template (+ missed key points if enabled) into one tip."""
    weakest = min(criteria_results, key=lambda c: c["expected"])
    crit = next(c for c in question.criteria if c.id == weakest["criterion_id"])
    missing = [k["text"] for k in weakest["key_points"] if not k["covered"]]
    text = crit.correction
    if use_key_points and missing:
        text += " Strong answers usually: " + "; ".join(missing[:2]) + "."
    return {"criterion_id": crit.id, "criterion": crit.name, "text": text}
