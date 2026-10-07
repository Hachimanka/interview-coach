"""Session report (T1.7): rubric score per answer, delivery measures, and at least one specific
correction, as JSON (for progress tracking and T3.1 training) and a printable HTML page."""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def build_report(result: dict) -> dict:
    answers = []
    for a in result["answers"]:
        r = a.result or {}
        answers.append({
            "question_id": a.question.id,
            "question": a.question.text,
            "end_reason": a.end_reason,
            "duration_s": a.duration_s,
            "eye": a.eye,
            **r,
        })
    scored = [x for x in answers if "score" in x]
    summary = {"answers": len(answers)}
    if scored:
        summary["average_score"] = round(float(np.mean([x["score"]["score"] for x in scored])), 2)
        weakest = min(scored, key=lambda x: x["score"]["score"])
        corrections = [weakest["score"]["correction"]["text"]]
        tips = [t for x in scored for t in x["delivery"]["tips"]]
        if tips:   # most frequent delivery tip
            corrections.append(max(set(tips), key=tips.count))
        summary["top_corrections"] = corrections
        wpms = [x["delivery"]["wpm"] for x in scored if x["delivery"]["wpm"]]
        summary["average_wpm"] = round(float(np.mean(wpms)), 1) if wpms else None
        summary["total_fillers"] = sum(x["delivery"]["fillers"]["total"] for x in scored)
        summary["long_pauses"] = sum(len(x["delivery"]["long_pauses"]) for x in scored)
        eyes = [x["delivery"]["eye_contact_pct"] for x in scored if x["delivery"]["eye_contact_pct"] is not None]
        summary["eye_contact_pct"] = round(float(np.mean(eyes)), 1) if eyes else None
    return {"meta": result["meta"], "summary": summary, "answers": answers}


def save(report: dict, out_dir: Path, audio: dict[str, list[np.ndarray]] | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "report.html").write_text(render_html(report), encoding="utf-8")
    for qid, blocks in (audio or {}).items():
        if blocks:
            sf.write(out_dir / f"{qid}.wav", np.concatenate(blocks), 16000)
    return out_dir / "report.html"


def _bar(score: float) -> str:
    pct = max(0, min(100, (score - 1) / 4 * 100))
    return f'<span class="bar"><span style="width:{pct:.0f}%"></span></span>'


def render_html(report: dict) -> str:
    m, s = report["meta"], report["summary"]
    lang = m.get("language", "en")
    e = html.escape
    parts = [f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Practice Interview Report</title>
<style>
body{{font:15px/1.55 system-ui,Segoe UI,Roboto,sans-serif;color:#1d1d1b;background:#fff;max-width:820px;margin:0 auto;padding:24px 16px}}
h1{{font-size:1.6rem;margin:.2em 0}} h2{{font-size:1.15rem;margin:1.6em 0 .4em;border-top:1px solid #ddd;padding-top:.8em}}
.muted{{color:#666}} .card{{border:1px solid #ddd;border-radius:10px;padding:12px 14px;margin:10px 0}}
.fix{{background:#eef4ff;border-left:4px solid #2f5bd3;padding:8px 12px;border-radius:0 8px 8px 0;margin:8px 0}}
.bar{{display:inline-block;width:120px;height:10px;background:#eee;border-radius:5px;vertical-align:middle;margin-left:6px}}
.bar span{{display:block;height:100%;background:#2f5bd3;border-radius:5px}}
table{{border-collapse:collapse;width:100%;font-size:.92rem}} td,th{{border:1px solid #ddd;padding:5px 8px;text-align:left;vertical-align:top}}
.sample{{background:#fff3d6;border:1px solid #e0b040;padding:8px 12px;border-radius:8px}}
details{{margin-top:6px}} @media print{{details{{display:none}}}}
</style></head><body>
<p class="muted">Practice interview report · {e(m.get('started', ''))} · {e(m.get('program', ''))}</p>
<h1>Your practice interview</h1>
<p class="muted">This is a rehearsal tool. It does not grade, rank or certify you, and no one else receives this report.</p>"""]
    if m.get("sample_rubric"):
        parts.append('<p class="sample">Scored with the <b>unapproved sample rubric</b> (development only).</p>')
    if m.get("ended_early"):
        parts.append('<p class="muted">You ended this session early. Only completed answers are shown.</p>')
    if "average_score" in s:
        parts.append(f"""<div class="card"><b>Average rubric score: {s['average_score']} / 5</b>{_bar(s['average_score'])}
<p><b>What to work on first:</b></p>""" + "".join(f'<div class="fix">{e(c)}</div>' for c in s["top_corrections"]) +
            f"""<p class="muted">Pace: {s.get('average_wpm') or '-'} words/min · Filler words: {s['total_fillers']} ·
Long pauses: {s['long_pauses']} · Eye contact: {s['eye_contact_pct'] if s['eye_contact_pct'] is not None else '-'}%</p></div>""")
    for i, a in enumerate(report["answers"], 1):
        q = a["question"].get(lang) or a["question"]["en"]
        parts.append(f"<h2>Question {i}</h2><p><b>{e(q)}</b></p>")
        if "score" not in a:
            parts.append(f'<p class="muted">Could not score this answer ({e(a.get("error", "no result"))}).</p>')
            continue
        sc, d = a["score"], a["delivery"]
        parts.append(f"<p>Rubric score: <b>{sc['score']} / 5</b>{_bar(sc['score'])}</p>")
        parts.append(f'<div class="fix">{e(sc["correction"]["text"])}</div>')
        parts.append("<table><tr><th>Criterion</th><th>Score</th><th>Closest example level</th></tr>" + "".join(
            f"<tr><td>{e(c['criterion'])}</td><td>{c['score']}</td><td>{c['matched_level']}</td></tr>"
            for c in sc["criteria"]) + "</table>")
        if d["tips"]:
            parts.append("<ul>" + "".join(f"<li>{e(t)}</li>" for t in d["tips"]) + "</ul>")
        parts.append(f"""<details><summary>Transcript and details</summary><p>{e(a['transcript']) or '<i>(no speech recognized)</i>'}</p>
<p class="muted">Pace {d['wpm'] or '-'} wpm · fillers {d['fillers']['total']} · long pauses {len(d['long_pauses'])} ·
started after {d['response_latency_s'] if d['response_latency_s'] is not None else '-'} s · ended by {e(a['end_reason'])}</p>
<p class="muted">Traceability: each criterion score comes from the approved example answer it matched most closely.</p>""" +
            "".join(f"<p class='muted'><b>{e(c['criterion'])}</b>, matched level {c['matched_level']}: “{e(c['matched_anchor'] or '')}”</p>"
                    for c in sc["criteria"]) + "</details>")
    parts.append("</body></html>")
    return "\n".join(parts)
