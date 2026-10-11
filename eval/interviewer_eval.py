"""G7: evaluate the generative interviewer on text interviews (no audio, fast), PC or Pi.

For each scripted student (strong, short, nervous, Taglish) it runs a 5-question interview through the
same prompts, memory and guard as the robot, and reports:
- guard pass rate per task (a failed sentence falls back to the approved line at runtime),
- personalization: questions that use a callback or the name, and callbacks grounded in what was said,
- latency p50 / p90 per call (the robot's thinking time).

    python -m eval.interviewer_eval                          # PC profile model
    python -m eval.interviewer_eval --model qwen2.5-0.5b-instruct-q4 --threads 3     # Pi-like
    python -m eval.interviewer_eval --model interviewer-1.5b-q4                      # after fine-tuning (G4)

Human-likeness itself is rated by people (students and faculty, Likert scale) on recorded sessions.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from coach.interviewer.guard import check_turn  # noqa: E402
from coach.interviewer.knowledge import slot  # noqa: E402
from coach.interviewer.memory import SessionMemory  # noqa: E402
from coach.profile import load_profile, make_embedder, make_knowledge, models_dir, resolve  # noqa: E402
from coach.scoring.rubric import load_session_bank, pick_questions  # noqa: E402

STUDENTS = {
    "strong": ["Good morning. I am a fourth year Computer Engineering student. For our capstone I built an RFID "
               "attendance system with an ESP32 and a web dashboard, and I want to learn how real products are made.",
               "In our microprocessors project two members wanted different sensors. I suggested testing both for one "
               "day. The ultrasonic sensor was more accurate, so we used it and everyone agreed.",
               "I used to underestimate debugging time. Now I add extra time to every estimate and track my hours.",
               "I wrote the ESP32 firmware that reads the RFID cards and sends them over WiFi to a MySQL database. "
               "My role was the firmware and the database, and it is now used in our department.",
               "I would add an RC low-pass filter before the ADC, keep the wires short, and average readings in code."],
    "short": ["I am an ECE student.", "We had a disagreement once.", "I am shy.", "We made a project, it was okay.",
              "Maybe use a filter."],
    "nervous": ["Um, sorry, I'm a bit nervous. I'm a third year ECE student and, uh, I like circuits.",
                "Uh, in our thesis we, um, argued about the PCB layout but we talked about it.",
                "I think my weakness is public speaking. I'm practicing by reporting in class.",
                "We made a solar powered weather station. I did the, um, soldering and testing.",
                "Um, I would use a capacitor and maybe a moving average filter."],
    "taglish": ["Hello po, I'm a fourth year CpE student. Yung capstone namin is a parking detector with ultrasonic sensors.",
                "Sa group project namin, nag-disagree kami sa design, so we voted and tested both.",
                "My weakness is time management, kaya gumagamit na ako ng planner.",
                "Ginawa ko yung Arduino code for the parking detector at yung LCD display.",
                "I would use shielded wires and a low-pass filter po."],
}


def run(cfg, interviewer, knowledge, name: str | None, answers: list[str]) -> dict:
    bank = load_session_bank(cfg)
    qs = pick_questions(bank, 5, vary=False)
    mem = SessionMemory(name=name)
    out = {"question": [], "follow_up": [], "facts": [], "times": [], "personal": 0, "grounded_callbacks": 0,
           "callbacks": 0, "turns": []}
    just = ""
    for i, (q, said) in enumerate(zip(qs, answers), 1):
        s = slot(cfg, q)
        cands = mem.callback_candidates(i) if s.get("callback", True) else []
        cb = None
        if cands:
            k, _ = knowledge.best_match(s["goal"], [f.text for f in cands]) if knowledge else (0, 0)
            cb = cands[k].text
        nm = name if mem.may_use_name() else None
        kw = dict(goal=s["goal"], example=q.text["en"], notes=mem.notes(), just_said=just, callback=cb, name=nm,
                  already=sorted(mem.used)[:4])
        t0 = time.monotonic()
        text = interviewer.question(**kw)
        out["times"].append(time.monotonic() - t0)
        grounding = mem.grounding() + " " + just
        ok, why = check_turn(text, "question", " ".join([kw["notes"], just, s["goal"], q.text["en"], cb or "",
                                                          grounding]), nm, memory=grounding)
        out["question"].append(ok)
        out["turns"].append({"q": i, "task": "question", "ok": ok, "check": why, "text": text})
        if ok:
            personal = bool(cb and any(w in text.lower() for w in cb.lower().split() if len(w) > 4)) or bool(nm and nm in text)
            out["personal"] += personal
            if cb:
                out["callbacks"] += 1
                mem.used.add(cb)
            mem.next_turn(bool(nm and nm in text))
        mem.add_answer(i, s["competency"], said)
        t0 = time.monotonic()
        facts = interviewer.facts(said)
        out["times"].append(time.monotonic() - t0)
        kept = mem.add_ai_facts(i, facts)
        out["facts"].append((len(kept), len(facts)))
        crit = q.criteria[0]
        t0 = time.monotonic()
        fu = interviewer.follow_up(goal=crit.description, example=(crit.follow_up or {}).get("en", "Could you tell me more?"),
                                   notes=mem.notes(), just_said=said)
        out["times"].append(time.monotonic() - t0)
        ok, why = check_turn(fu, "follow_up", " ".join([mem.notes(), said, crit.description, mem.grounding()]),
                             None, memory=said + " " + mem.grounding())
        out["follow_up"].append(ok)
        out["turns"].append({"q": i, "task": "follow_up", "ok": ok, "check": why, "text": fu})
        just = said
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="pc")
    ap.add_argument("--model", help="registry name in config/models.yaml (default: the profile's reply.llm_model)")
    ap.add_argument("--threads", type=int)
    args = ap.parse_args()
    cfg = load_profile(args.profile)
    name = args.model or cfg.reply.llm_model
    from coach.interviewer.interviewer import Interviewer
    from coach.reply.llm import LocalLLM

    path = models_dir(cfg) / "llm" / cfg.registry["llm"][name]["file"]
    llm = LocalLLM(path, threads=args.threads or cfg.reply.get("llm_threads", 4), n_ctx=cfg.reply.get("llm_ctx", 4096))
    iv = Interviewer(llm)
    kb = make_knowledge(cfg, make_embedder(cfg))
    results = {}
    for student, answers in STUDENTS.items():
        r = run(cfg, iv, kb, "Leonard" if student != "short" else None, answers)
        results[student] = r
        print(f"{student:8s} questions ok {sum(r['question'])}/5  follow-ups ok {sum(r['follow_up'])}/5  "
              f"personalized {r['personal']}  facts kept {sum(k for k, _ in r['facts'])}/{sum(n for _, n in r['facts'])}")
    times = [t for r in results.values() for t in r["times"]]
    q_ok = [x for r in results.values() for x in r["question"]]
    f_ok = [x for r in results.values() for x in r["follow_up"]]
    summary = {"model": name, "question_pass": round(sum(q_ok) / len(q_ok), 2),
               "follow_up_pass": round(sum(f_ok) / len(f_ok), 2),
               "latency_p50_s": round(statistics.median(times), 2),
               "latency_p90_s": round(sorted(times)[int(0.9 * (len(times) - 1))], 2)}
    print(json.dumps(summary, indent=2))
    out = resolve(cfg, "eval/results") / f"interviewer-{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "students": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"details: {out}")


if __name__ == "__main__":
    main()
