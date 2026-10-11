"""G3: generate synthetic CpE/ECE practice interviews with a large open "teacher" model (Google Colab, GPU).

The teacher plays BOTH sides: a student persona (training/interviewer/dataset.py PERSONAS) and the
interviewer. Interviewer turns are written with the SAME prompts the robot uses (question_prompt,
follow_up_prompt, facts_prompt, coach_prompt), and only turns that pass the runtime guard are kept.
No real student data is used.

Colab (Runtime > T4 GPU or better):
    !git clone <your repo> && cd interview-coach
    !pip install -r requirements-common.txt transformers accelerate bitsandbytes
    !python -m training.interviewer.make_synthetic --dialogues 600 --out data/synthetic/interviewer.jsonl

Faculty should read a random sample (e.g. 50 records) before training (G3 review).
"""
from __future__ import annotations

import argparse
import random

from coach.interviewer.interviewer import coach_prompt, facts_prompt, follow_up_prompt, question_prompt
from coach.interviewer.knowledge import blueprint
from coach.interviewer.memory import SessionMemory
from coach.profile import load_profile
from coach.scoring.rubric import load_session_bank, pick_questions
from training.interviewer.dataset import PERSONAS, PROJECTS, make_record, write_jsonl

NAMES = ["Leonard", "Maria", "John Paul", "Angela", "Mark", "Kristine", "Jerome", "Patricia", "Carlo", "Bea"]


class Teacher:
    """A large instruction model through transformers (4-bit to fit a Colab GPU)."""

    def __init__(self, model: str, four_bit: bool = True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16) if four_bit else None
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, device_map="auto", quantization_config=q)

    def chat(self, messages: list[dict], max_new_tokens: int = 160, temperature: float = 0.7) -> str:
        ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        out = self.model.generate(ids, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
                                  top_p=0.9, pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def student_turn(teacher: Teacher, persona: str, profile: dict, question: str, history: list[str]) -> str:
    system = (f"You are role-playing a Filipino {profile['course']} student in a practice job interview for "
              f"on-the-job training. You are {PERSONAS[persona]}. Your main project: {profile['project']}. "
              "Answer in 1-5 spoken sentences, as you would say it out loud. Output only your answer.")
    msgs = [{"role": "system", "content": system}]
    for i in range(0, len(history) - 1, 2):
        msgs += [{"role": "user", "content": history[i]}, {"role": "assistant", "content": history[i + 1]}]
    msgs.append({"role": "user", "content": question})
    return teacher.chat(msgs, 180, 0.9)


def interviewer_turn(teacher: Teacher, user: str) -> str:
    from coach.interviewer.interviewer import build_messages

    return teacher.chat(build_messages(user), 120, 0.6).split("\n")[0].strip().strip('"')


def dialogue(teacher: Teacher, cfg, rng: random.Random) -> list[dict]:
    persona = rng.choice(list(PERSONAS))
    profile = {"course": rng.choice(["Computer Engineering", "Electronics Engineering"]), "project": rng.choice(PROJECTS)}
    name = rng.choice(NAMES + [None, None])
    bank = load_session_bank(cfg)
    slots = blueprint(cfg)
    memory = SessionMemory(name=name)
    records, history, just_said = [], [], ""
    for i, q in enumerate(pick_questions(bank, cfg.session.questions_per_session, True, rng), 1):
        slot = slots.get(q.id, {"goal": q.text["en"], "competency": q.bank, "callback": True})
        cands = memory.callback_candidates(i) if slot.get("callback", True) else []
        cb = rng.choice(cands).text if cands and rng.random() < 0.7 else None
        use_name = memory.may_use_name() and rng.random() < 0.6
        user = question_prompt(goal=slot["goal"], example=q.text["en"], notes=memory.notes(), just_said=just_said,
                               callback=cb, name=name if use_name else None)
        asked = interviewer_turn(teacher, user)
        records.append(make_record(user, asked, persona))
        memory.next_turn(bool(use_name and name and name in asked))
        if cb:
            memory.used.add(cb)
        answer = student_turn(teacher, persona, profile, asked, history)
        history += [asked, answer]
        memory.add_answer(i, slot["competency"], answer)
        facts_user = facts_prompt(answer)
        facts = teacher.chat([{"role": "user", "content": facts_user + "\nList up to 3 facts, one per line, "
                               "starting with '- ', using the student's own words."}], 80, 0.2)
        records.append(make_record(facts_user, facts, persona))
        memory.add_ai_facts(i, facts.splitlines())
        just_said = answer
        if persona in ("weak", "average", "nervous", "taglish") and rng.random() < 0.6:
            crit = rng.choice(q.criteria)
            fu_user = follow_up_prompt(goal=crit.description, example=(crit.follow_up or {}).get("en", "Could you tell me more?"),
                                       notes=memory.notes(), just_said=answer)
            fu = interviewer_turn(teacher, fu_user)
            records.append(make_record(fu_user, fu, persona))
            fu_answer = student_turn(teacher, persona, profile, fu, history)
            history += [fu, fu_answer]
            memory.add_answer(i, slot["competency"], fu_answer)
            just_said = ""
    crit = rng.choice([c for q in bank.questions for c in q.criteria])
    c_user = coach_prompt(name=name, strength=f"{crit.name}: {crit.description}", work_on=crit.correction,
                          delivery=rng.choice([None, "Try to use fewer filler words like um.", "Slow down a little."]))
    records.append(make_record(c_user, interviewer_turn(teacher, c_user), persona))
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--dialogues", type=int, default=600)
    ap.add_argument("--out", default="data/synthetic/interviewer.jsonl")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    cfg = load_profile("pc")
    rng = random.Random(args.seed)
    teacher = Teacher(args.teacher)
    records = []
    for d in range(args.dialogues):
        records += dialogue(teacher, cfg, rng)
        if (d + 1) % 20 == 0:
            print(f"{d + 1}/{args.dialogues} dialogues, {len(records)} records")
    kept, dropped = write_jsonl(records, args.out)
    print(f"kept {kept} records that pass the runtime guard, dropped {dropped} -> {args.out}")


if __name__ == "__main__":
    main()
