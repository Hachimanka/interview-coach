"""The generative interviewer (G2): a local LLM that writes its own questions and replies.

What it may say is limited three ways:
1. the competency slot (samples/blueprint.json) fixes WHAT each question must find out;
2. the prompt gives it only the session memory, what the student just said, and 1-2 knowledge passages;
3. guard.check_turn() checks the output; a failing sentence is replaced by the approved line.

All tasks share one fixed system prompt and the same worked examples, so llama.cpp keeps that prefix
cached and only the short variable part at the end is processed on each call (faster on the Pi).
The same prompt functions build the synthetic training data (training/interviewer), so a fine-tuned
model drops in by changing reply.llm_model.
"""
from __future__ import annotations

import re

SYSTEM = (
    "You are a friendly, professional interviewer running a PRACTICE job interview with a Filipino "
    "Computer Engineering (CpE) or Electronics Engineering (ECE) student who is applying for on-the-job training. "
    "Speak like a real person: warm, natural and brief.\n"
    "Rules for everything you say:\n"
    "- Use only what you are given: the student's words, the NOTES from earlier in this session, the GOAL, "
    "the PLAIN QUESTION and the BACKGROUND. Never invent details the student did not say.\n"
    "- Never praise, criticize, grade or judge the answer: never say great, good, excellent, nice, impressive, "
    "wrong, correct or similar. (Praise is added separately by the system.)\n"
    "- Do not teach or give the answer. Ask, then let the student talk.\n"
    "- Stay on the interview, the student's experience and CpE/ECE topics. Never ask about age, religion, "
    "politics, family, relationships, health, money or where they live.\n"
    "- You are the interviewer: speak to the student as \"you\". Never say we, our or my.\n"
    "- Use the student's NAME only if one is given, and at most once.\n"
    "- English only. Output only the words you say.\n"
    "Tasks:\n"
    "QUESTION: React to what the student just said in a few words, then ask ONE next question that finds out "
    "the GOAL. If a CALLBACK is given, connect the question to it. At most {max_words} words.\n"
    "FOLLOW_UP: Briefly restate one detail the student just said, then ask ONE follow-up question that finds "
    "out the GOAL. At most 30 words.\n"
    "FACTS: List up to 3 short facts about the student from what they just said, one per line starting with "
    "\"- \", using their own words. Write \"- none\" if there are none.\n"
    "COACH: After the interview, write 2 or 3 encouraging sentences to the student. Mention the STRENGTH, then "
    "the WORK ON point, then the DELIVERY tip if given. Here you may praise, but only the STRENGTH. No question."
)


def _name_line(name: str | None) -> str:
    return f"NAME: {name}" if name else "NAME: (do not use a name)"


def _said(text: str, n: int = 70) -> str:
    return " ".join(text.split()[:n])     # the start of a long answer carries most of it; keeps it fast


def question_prompt(*, goal: str, example: str, notes: str, just_said: str = "", callback: str | None = None,
                    name: str | None = None, background: list[str] | None = None,
                    already: list[str] | None = None) -> str:
    """The variable part of a QUESTION call. The INSTRUCTION line spells out the structure, because
    small models follow a concrete instruction at the end far better than rules at the top."""
    steps = []
    if just_said:
        steps.append("start with a few words that react to what the student just said")
    if callback:
        steps.append(f"then say that earlier they mentioned: {callback}")
        steps.append("then ask about the GOAL in connection with that")
    else:
        steps.append("then ask about the GOAL")
    if name:
        steps.append(f"use the name {name} once")
    if already:
        steps.append("do not bring up again: " + "; ".join(already))
    steps.append("do not copy the PLAIN QUESTION word for word")
    user = (f"TASK: QUESTION\n{_name_line(name)}\nNOTES:\n{notes}\n"
            f"STUDENT JUST SAID: \"{_said(just_said) or '(nothing yet: this is the first question)'}\"\n"
            f"CALLBACK: {callback or '(none)'}\nGOAL: {goal}\nPLAIN QUESTION: {example}")
    if background:
        user += "\nBACKGROUND: " + " ".join(background)
    return user + "\nINSTRUCTION: " + "; ".join(steps) + "."


def follow_up_prompt(*, goal: str, example: str, notes: str, just_said: str, name: str | None = None,
                     background: list[str] | None = None) -> str:
    user = (f"TASK: FOLLOW_UP\n{_name_line(name)}\nNOTES:\n{notes}\n"
            f"STUDENT JUST SAID: \"{_said(just_said)}\"\nGOAL: {goal}\nPLAIN QUESTION: {example}")
    if background:
        user += "\nBACKGROUND: " + " ".join(background)
    return user + ("\nINSTRUCTION: restate one detail the student just said, speaking to them as \"you\" "
                   "(never \"we\"); then ask about the GOAL; do not copy the PLAIN QUESTION word for word.")


def facts_prompt(said: str) -> str:
    return f"TASK: FACTS\nSTUDENT JUST SAID: \"{_said(said, 90)}\""


def coach_prompt(*, name: str | None, strength: str, work_on: str, delivery: str | None = None) -> str:
    user = f"TASK: COACH\n{_name_line(name)}\nSTRENGTH: {strength}\nWORK ON: {work_on}"
    return user + (f"\nDELIVERY: {delivery}" if delivery else "")


# Worked examples (few-shot), built with the same prompt functions as the live calls.
EXAMPLES = [
    (question_prompt(
        name="Leonard",
        notes="- Q1 (Self-introduction and motivation): fourth year Computer Engineering student; built an "
              "attendance system with an RFID reader for capstone",
        just_said="I am applying because I want to learn how real embedded products are made.",
        callback="built an attendance system with an RFID reader for capstone",
        goal="a real time they worked in a team and faced a disagreement: what happened, what they did, and the result",
        example="Tell me about a time you worked in a team and faced a disagreement. What did you do?"),
     "Thanks, Leonard, wanting to see how real products are made makes sense. Earlier you mentioned your RFID "
     "attendance system for capstone. Did your team ever disagree while building it, and how did you handle it?"),
    (question_prompt(
        notes="- Q2 (Teamwork and handling disagreement): two members wanted different sensors; tested both for one day",
        just_said="So we tested both sensors for a day and the ultrasonic sensor was more accurate, so everyone agreed.",
        goal="their step-by-step method for finding and fixing a bug in a program or circuit, and the tools they use",
        example="How do you approach debugging a program or circuit that is not working?",
        background=["Systematic debugging means reproducing the problem, isolating it by testing one part at a "
                    "time, fixing it, then retesting the whole system."]),
     "So testing both sensors side by side settled it. Let me ask about problem solving: when a program or "
     "circuit stops working, what steps do you take to find and fix the problem?"),
    (follow_up_prompt(
        notes="- (nothing yet)", just_said="We made a water monitor with Arduino and some sensors.",
        goal="why they chose the Arduino and how they used it",
        example="You mentioned the Arduino. Why did you choose it for that work?"),
     "So you made a water monitor with an Arduino. What made you pick the Arduino, and what did it do in the system?"),
    (facts_prompt("Good morning. I'm a third year ECE student. For our thesis we're making a solar charger, and I "
                  "handle the PCB layout in KiCad."),
     "- third year ECE student\n- thesis is a solar charger\n- handles the PCB layout in KiCad"),
    (coach_prompt(name="Leonard", strength="Technical depth: you explained the components and why you chose them",
                  work_on="Say exactly which part you built and what the project achieved, ideally with a number.",
                  delivery="Try to use fewer filler words like um."),
     "Leonard, you explained your components and why you chose them very clearly. Next time, say exactly which "
     "part you built and what it achieved, ideally with a number. Also try to use fewer filler words."),
]

_PREFIX = re.compile(r"^(interviewer|robot|assistant|you)\s*:\s*", re.I)


def build_messages(user: str, max_words: int = 40) -> list[dict]:
    """System prompt + worked examples + this call. Training data uses exactly the same messages."""
    msgs = [{"role": "system", "content": SYSTEM.format(max_words=max_words)}]
    for u, a in EXAMPLES:
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    return msgs + [{"role": "user", "content": user}]


def _clean(text: str) -> str:
    text = _PREFIX.sub("", text.strip()).strip().strip('"').strip()
    return " ".join(text.split())


class Interviewer:
    def __init__(self, llm, max_words: int = 40, warm: bool = True):
        self.llm = llm
        self.max_words = max_words
        self.last_seconds = 0.0
        if warm:     # caches the fixed prefix, so the first real question is fast
            self.facts("I am a Computer Engineering student.")

    def messages(self, user: str) -> list[dict]:
        return build_messages(user, self.max_words)

    def _ask(self, user: str, max_tokens: int, stop: list[str], temperature: float = 0.4) -> str:
        out = self.llm.chat(self.messages(user), max_tokens=max_tokens, temperature=temperature, top_p=0.9, stop=stop)
        self.last_seconds = self.llm.last_seconds
        return out

    def question(self, **kw) -> str:
        return _clean(self._ask(question_prompt(**kw), 90, ["\n"]))

    def follow_up(self, **kw) -> str:
        return _clean(self._ask(follow_up_prompt(**kw), 70, ["\n"]))

    def facts(self, said: str) -> list[str]:
        out = self._ask(facts_prompt(said), 70, ["\n\n", "TASK"], 0.2)
        return [line.strip() for line in out.splitlines() if line.strip().startswith(("-", "•", "*"))][:3]

    def coach(self, **kw) -> str:
        return _clean(self._ask(coach_prompt(**kw), 110, ["\n"], 0.4))
