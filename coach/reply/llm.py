"""Local AI that rephrases an approved reply so it refers to what the student said (llama.cpp, offline).

It never chooses WHAT to ask: the approved line (rubric follow-up or template) fixes the meaning, and
guard.py rejects any output that drifts. English only.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

SYSTEM = (
    "You are a friendly, professional job interviewer in a PRACTICE interview with an engineering student. "
    "You get what the student just said and an APPROVED reply. Rewrite the approved reply so it sounds "
    "natural and briefly refers to what the student said.\n"
    "Rules:\n"
    "- Begin by briefly restating ONE specific detail the student said, using their own words.\n"
    "- Then keep the meaning of the approved reply. If it asks a question, ask that same question.\n"
    "- One or two short sentences, at most {max_words} words in total.\n"
    "- Do not praise, criticize, grade or judge the answer. Never say great, good, excellent, nice, "
    "impressive, wrong, correct or similar.\n"
    "- Do not add facts the student did not say. Do not give advice.\n"
    "- English only. Output only the rewritten reply, nothing else."
)


# Worked examples (few-shot): small models follow examples far better than rules alone.
EXAMPLES = [
    ('I built a line-following robot with an Arduino and two IR sensors for our robotics class.',
     'You mentioned the Arduino. Why did you choose it for that work?', 'Arduino',
     'So you built a line-following robot with an Arduino and IR sensors. Why did you choose the Arduino for it?'),
    ('Um, we had a disagreement about the design, but we talked about it and agreed.',
     'How did that situation end? What was the result?', None,
     'You said your team disagreed about the design and then talked it through. How did it end, and what was the result?'),
    ('I designed a small board in KiCad for a power supply in our lab.',
     'Thank you for sharing that about the KiCad.', 'KiCad',
     'Thank you for sharing how you designed a power supply board in KiCad.'),
]


def _user(said: str, approved: str, topic: str | None) -> str:
    text = f'Student said: "{said}"\nApproved reply: "{approved}"\n'
    if topic:
        text += f'Topic to mention by this exact name: "{topic}"\n'
    return text + "Rewritten reply:"


class LocalLLM:
    """One llama.cpp model, loaded once and shared by the rephraser and the generative interviewer.
    llama.cpp is not thread-safe, so calls are serialized. It keeps the longest matching prompt prefix
    in its KV cache, so prompts that start with the same fixed text (system + examples) are fast."""

    def __init__(self, model_path: str | Path, threads: int = 4, n_ctx: int = 4096):
        from llama_cpp import Llama   # imported here so the rest of coach/ works without it

        from coach.vision.face import quiet_native_logs

        with quiet_native_logs():     # llama.cpp prints model-loading details to stderr
            self.llm = Llama(model_path=str(model_path), n_ctx=n_ctx, n_threads=threads, verbose=False)
        self.path = Path(model_path)
        self.lock = threading.Lock()
        self.last_seconds = 0.0

    def chat(self, messages: list[dict], max_tokens: int = 80, temperature: float = 0.4, top_p: float = 0.9,
             stop: list[str] | None = None) -> str:
        with self.lock:
            t0 = time.monotonic()
            out = self.llm.create_chat_completion(messages=messages, max_tokens=max_tokens, temperature=temperature,
                                                  top_p=top_p, stop=stop or [])
            self.last_seconds = time.monotonic() - t0
        return out["choices"][0]["message"]["content"].strip()


class Rephraser:
    def __init__(self, model_path: str | Path | None = None, threads: int = 4, max_words: int = 25,
                 n_ctx: int = 1024, llm: LocalLLM | None = None):
        self.llm = llm or LocalLLM(model_path, threads, n_ctx)
        self.max_words = max_words
        self.last_seconds = 0.0
        # Warm-up: the first generation is much slower (it missed the 5 s limit in a simulated interview).
        # Running the few-shot prompt once now caches it, so the first real reply is fast.
        self.rephrase("Why did you choose it?", "I used an Arduino.", "Arduino")

    def rephrase(self, approved: str, said: str, topic: str | None = None) -> str:
        said = " ".join(said.split()[:90])     # the end of a long answer adds little; keeps it fast
        messages = [{"role": "system", "content": SYSTEM.format(max_words=self.max_words)}]
        for ex_said, ex_approved, ex_topic, ex_out in EXAMPLES:
            messages += [{"role": "user", "content": _user(ex_said, ex_approved, ex_topic)},
                         {"role": "assistant", "content": ex_out}]
        messages.append({"role": "user", "content": _user(said, approved, topic)})
        out = self.llm.chat(messages, max_tokens=60, temperature=0.4, top_p=0.9, stop=["\n"])
        self.last_seconds = self.llm.last_seconds
        return out.strip('"').strip()
