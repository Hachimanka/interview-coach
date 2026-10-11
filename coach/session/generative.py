"""Generative interview mode (G2-G6), mixed into Session.

English sessions only (Filipino sessions keep the approved pre-recorded lines). The flow stays the same
as the scripted mode, but:
- the robot asks the student's first name (and confirms it), and remembers everything they say;
- each question is written by the local AI for its competency slot, personalized with what the student
  said earlier ("Earlier you mentioned your ESP32 attendance system..."), and checked by the guard;
  if the check fails or the AI is late, the rubric question is asked as written;
- follow-ups are written by the AI from the approved follow-up's goal (fallback: the approved line);
- a strong answer gets specific praise, only for the criterion the scorer rated high;
- after the session, the AI writes a short coaching summary from the scores (fallback: a template).

Slow AI work is hidden: the next question is written while the student is still answering the
follow-up (or while the robot acknowledges the answer), so the student rarely waits for it.
"""
from __future__ import annotations

import dataclasses
import re
from concurrent.futures import TimeoutError as FuturesTimeout

from coach.interviewer.guard import check_turn
from coach.interviewer.knowledge import slot as slot_of
from coach.interviewer.memory import extract_name
from coach.reply.topics import canonicalize, glossary, templates
from coach.session.lines import GEN_LINES


class GenerativeMixin:
    # ---------- state ----------
    @property
    def generative(self) -> bool:
        return (self.interviewer is not None and self.icfg.get("mode", "scripted") == "generative"
                and self._l == "en")

    def _say_text(self, text: str, key: str | None = None, keep: bool = False) -> None:
        speech = self.speech.question(key, "en", text) if key else self.speech.live(text, "en")
        self.face.say(speech, keep_subtitle=text if keep else None, stop_check=self._stop_pressed)

    def _gen_line(self, key: str) -> None:
        self._say_text(GEN_LINES[key], key=f"gen.{key}")

    # ---------- the student's name ----------
    def _ask_name(self) -> None:
        """Ask for a first name and confirm it once. A wrong or missing name is simply not used."""
        if not self.icfg.get("ask_name", True) or not self.listener:
            return
        for attempt in range(2):
            self._gen_line("ask_name" if attempt == 0 else "name_retry")
            heard = self._hear_utterance(self.icfg.get("name_timeout_s", 12))
            name = extract_name(heard) if heard else None
            self.meta.setdefault("name_heard", []).append(heard or "")
            if not name:
                continue
            self.face.set_state("talking")
            self._say_text(f"Nice to meet you, {name}. Did I get your name right?")
            if self._choose("consent", ["yes", "no"], "name_title") == "yes":
                self.memory.name = name
                self.memory.last_name_turn = self.memory.turn     # just used it
                self.meta["name_confirmed"] = True
                return
        self.meta["name_confirmed"] = False
        self._gen_line("name_skip")

    def _hear_utterance(self, timeout: float) -> str | None:
        """Raw text of the next thing the student says (fast command model), or None."""
        import time

        from coach.session.state_machine import SessionStopped

        self.face.panel("none")
        self.listener.open()
        self.face.listen("name")
        end = time.monotonic() + timeout
        try:
            while time.monotonic() < end:
                ev = self.face.server.next_event(timeout=0)
                if ev and ev["name"] == "stop":
                    raise SessionStopped()
                heard = self.listener.poll(["stop"])
                if heard and heard.command == "stop":
                    raise SessionStopped()
                if heard and heard.text.strip():
                    return heard.text
            return None
        finally:
            self.face.listen("none")

    # ---------- questions ----------
    def _answer_text(self, ans) -> str:
        """Best text available right now: the accurate transcript if it is complete, else the quick one."""
        if ans.segments and len(ans.segments) == ans.queued:
            text = " ".join(t.text for _, t in ans.segments if t.text.strip())
        else:
            text = (ans.probe or {}).get("text", "") or " ".join(t.text for _, t in ans.segments)
        return canonicalize(text)

    def _remember(self, i: int, q, ans) -> str:
        """Rule facts (topics, feelings) now. Returns the text, for _remember_ai()."""
        text = self._answer_text(ans)
        if ans.followup:
            text = (text + " " + self._answer_text(ans.followup)).strip()
        if text:
            self.memory.add_answer(i, slot_of(self.cfg, q)["competency"], text)
        return text

    def _remember_ai(self, i: int, text: str) -> None:
        """AI facts in the background, queued AFTER the next question (the student is waiting for that;
        facts are for callbacks in later questions). Every AI job runs in order on one thread."""
        if text and self.icfg.get("memory_llm", True):
            def job():
                try:
                    self.memory.add_ai_facts(i, self.interviewer.facts(text))
                except Exception as e:   # memory is a nice-to-have; never break the interview
                    print(f"[coach] memory facts failed: {e!r}")
            self._ai().submit(job)

    def _pick_callback(self, i: int, slot: dict):
        if not (self.icfg.get("callbacks", True) and slot.get("callback", True)):
            return None
        facts = self.memory.callback_candidates(i)
        if not facts:
            return None
        if self.knowledge is None:
            return facts[0]
        k, sim = self.knowledge.best_match(slot["goal"], [f.text for f in facts])
        return facts[k] if k >= 0 else None

    def _technical(self, q) -> bool:
        return q.bank != "common"

    def _prefetch_question(self, i: int, q, just_said: str = ""):
        """Start writing question i now (in the AI thread). Returns a future of a result dict."""
        def job() -> dict:
            slot = slot_of(self.cfg, q)
            cb = self._pick_callback(i, slot)
            name = self.memory.name if self.memory.may_use_name() else None
            bg = []
            if self.knowledge is not None and self._technical(q):
                query = (just_said + " " + slot["goal"]).strip()
                bg = [p["text"] for p in self.knowledge.search(query, k=1, min_sim=self.icfg.get("kb_min_sim", 0.80))]
            kw = dict(goal=slot["goal"], example=q.text["en"], notes=self.memory.notes(), just_said=just_said,
                      callback=cb.text if cb else None, name=name, background=bg,
                      already=sorted(self.memory.used)[:4])
            info = {"slot": slot["competency"], "goal": slot["goal"], "plain_question": q.text["en"],
                    "callback": kw["callback"], "knowledge": bg}
            try:
                out = self.interviewer.question(**kw)
            except Exception as e:
                return dict(info, ok=False, check=f"ai error: {e!r}", text=None)
            grounding = self.memory.grounding() + " " + just_said
            ctx = " ".join([kw["notes"], just_said, slot["goal"], q.text["en"], kw["callback"] or "", *bg, grounding])
            ok, why = check_turn(out, "question", ctx, name, memory=grounding)
            if ok and self.knowledge is not None:
                sim = self.knowledge.similarity(out, q.text["en"])
                info["similarity_to_plain"] = round(sim, 3)
                if sim < self.icfg.get("on_slot_min_sim", 0.80):
                    ok, why = False, f"drifted from the competency (similarity {sim:.2f})"
            return dict(info, ok=ok, check=why, ai_text=out, text=out if ok else None,
                        used_name=bool(ok and name and name in out), seconds=round(self.interviewer.last_seconds, 2))
        return self._ai().submit(job)

    def _generated_question(self, i: int, q):
        """(question to score with, speech, text, info). Waits for the prefetched question; says a short
        "let me think" line if it is not ready; asks the rubric question as written if it fails or is late."""
        fut, self._next_q = (self._next_q or self._prefetch_question(i, q)), None
        res = None
        try:
            res = fut.result(timeout=0.6)
        except FuturesTimeout:
            self._think_n = getattr(self, "_think_n", 0) + 1
            self.face.set_state("thinking")
            self._gen_line(f"think_{(self._think_n - 1) % 3 + 1}")
            try:
                res = fut.result(timeout=self.icfg.get("question_timeout_s", 8))
            except FuturesTimeout:
                res = {"ok": False, "check": "ai too slow", "text": None}
        except Exception as e:
            res = {"ok": False, "check": f"ai error: {e!r}", "text": None}
        if res.get("ok"):
            text = res["text"]
            speech = self.speech.live(text, "en")
            if i > 1:     # "Good morning" belongs to the first question only
                text = re.sub(r"^(good (morning|afternoon|evening)|hello|hi)\b[,!.]?\s*", "", text, flags=re.I)
                text = text[:1].upper() + text[1:]
            if res.get("callback"):
                self.memory.used.add(res["callback"])
            self.memory.next_turn(res.get("used_name", False))
            source = "ai"
        else:
            if i > 1:
                self._line("next")
            speech, text, _ = self._clip(q.id, q.text, "en")
            source = "rubric"
        print(f"[coach] Q{i} ({source}{'' if res.get('ok') else ', ' + str(res.get('check'))}): {text}")
        info = {k: v for k, v in res.items() if k not in ("text",)}
        info["source"] = source
        asked = dataclasses.replace(q, text=dict(q.text, en=text))   # same id and criteria: same rubric
        return asked, speech, text, info

    # ---------- replies ----------
    def _follow_up_goal(self, plan: dict, q) -> str:
        if plan["kind"] == "topic_follow_up":
            return f"why they chose the {plan['topic'].name} and how they used it in that work"
        if plan["kind"] == "rubric_follow_up" and plan.get("key"):
            crit_id = plan["key"].rsplit(".", 1)[-1]
            crit = next((c for c in q.criteria if c.id == crit_id), None)
            if crit:
                return crit.description[0].lower() + crit.description[1:].rstrip(".")
        return "a specific example or more detail about what they just said"

    def _generative_follow_up(self, plan: dict, q, ans):
        """(text, speech, source, note): the AI writes the follow-up from the approved one's goal."""
        said = canonicalize((ans.probe or {}).get("text", ""))
        name = self.memory.name if self.memory.may_use_name() else None
        goal = self._follow_up_goal(plan, q)
        bg = []
        if self.knowledge is not None and self._technical(q) and said:
            bg = [p["text"] for p in self.knowledge.search(said, k=1, min_sim=self.icfg.get("kb_min_sim", 0.80))]
        kw = dict(goal=goal, example=plan["approved"], notes=self.memory.notes(), just_said=said, name=name,
                  background=bg)
        fut = self._ai().submit(self.interviewer.follow_up, **kw)
        try:
            out = fut.result(timeout=self.icfg.get("follow_up_timeout_s", 6))
            grounding = said + " " + self.memory.grounding()
            ctx = " ".join([kw["notes"], said, goal, plan["approved"], *bg, grounding])
            ok, note = check_turn(out, "follow_up", ctx, name, memory=grounding)
            if ok:
                self.memory.next_turn(bool(name and name in out))
                return out, self.speech.live(out, "en"), "ai", "ok"
            note = f"rejected: {note} ({out})"
        except FuturesTimeout:
            note = "ai too slow"
        except Exception as e:
            note = f"ai error: {e!r}"
        text, speech, source, _ = self._realize(plan, ans)    # the approved line (rephraser is off here)
        return text, speech, source, note

    def _maybe_praise(self, ans) -> dict | None:
        """Specific praise, only for what the scorer rated high (so it never contradicts the report)."""
        if not self.icfg.get("praise", True) or not ans.probe or ans.probe["score"].get("too_short"):
            return None
        crits = ans.probe["score"].get("criteria") or []
        if not crits:
            return None
        best = max(crits, key=lambda c: c["expected"])
        if best["expected"] < self.icfg.get("praise_min", 4.0):
            return None
        t = templates()
        what = t["praise"].get(best["criterion_id"]) or t["praise"]["default"]
        name = self.memory.name if self.memory.may_use_name() else None
        self._praise_n = getattr(self, "_praise_n", 0) + 1
        tmpl = t["praise_templates"][(self._praise_n - 1) % len(t["praise_templates"])]
        text = tmpl.format(name=f", {name}" if name else "", what=what)
        self.face.say(self.speech.live(text, "en"), stop_check=self._stop_pressed)
        self.memory.next_turn(bool(name))
        return {"criterion": best["criterion"], "probe_score": best["expected"], "said": text}

    # ---------- closing and coaching ----------
    def _closing_text(self) -> str:
        name = self.memory.name
        text = f"That was the last question. Thank you{', ' + name if name else ''}!"
        cats = {t["name"]: t["category"] for t in glossary()}
        topics = [x for x in self.memory.topics() if cats.get(x) in ("hardware", "activity", "software", "tool")]
        if topics:
            t = topics[0]
            text += f" I enjoyed hearing about {'your' if cats[t] == 'activity' else 'the'} {t}."
        return text

    def _coach_summary(self, answers) -> dict | None:
        """Short coaching paragraph from the scores (AI-written, checked; template fallback)."""
        scored = [a.result for a in answers if a.result and "score" in a.result]
        if not scored:
            return None
        crits = [c for r in scored for c in r["score"]["criteria"]]
        best = max(crits, key=lambda c: c["expected"])
        weakest = min(scored, key=lambda r: r["score"]["score"])
        work_on = weakest["score"]["correction"]["text"]
        tips = [t for r in scored for t in r["delivery"]["tips"]]
        delivery = max(set(tips), key=tips.count) if tips else None
        praise = templates()["praise"].get(best["criterion_id"]) or templates()["praise"]["default"]
        strength = f"{best['criterion']}: {praise[0].lower() + praise[1:]}"
        name = self.memory.name
        fallback = (f"{name + ', y' if name else 'Y'}our strongest area was {best['criterion'].lower()}. "
                    f"{praise} To improve: {work_on}" + (f" Also: {delivery}" if delivery else ""))
        out = {"strength": best["criterion"], "work_on": work_on, "delivery": delivery,
               "text": fallback, "source": "template"}
        if self.interviewer is None or not self.icfg.get("coach_summary", True):
            return out
        fut = self._ai().submit(self.interviewer.coach, name=name, strength=strength, work_on=work_on,
                                delivery=delivery)
        try:
            text = fut.result(timeout=self.icfg.get("coach_timeout_s", 12))
            ctx = " ".join([name or "", strength, work_on, delivery or "", best["criterion"]])
            ok, note = check_turn(text, "coach", ctx, name, allow_praise=True)
            out["check"] = note
            if ok:
                out.update(text=text, source="ai")
        except FuturesTimeout:
            out["check"] = "ai too slow"
        except Exception as e:
            out["check"] = f"ai error: {e!r}"
        return out
