# Robotic Mock-Interview Coach (BROwsers)

Runtime and ML code for the thesis robot. **The plan and task tracker live in
`~/Downloads/Interview-Coach-ML-Plan.html`.** Its JSON block is the source of truth for what to do next.

**Technology stack, how the ML works, PC setup and Google Colab training: see [TECH_STACK.md](TECH_STACK.md),
or open [TECH_STACK.html](TECH_STACK.html) in a browser for the same content with an architecture diagram.**

## Approach
Pretrained models on the PC first → measure against the gates → train only what fails →
shrink and port to the Raspberry Pi 5 by switching `config/pc.yaml` → `config/pi5.yaml`.

## Layout
| Folder | What | Torch allowed? |
|---|---|---|
| `coach/` | Runtime package (runs on PC and Pi) | **No** |
| `config/` | `pc.yaml`, `pi5.yaml` profiles, `models.yaml` registry | n/a |
| `training/` | Training + export scripts (PC/Colab) | Yes |
| `eval/` | WER, kappa, gaze, filler, latency benchmarks | Yes |
| `offline/` | Pre-synthesize question audio, draft anchors | Yes |
| `data/` | Recordings, transcripts, ratings, rubric (**never committed**) | n/a |
| `models/` | Downloaded / exported weights (not committed) | n/a |
| `cache/` | Generated question WAVs, subtitles, lip-sync (not committed) | n/a |

Every stub names its task ID, e.g. `TODO(T1.2)`. Find them with `grep -rn "TODO(T" .`

## Setup (PC, Windows)
Use Python 3.11 to match Raspberry Pi OS Bookworm.
```
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements-pc.txt
pytest
pip freeze > requirements-lock-pc.txt
```
Training (`requirements-train.txt`) runs on Google Colab, because this PC has no NVIDIA GPU.

## Run it (PC)
```
.venv\Scripts\activate
python -m offline.download_models --profile pc --all-asr --all-llm   # once: VAD, Whisper, Piper, MediaPipe, embedder, local AI
python -m offline.synth_questions                          # once per rubric change: question audio EN + FIL
python -m coach.session.run                                # opens the face page; speak to the laptop mic
```
- No touch screen is needed: every button can be spoken, and each button shows what to say (see below).
- Keyboard backup: **Enter** = Done, **Esc** = Stop, **1-9** = pick a button, **E** / **F** = force English / Filipino.
- `--profile pi5` uses the light settings (needs the T4.2 ONNX embedder export for scoring).
- `--record` also saves answer audio, for pilot data collection (T0.5, with consent).
- Reports are saved in `data/sessions/<time>/report.html` only if the student chooses to keep them.

## Generative interviewer (interviewer.mode: generative, the default)
In English sessions the robot is a **real conversational interviewer**, not a list of fixed questions
(`coach/interviewer/`, `coach/session/generative.py`). Everything runs offline on the PC / Pi.
- **Asks your first name** after consent, confirms it ("Nice to meet you, Leonard. Did I get your name right?"),
  and uses it naturally, at most every 2 turns (`interviewer.name_every`).
- **Remembers the whole session** (`coach/interviewer/memory.py`): topics, feelings and short facts from every
  answer, so Q5 can say "Earlier you mentioned your ESP32 attendance system...". AI-written facts are kept only if
  the student really said those words. Memory is RAM only and is cleared when the session ends.
- **Writes its own questions** for each competency slot (`samples/blueprint.json`, one slot per rubric question):
  the slot fixes WHAT to find out, the AI chooses the wording, reacts to the last answer and connects to earlier
  ones. The answer is scored with that slot's rubric, so scoring stays valid and traceable.
- **Writes its own follow-ups** from the approved follow-up's goal, and looks up the CpE/ECE knowledge library
  (`samples/knowledge/cpe_ece.json`) to stay relevant on technical questions.
- **Praises only what the scorer found** ("That's a really good answer, Leonard. You explained the technology you used
  and why you chose it."): only when a criterion scored at least `interviewer.praise_min`, so praise never
  contradicts the report.
- **Writes a coaching summary** at the top of the report from the scores ("Your coach says").
- **Guard v2** (`coach/interviewer/guard.py`) checks every AI sentence: no judging, no "we/our", no private topics
  (age, religion, family, money...), no invented names, numbers or technical terms, and no false "you mentioned ..."
  claims. A failing or late sentence is replaced by the approved rubric question / follow-up. The report shows how
  each question was written. Misheard terms are fixed first ("ESB 32" -> "ESP32", `misheard` in the glossary).
- Thinking time is hidden: the next question is written while the student answers the follow-up; if it is still
  not ready, the robot says "Let me think about my next question" like a person would.
- Filipino sessions keep the approved pre-recorded lines (small models are weak in Filipino; live Filipino voice
  would need an ONNX export of the MMS voice). `interviewer.mode: scripted` turns the whole feature off.

Measure it: `python -m eval.interviewer_eval` (guard pass rate, personalization, latency; `--model` for the Pi
model or a fine-tuned one). Improve it: synthetic data + LoRA on Colab, see `training/interviewer/`.

## Voice commands and language
The student never has to touch the screen. Commands work in English and Filipino at every step
(`coach/session/voice.py` holds the phrase lists; `voice:` in `config/*.yaml` turns it off or tunes it).

| Step | Say (English) | Say (Filipino) |
|---|---|---|
| Start | "start", "hello" | "simulan", "kumusta" |
| Consent | "yes", "I agree" / "no" | "oo", "sige" / "hindi", "ayoko" |
| End of an answer | "that's my answer", "I'm done" | "iyon na po ang sagot ko", "tapos na po" |
| Stop the session | "stop the interview" | "itigil ang interview" |
| Keep the report | "keep it" / "delete it" | "itago" / "burahin" |

- **Courses: CpE and ECE only** (`session.programs`). The robot **never asks the student's course**:
  every session asks the common questions plus one CpE and one ECE technical question, rotated between
  sessions (`session.vary_questions`). The rubric files are `samples/rubric/common.json`, `cpe.json`, `ece.json`.
- There is no language screen and **one session = one language, never mixed**. The robot starts in
  English; the student's **first spoken reply** decides the language once ("simulan" / "oo" = Filipino,
  "start" / "yes" = English). From the first question to the end it is locked: answers (English, Filipino
  or Taglish) and end phrases never change it. Filipino is used only if every Filipino clip exists;
  otherwise the whole session stays English. Staff can still force it with **E** / **F**.
- The robot waits for the student. It starts only on "start" / tap (`session.start_on_face: false`),
  repeats a question on a choice screen if nobody answers (`voice.choice_reprompts`). An answer ends on
  "that's my answer" / Done, or after 5 s of silence once the student has spoken
  (`audio.end_of_answer_silence_s`). If the student never starts, "Take your time" shows at 5 s and the
  robot moves on at 15 s (`audio.no_answer_timeout_s`). It ignores the mic for 0.4 s after it speaks, so
  its own echo is never taken as the student's answer (`audio.echo_guard_ms`).
- The robot **replies to what the student said** (`coach/reply/`):
  1. It mirrors the student's laughter once per session ("Haha!" + amused face; YAMNet detects laughter;
     it never laughs first), otherwise says a neutral acknowledgment ("I see.").
  2. If the student said they're nervous or stuck ("I'm nervous", "kinakabahan", "hindi ko alam") it reassures them.
  3. It picks **one** approved reply: a follow-up about a topic they mentioned (from `samples/glossary.json`,
     e.g. "You mentioned the Arduino. Why did you choose it?"), else the rubric follow-up / "tell me more",
     else a topic acknowledgment. At most one follow-up per question; its answer is scored with the main one.
  4. In English sessions a **local AI** (Qwen2.5 via llama.cpp: 1.5B on the PC, 0.5B on the Pi) rephrases that
     approved line to refer to the student's words ("You mentioned using an Arduino for a water monitor. Why did
     you choose it?"). `coach/reply/guard.py` checks every AI reply (no judging words, keeps the question, no new
     facts or numbers, keeps the topic, never speaks as the student). If a check fails or the AI takes longer than
     `reply.llm_timeout_s`, the approved line is spoken as-is. Filipino sessions use approved clips only.
  The report records what the robot said, whether the AI wrote it, and why. Settings: `session.follow_ups`,
  `follow_up_min_words`, `follow_up_below`, and the `reply:` section. After changing follow-ups, robot lines or
  templates, run `python -m offline.synth_questions`. The laughter threshold must be calibrated with real recordings.
- Spoken commands use a small fast Whisper (`voice.command_model: base`, int8): the robot proceeds about
  2 s after "yes" / "that's my answer" on the PC (it was ~8 s with the answer model). Answers are still
  transcribed by the accurate answer model for scoring.
- Every spoken line is followed by 0.35 s of silence so the sound card plays the last word before the
  stream closes (it used to cut off the end of sentences).
- Filipino speech exists only as pre-made clips. Until `python -m offline.synth_questions` has been run,
  the robot understands Filipino but answers in English.
- The robot does not listen while it is speaking (the mic would hear its own voice). Use **Esc** to stop it mid-sentence.
- Consent is never assumed: unclear speech is asked again, and no answer within `choice_timeout_s` counts as No.

## Test it
```
pytest                                         # modules + full simulated sessions (tapped and spoken)
python -m eval.simulate_session --sessions 3   # gate G1 with a simulated student (Piper voice)
python -m eval.simulate_session --profile pi5 --scoring-backend sentence_transformers
python -m eval.simulate_session --sessions 1 --generative   # AI interviewer: says a name, prints each question
python -m eval.interviewer_eval                # G7: guard pass rate, personalization, latency of the local AI
```

## Rules
- Never train or tune on `data/splits/test_locked/`.
- Recordings never leave the device or PC.
- The local AI never decides the rubric: questions stay inside faculty-approved competency slots and are scored
  with the approved criteria; every AI sentence passes the guard or is replaced by an approved line.
- Synthetic training data only (no student recordings) unless consent and ethics approval cover it (RA 10173).
