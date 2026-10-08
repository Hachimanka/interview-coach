# Robotic Mock-Interview Coach (BROwsers)

Runtime and ML code for the thesis robot. **The plan and task tracker live in
`~/Downloads/Interview-Coach-ML-Plan.html`.** Its JSON block is the source of truth for what to do next.

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
python -m offline.download_models --profile pc --all-asr   # once: VAD, Whisper, Piper, MediaPipe, embedder
python -m offline.synth_questions                          # once per rubric change: question audio EN + FIL
python -m coach.session.run                                # opens the face page; speak to the laptop mic
```
- No touch screen is needed: every button can be spoken, and each button shows what to say (see below).
- Keyboard backup: **Enter** = Done, **Esc** = Stop, **1-9** = pick a button, **E** / **F** = force English / Filipino.
- `--profile pi5` uses the light settings (needs the T4.2 ONNX embedder export for scoring).
- `--record` also saves answer audio, for pilot data collection (T0.5, with consent).
- Reports are saved in `data/sessions/<time>/report.html` only if the student chooses to keep them.

## Voice commands and language
The student never has to touch the screen. Commands work in English and Filipino at every step
(`coach/session/voice.py` holds the phrase lists; `voice:` in `config/*.yaml` turns it off or tunes it).

| Step | Say (English) | Say (Filipino) |
|---|---|---|
| Start | "start", "hello" | "simulan", "kumusta" |
| Consent | "yes", "I agree" / "no" | "oo", "sige" / "hindi", "ayoko" |
| Program | "computer", "electrical", "electronics", or "one / two / three" | same, or "isa / dalawa / tatlo" |
| End of an answer | "that's my answer", "I'm done" | "iyon na po ang sagot ko", "tapos na po" |
| Stop the session | "stop the interview" | "itigil ang interview" |
| Keep the report | "keep it" / "delete it" | "itago" / "burahin" |

- There is no language screen. The robot starts in both languages, takes the language from the first
  command ("simulan" = Filipino), and then asks each question in the language of the previous answer.
  Taglish answers are understood; the robot replies in whichever language dominated.
- Filipino speech exists only as pre-made clips. Until `python -m offline.synth_questions` has been run,
  the robot understands Filipino but answers in English.
- The robot does not listen while it is speaking (the mic would hear its own voice). Use **Esc** to stop it mid-sentence.
- Consent is never assumed: unclear speech is asked again, and no answer within `choice_timeout_s` counts as No.

## Test it
```
pytest                                         # modules + full simulated sessions (tapped and spoken)
python -m eval.simulate_session --sessions 3   # gate G1 with a simulated student (Piper voice)
python -m eval.simulate_session --profile pi5 --scoring-backend sentence_transformers
```

## Rules
- Never train or tune on `data/splits/test_locked/`.
- Recordings never leave the device or PC.
- No live LLM at runtime; rephrasings are generated offline and approved by faculty.
