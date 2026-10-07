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
- On the face page: tap/click the buttons. **Enter** = Done, **Esc** = Stop.
- `--profile pi5` uses the light settings (needs the T4.2 ONNX embedder export for scoring).
- `--record` also saves answer audio, for pilot data collection (T0.5, with consent).
- Reports are saved in `data/sessions/<time>/report.html` only if the student chooses to keep them.

## Test it
```
pytest                                         # 35 tests: modules + full simulated sessions
python -m eval.simulate_session --sessions 3   # gate G1 with a simulated student (Piper voice)
python -m eval.simulate_session --profile pi5 --scoring-backend sentence_transformers
```

## Rules
- Never train or tune on `data/splits/test_locked/`.
- Recordings never leave the device or PC.
- No live LLM at runtime; rephrasings are generated offline and approved by faculty.
