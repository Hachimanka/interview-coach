# Tech Stack: Robotic Mock-Interview Coach

How the project is built: every technology in the stack, what it does, how the ML works, how the PC is
set up, and how to use Google Colab for training. Versions are the ones pinned in
`requirements-lock-pc.txt` (Python 3.11.9, Windows PC, CPU only, no GPU).

> Status (2026-10): Stage 1 is done, plus the **generative interviewer** (G1, G2, G5, G6, G7 on the PC).
> Everything runs with **pretrained models only**; no model has been trained yet. The interviewer's
> fine-tuning scripts (G3/G4) are ready for Colab. Other training waits for real recordings (Stage 0).

---

## 1. The big picture

```
            ┌──────────── voice in ────────────┐                       ┌──── voice out ────┐
 Mic ──► Silero VAD ──► Whisper (answers) ──► transcript ──► Scoring ──► Report (HTML + JSON)
            │          Whisper (commands) ──► "yes" / "that's my answer" / "stop the interview"
            │                                     │
            │                                     └──► Delivery: pace, pauses, filler words
 Camera ──► MediaPipe Face Landmarker ──► eye contact % ──► Report
                       └──► where the student is ──► face page eyes (+ robot arm later)

 Competency slot + session memory + CpE/ECE library ──► local AI (Qwen2.5, llama.cpp) ──► guard v2
                                     (approved rubric question if the guard fails) ──► Piper ──► speaker
                                                                                  + animated face + subtitles
```

Everything runs **locally and offline**. Internet is only needed once, to install packages and
download the models.

---

## 2. Stack at a glance

| Layer | Technology | Version | Used for |
|---|---|---|---|
| Language | **Python** | 3.11.9 | Everything (same version as Raspberry Pi OS Bookworm) |
| Voice activity | **Silero VAD** (ONNX) via **onnxruntime** | silero-vad 6.2.3 / onnxruntime 1.30.0 | Is the student speaking? Splits speech, measures pauses |
| Speech-to-text | **OpenAI Whisper** via **faster-whisper** (CTranslate2) | 1.2.1 / ctranslate2 4.8.2 | Answers (accurate model) and commands (fast model) |
| Text-to-speech (EN) | **Piper** (`en_US-lessac-medium`) | piper-tts 1.8.0 | The robot's English voice |
| Text-to-speech (FIL) | **Meta MMS-TTS Tagalog** via **transformers** + **torch** | 5.18.0 / 2.14.1 | Filipino clips, made in advance on the PC only |
| Vision | **Google MediaPipe** Face Landmarker | 1.0.1 | Face presence, turn angle, head pose, eye contact |
| Camera | **OpenCV** | 5.0 | Reads camera frames |
| Scoring | **multilingual-e5-small** via **sentence-transformers** | 6.1.0 | Compares answers with the rubric's example answers |
| Interviewer (local AI) | **Qwen2.5-Instruct** (GGUF 4-bit) via **llama.cpp** (`llama-cpp-python`) | 0.3.36 | Writes questions, follow-ups, memory facts and the coaching summary (1.5B PC, 0.5B Pi); in scripted mode only rephrases approved replies |
| Knowledge library | ~48 CpE/ECE passages, embedded with **multilingual-e5-small** | - | Keeps technical questions relevant (`samples/knowledge/`) |
| Laughter | **YAMNet** via MediaPipe Audio Classifier | 1.0.1 | Detects the student's laughter so the robot can mirror it once |
| Math / ML | **NumPy**, **SciPy**, **scikit-learn** | 2.4.6 / 1.17.1 / 1.9.1 | Signal math, resampling, future classifiers |
| Audio I/O | **sounddevice**, **soundfile** | 0.5.6 / 0.14.0 | Mic input, speaker output, WAV files |
| Face UI | **HTML + SVG + CSS + JavaScript** (no framework) | - | Animated face, subtitles, buttons (Chromium kiosk on the Pi) |
| UI link | **websockets** + Python `http.server` | 17.2 | Python app ↔ face page, live |
| Settings | **YAML** (PyYAML) | 6.0.3 | `config/pc.yaml`, `config/pi5.yaml`, `config/models.yaml` |
| Robot arm (later) | **pyserial** → ESP32 + PCA9685 | 3.5 | Turn and nod commands (T4.5) |
| Testing | **pytest** | 9.1.1 | 100+ tests, including full simulated sessions |
| Training (later) | **PyTorch**, **transformers**, **peft**, **optimum** on **Google Colab** | - | Only what fails a checkpoint (Stage 3) |
| Version control | **Git** + **GitHub** | - | Code only; recordings and models never committed |

---

## 3. Each part, explained

### 3.1 Voice in: hearing the student

**Silero VAD (Voice Activity Detection)**
- *What it is:* a tiny neural network (2 MB) that answers "is this speech?" for every 32 ms of audio.
- *How it's used:* `coach/audio/vad.py` feeds it 512-sample blocks from the mic (16 kHz). Speech that
  stops for 0.6 s closes a **segment**. Segments go to Whisper while the student is still talking, so
  the transcript is mostly ready when the answer ends.
- *Details that matter:*
  - The mic is ignored for **0.4 s** after the robot speaks, so its own echo isn't taken as an answer.
  - Trailing silence is cut off before Whisper, because Whisper invents words like "You" in silence.
  - It runs on onnxruntime directly (no PyTorch), so it's light enough for the Pi.

**Whisper (speech-to-text), two models**
- *What it is:* OpenAI's multilingual speech recognizer, run through **faster-whisper**, a fast
  C++ engine (CTranslate2) that also supports int8 compression.
- **Answer model** (accurate): `small` on the PC, `base` int8 on the Pi. It transcribes the answers that
  get scored, with word timestamps (used for pace). A hint prompt ("Umm, uh…") keeps filler words in the
  transcript, which Whisper normally deletes. Languages are limited to English and Tagalog.
- **Command model** (fast): `base` int8, greedy decoding, no timestamps, language given. It recognizes
  short commands in about **1 s** instead of about 7.5 s with the answer model.
- *Where:* `coach/asr/faster_whisper_backend.py`; the fast model is built by `make_command_asr()` in
  `coach/profile.py`.

**Voice commands** (`coach/session/voice.py`)
- The student never has to touch the screen. "start / simulan", "yes / oo", "that's my answer / tapos
  na po", "stop the interview / itigil ang interview", "keep it / itago".
- Whisper's text is matched against phrase lists with **fuzzy matching**, so "top the interview" still
  counts as "stop the interview". "That's my answer" only counts at the **end** of what was said.
- A spoken "yes" to recording consent must be clear; consent is never assumed.

### 3.2 Voice out: the robot speaking

**Piper** (English) and **MMS-TTS Tagalog** (Filipino)
- Every robot line and rubric question is **synthesized in advance** by `offline/synth_questions.py`
  into `cache/questions/` as WAV files, plus a JSON file with:
  - **subtitle cues**: the text split into ≤ 2 lines × 42 characters, timed by text length
  - **lip-sync levels**: the audio's loudness every 30 ms turned into mouth-open levels 0–3
- At runtime a clip just loads (< 0.2 s), so speaking never waits for synthesis. Piper also speaks
  live English lines when needed.
- Every clip ends with 0.35 s of silence; without it Windows cut off the last syllable.
- *Known limit:* MMS Tagalog mispronounces English technical words inside Filipino sentences. Faculty
  should listen to the Filipino clips.

### 3.3 Language: English or Filipino, never mixed
- The robot starts in **English**. The student's **first spoken reply** decides the language once
  ("simulan" / "oo" = Filipino), and it's **locked for the whole interview**.
- Filipino is used only if **every** Filipino clip exists; otherwise the whole session stays English.

### 3.4 Vision: the camera

**MediaPipe Face Landmarker** (`coach/vision/`)
- *What it is:* Google's model that finds **478 points** on a face, including the irises, and the
  head's 3D rotation.
- *How it's used:* one pass per frame gives everything:
  - **presence**: is someone there?
  - **turn angle**: where the student is, for the face page's eyes and later the robot arm
  - **eye contact**: head yaw and pitch within ±15° **and** irises near the center of the eyes
- Eye contact % is computed over each answer and goes into the report.
- Runs at 30 fps on the PC and is capped at 12 fps on the Pi.

### 3.5 Scoring: how an answer gets its rubric score

**multilingual-e5-small** (sentence embeddings) + rules (`coach/scoring/`)
1. Each rubric criterion has faculty **example answers** at levels 5, 3 and 1.
2. The answer and every example become **embedding vectors**: numbers that capture meaning, so a
   Taglish answer can be compared with English examples.
3. **Cosine similarity** says how close the answer is to each level's examples.
4. A softmax over the levels (temperature 0.02) gives a score from 1 to 5.
5. The report stores **which criterion and which example answer** produced the score. That's the
   "traceable to an approved criterion" promise in the proposal.
6. The correction tip is the faculty-written tip of the **lowest-scoring criterion**.

This is **scoring v0**: pretrained, no training. **Scoring v1** (T3.1) will train a small ordinal
classifier on the same features (`coach/scoring/features.py`) against real faculty ratings.

### 3.6 The robot's reply to what the student said (`coach/reply/`)
1. **Laughter or acknowledgment:** YAMNet checks each speech segment for laughter. If the student laughed,
   the robot mirrors it **once per session** ("Haha!" + an "amused" face) and never laughs first.
   Otherwise it says a neutral "I see." / "Okay, I understand.", which also covers the next step's wait.
2. **Understand** (from a rough transcript by the fast Whisper, about 2 s):
   - **topics** from the approved **glossary** (`samples/glossary.json`, ~60 CpE/ECE terms with mishearings,
     so "ESB 32" becomes ESP32)
   - **feelings** ("I'm nervous", "kinakabahan", "hindi ko alam") → reassurance line
   - answer length and weakest criterion (rough score)
3. **Decide one reply** (meaning comes only from approved lines): a topic follow-up ("You mentioned the
   Arduino. Why did you choose it?") if the answer isn't yet strong → else the rubric follow-up or "tell me
   more" → else a topic acknowledgment. At most **one follow-up** per question; topics never repeat.
4. **Say it:** in English sessions the **local AI** (Qwen2.5 on llama.cpp, ~1.6 s on the PC) **rephrases the
   approved line** to refer to the student's words, e.g. "You mentioned using an Arduino for a water monitor.
   Why did you select it for that project?". `guard.py` rejects any reply that judges, drops the question,
   adds numbers or new facts, drops the topic, or speaks as the student. A rejected or late (> 5 s) reply is
   replaced by the approved line. Filipino sessions use approved clips only (no live Filipino voice).
- The report records each reply: what was said, whether the AI wrote it, the check result, and why it was chosen.
- This matches the proposal: a local language model "may rephrase or follow up only from approved items".

### 3.6b The generative interviewer (`coach/interviewer/`, `coach/session/generative.py`)
In English sessions (`interviewer.mode: generative`, the default) the robot talks like a real interviewer
with CpE/ECE knowledge, instead of reading fixed questions.

| Part | File | What it does |
|---|---|---|
| Blueprint | `samples/blueprint.json` | One **competency slot** per rubric question: WHAT the question must find out. The AI picks the wording; the answer is scored with that slot's rubric, so scoring stays valid. |
| Session memory | `coach/interviewer/memory.py` | The student's **first name** (asked and confirmed after consent), topics, feelings, and short facts from every answer. RAM only, cleared at the end. AI facts are kept only if the student really said those words. |
| Knowledge library | `coach/interviewer/knowledge.py` | Looks up the CpE/ECE passage closest to what the student said, so a technical question stays relevant. |
| Interviewer | `coach/interviewer/interviewer.py` | One prompt with worked examples for 4 tasks: QUESTION, FOLLOW_UP, FACTS, COACH. The fixed part is cached by llama.cpp, so only the short variable part is processed each time. |
| Guard v2 | `coach/interviewer/guard.py` | Rejects judging, "we/our", private topics, invented names/numbers/technical terms, and false "you mentioned ..." claims. A rejected or late sentence is replaced by the approved line. |

**One question, step by step:** after an answer, the robot says a short acknowledgment ("I see.") and, if a
criterion scored ≥ 4 in the quick check, **specific praise** ("That's a really good answer, Leonard. You explained
the technology you used and why you chose it."). It may ask one AI-written follow-up. While the student answers
that, the AI already writes the **next question** from: the slot's goal, the memory notes, one earlier fact to
connect to ("Earlier you mentioned your ESP32 attendance system..."), the name (at most every 2 turns), and one
library passage. If the question isn't ready, the robot says "Let me think about my next question."

**Measured on the PC** (`python -m eval.interviewer_eval`, 4 scripted students × 5 questions):

| Model | Questions passing the guard | Follow-ups passing | Thinking time p50 / p90 |
|---|---|---|---|
| Qwen2.5-1.5B (PC) | 90% | 85% | 2.9 s / 4.5 s |
| Qwen2.5-0.5B (Pi) | 25% | 60% | 2.0 s / 3.1 s |

The rest falls back to approved lines, so the interview never breaks. The 0.5B model often talks as if it were
the student; that is what the fine-tuning in section 7 is for. In a full simulated interview with the 1.5B model, all 5
questions were AI-written and personalized, e.g. "Leonard, for your capstone project, you built an attendance system
with an ESP32. Did you face any disagreement while working on it, and what did you do to resolve it?"

### 3.7 Delivery: how the student spoke (`coach/delivery/`)
Pure rules, no ML:
- **Pace:** words per minute, from Whisper's word timings.
- **Long pauses:** gaps of 3 s or more between speech segments.
- **Response delay:** how long the student took to start.
- **Filler words:** "um, uh, hmm" always count. "ano, parang, like" count only next to a pause mark or
  when repeated, because they're also real words.
- Each measure turns into a plain-language tip when it's outside the target range.

### 3.8 The session and the face page
- `coach/session/state_machine.py` runs the interview: wait for "start" → greeting → consent →
  questions with replies → "processing your results" → results → keep or delete the report.
- **Timing rules:**
  - 5 s of silence after speaking ends an answer
  - a student who never starts is moved on after 15 s
  - a choice screen asks again if nobody answers
- **Background work** runs on two threads: one for accurate transcription and scoring, and one for
  decisions the student is waiting for (commands, follow-ups), so those never wait behind slow work.
- **Face page** (`coach/face/ui/index.html`): an SVG face with 7 expressions, none negative, eyes that
  follow the student, lip-sync, subtitles and small choice buttons under the face. Python sends it live
  updates over a WebSocket. The same page runs full screen in Chromium on the Pi.
- **Courses:** CpE and ECE only. The robot never asks the course; each session asks the common questions
  plus one CpE and one ECE technical question, rotated between sessions.

### 3.9 Reports
`coach/session/report.py` writes `report.html` (printable, plain language) and `report.json` (for
progress tracking and later training) into `data/sessions/<date-time>/`, only if the student chooses to
keep it.

---

## 4. One code base, two machines

Moving from the PC to the Pi is a **settings change**, not a rewrite:

| Setting | `config/pc.yaml` | `config/pi5.yaml` |
|---|---|---|
| Answer Whisper | small, float32, beam 5 | base, int8, beam 1, 2 threads |
| Command Whisper | base, int8 | base, int8 |
| Camera | 1280×720 @ 30 fps | 640×480 @ 12 fps |
| Scoring engine | sentence-transformers (PyTorch) | ONNX int8, no PyTorch (needs the T4.2 export) |
| Robot speech | live + clips | clips only |
| Face page | 60 fps, glow effects | 24 fps, no effects |
| Local AI | Qwen2.5-1.5B Q4, 6 threads | Qwen2.5-0.5B Q4, 3 threads (fine-tuned version planned, G4) |

**Rule:** the runtime package `coach/` never imports PyTorch, so it can run on the Pi. A test checks
this in a fresh Python process. PyTorch is only used in `training/`, `offline/` and the PC scoring backend.

---

## 5. Folder structure

```
interview-coach/
├── coach/                 runtime package (runs on PC and Pi, no torch imports)
│   ├── audio/             mic capture, Silero VAD, resampling
│   ├── asr/               Whisper backends (faster-whisper; whisper.cpp planned)
│   ├── tts/               Piper backend, clip cache with subtitles + lip-sync
│   ├── vision/            MediaPipe face tracking, eye contact, camera thread
│   ├── delivery/          pace, pauses, filler words, tips
│   ├── scoring/           rubric loader, embedders, scorer, corrections
│   ├── interviewer/       generative interviewer: prompts, session memory, knowledge library, guard v2
│   ├── reply/             topics, feelings, laughter, rephraser (scripted mode), shared local AI
│   ├── face/              face page (ui/), WebSocket server, expressions, lip-sync, subtitles
│   ├── session/           interview flow, voice commands, robot lines, report, launcher (run.py)
│   ├── robot/             ESP32 serial bridge (planned, T4.5)
│   └── profile.py         loads pc / pi5 settings, builds every model
├── config/                pc.yaml, pi5.yaml, models.yaml (model list + download links)
├── samples/rubric/        UNAPPROVED sample rubric: common.json, cpe.json, ece.json
├── samples/               also blueprint.json (competency slots), knowledge/ (CpE/ECE library), glossary, replies
├── offline/               one-time tools: download_models.py, synth_questions.py
├── training/              training + export scripts; interviewer/ = synthetic data + LoRA (G3/G4, Colab)
├── eval/                  simulated student, benchmarks (WER, kappa, gaze, fillers: placeholders)
├── tests/                 pytest suite
├── models/                downloaded models, about 1.4 GB (NOT in git)
├── cache/                 generated speech clips (NOT in git)
├── data/                  recordings, ratings, reports (NOT in git, never uploaded)
└── requirements-*.txt     common / pc / pi / train / lock-pc
```

---

## 6. How the PC was set up

```powershell
# 1. Python 3.11 (Raspberry Pi OS Bookworm uses 3.11 too)
winget install Python.Python.3.11

# 2. Project environment
cd "C:\Users\Laizame Forrosuelo\RoboticArm\interview-coach"
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements-lock-pc.txt        # exact versions (or requirements-pc.txt for latest)

# 3. Models, about 1.4 GB, once
python -m offline.download_models --profile pc --all-asr --all-llm

# 4. Robot speech clips, again after any change to questions or robot lines
python -m offline.synth_questions

# 5. Check and run
pytest
python -m coach.session.run
```

**Requirements files:**

| File | For | Contains |
|---|---|---|
| `requirements-common.txt` | PC + Pi | runtime packages, no PyTorch |
| `requirements-pc.txt` | PC | common + sentence-transformers (PyTorch) + pytest |
| `requirements-pi.txt` | Pi 5 | common only |
| `requirements-train.txt` | PC / Colab | pc + torch, transformers, peft, datasets, optimum, jiwer, pandas |
| `requirements-lock-pc.txt` | PC | exact versions that are known to work |

**Offline:** `run.py` sets `HF_HUB_OFFLINE=1`, so models load only from `models/` and the app never
contacts the internet. MediaPipe's startup log noise is silenced in `coach/vision/face.py`.

---

## 7. Training on Google Colab

The PC has **no NVIDIA GPU**. Most planned training is small and runs fine on the PC CPU. Colab (a free
GPU in the browser) is only needed for the heavy job.

| Task | What | Where |
|---|---|---|
| T3.1 Scoring v1 | small ordinal classifier on embedding features | **PC** (seconds to minutes on CPU) |
| T3.2 Eye-contact classifier | tiny model on face-landmark numbers | **PC** |
| T3.3 Filler classifier | small audio/text classifier | **PC** |
| T3.4 Whisper fine-tune (LoRA) | adapts Whisper to Filipino / Cebuano-accented English | **Colab GPU** (hours on CPU) |
| T4.2 Model export | embedder → ONNX int8, Whisper → CTranslate2 int8 | PC or Colab |
| G3 Interviewer data | a large open teacher model (Qwen2.5-14B) role-plays students and the interviewer | **Colab GPU** |
| G4 Interviewer fine-tune | LoRA on Qwen2.5-1.5B / 0.5B, then GGUF Q4 for llama.cpp | **Colab GPU** (free T4 is enough) |

The training scripts in `training/` are still placeholders (T3.x). They get written once Stage 2 shows
what needs training.

### ⚠ Privacy before you upload anything
Colab runs on Google's servers. The proposal promises that **recordings stay on the device**, and
student voices and faces are personal data under the **Data Privacy Act (RA 10173)**.
- **Fine to use on Colab:** public datasets (Google FLEURS, Mozilla Common Voice Filipino), code, and
  model files.
- **Student recordings:** only if the signed consent form (T0.2) and your adviser / ethics committee
  allow cloud processing. Otherwise train on public data on Colab, and do any step that needs student
  audio on the PC.
- Never upload names. Use anonymous codes.

### Colab steps (for T3.4, once the script exists)
1. Open https://colab.research.google.com → **New notebook**.
2. **Runtime → Change runtime type → T4 GPU**.
3. Add your GitHub token under **Secrets** (key icon) as `GH_TOKEN`, since the repository is private.
4. Run these cells:

```python
# Cell 1: get the code
from google.colab import userdata
token = userdata.get("GH_TOKEN")
!git clone https://{token}@github.com/<your-username>/interview-coach.git
%cd interview-coach

# Cell 2: install training packages (Colab already has torch with GPU support)
!pip install -q -r requirements-train.txt

# Cell 3: models (same downloader as the PC)
!python -m offline.download_models --profile pc --all-asr

# Cell 4: save results to Google Drive so they survive the session
from google.colab import drive
drive.mount("/content/drive")

# Cell 5: train and export (T3.4 / T4.2, scripts to be written)
!python -m training.whisper_lora --out /content/drive/MyDrive/interview-coach/whisper_ft
!python -m training.export.export_whisper_ct2 --src /content/drive/MyDrive/interview-coach/whisper_ft \
        --out /content/drive/MyDrive/interview-coach/whisper_ft_int8
```

5. Download the exported folder from Drive into `models/whisper/` on the PC, and point
   `asr.model` in the config at it.

### Colab steps for the interviewer (G3 → G4), ready now
Synthetic data only: no student recordings or names are uploaded.
```python
!pip install -q -r requirements-train.txt
# G3: ~600 made-up interviews; only turns that pass the robot's own guard are kept
!python -m training.interviewer.make_synthetic --dialogues 600 --out /content/drive/MyDrive/ic/interviewer.jsonl
# (faculty read a random sample of the records before training)
# G4: LoRA fine-tune, then convert to GGUF Q4_K_M (commands at the top of finetune_lora.py)
!python -m training.interviewer.finetune_lora --base Qwen/Qwen2.5-1.5B-Instruct \
        --data /content/drive/MyDrive/ic/interviewer.jsonl --out /content/drive/MyDrive/ic/interviewer-1.5b
```
Copy `interviewer-1.5b-q4_k_m.gguf` to `models/llm/`, run
`python -m eval.interviewer_eval --model interviewer-1.5b-q4`, and set `reply.llm_model: interviewer-1.5b-q4`
only if it beats the base model. Repeat with `Qwen/Qwen2.5-0.5B-Instruct` for the Pi.

**Colab notes:**
- Free sessions disconnect after a few hours of inactivity or use. Save checkpoints to Drive.
- Colab's Python version can differ from 3.11. Use `requirements-train.txt`, not the PC lock file.
- Colab has no access to your mic, camera or face page. It's only for training, not for running the robot.

---

## 8. Testing

- `pytest` runs 100+ tests, including:
  - each module with known test audio (Piper speaks sentences with known words)
  - the face page server
  - scoring order (good > mid > poor)
  - full simulated interviews (tapped and spoken)
  - language locking, timing rules, follow-ups
  - the generative interviewer: names, memory, guard v2, AI questions with fallback, praise, coaching,
    and a full simulated interview with the real local model
  - the "no PyTorch in `coach/`" check
- `python -m eval.simulate_session --sessions 3` runs gate G1 with a simulated student
  (`--generative` for the AI interviewer).
- `python -m eval.interviewer_eval` measures the AI interviewer (guard pass rate, personalization, latency).
- Model-dependent tests skip themselves if `models/` is empty.

---

## 9. Not done yet (PC side)

- **Stage 0 (team):** consent form, faculty rubric (replaces the sample), mic array + camera, pilot
  recordings, human ratings, typed transcripts.
- **Stage 2:** measure word error rate, scoring agreement (κ ≥ 0.60), eye-contact and filler accuracy
  on real data. Needs Stage 0.
- **Stage 3:** train only what fails (scoring v1 almost certainly).
- **T4.5:** ESP32 robot-arm bridge.
- **Generative interviewer:** G0 faculty approval of the slots, knowledge passages and praise lines; G3/G4 run on
  Colab; human-likeness ratings (G7) on recorded sessions; off-topic steering; live Filipino voice (MMS → ONNX).
- Full plan and task tracker: `~/Downloads/Interview-Coach-ML-Plan.html`.
