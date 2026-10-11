"""Fixed robot lines, pre-synthesized with the questions (cache key: line.<name>).

After changing any text here, run `python -m offline.synth_questions` again: Filipino lines are
spoken only from pre-made clips, and a clip made from older wording is not reused.
"""

LINES: dict[str, dict[str, str]] = {
    "greeting": {
        "en": "Hello! I am your practice interviewer. This session is only for practice, and you can stop at any time.",
        "fil": "Kumusta! Ako ang iyong practice interviewer. Ang session na ito ay para lamang sa pagsasanay, at maaari kang huminto anumang oras.",
    },
    "consent": {
        "en": "Your answers will be recorded on this device to give you feedback. Do you agree to continue? Please say yes or no.",
        "fil": "Ire-record ang iyong mga sagot sa device na ito para mabigyan ka ng feedback. Pumapayag ka bang magpatuloy? Sabihin ang oo o hindi.",
    },
    "consent_touch": {
        "en": "Your answers will be recorded on this device to give you feedback. Do you agree to continue?",
        "fil": "Ire-record ang iyong mga sagot sa device na ito para mabigyan ka ng feedback. Pumapayag ka bang magpatuloy?",
    },
    "instructions": {
        "en": "I will ask you a few questions. Answer out loud. When you finish each answer, say: that's my answer.",
        "fil": "Magtatanong ako ng ilang tanong. Sumagot nang malakas. Kapag tapos ka na sa bawat sagot, sabihin: iyon na po ang sagot ko.",
    },
    "instructions_touch": {
        "en": "I will ask you a few questions. Answer out loud, and press Done when you finish each answer.",
        "fil": "Magtatanong ako ng ilang tanong. Sumagot nang malakas, at pindutin ang Done kapag tapos ka na sa bawat sagot.",
    },
    "retry": {
        "en": "Sorry, I didn't catch that. Please say it again.",
        "fil": "Paumanhin, hindi ko narinig. Pakiulit po.",
    },
    # Short acknowledgments right after an answer (rotated so the robot does not sound repetitive).
    # They are neutral on purpose: they never praise or criticize before the rubric score exists.
    "ack_1": {"en": "I see.", "fil": "Naiintindihan ko po."},
    "ack_2": {"en": "Okay, I understand.", "fil": "Sige po, naiintindihan ko."},
    "ack_3": {"en": "Alright.", "fil": "Okay po."},
    "reassure": {    # the student said they are nervous or stuck; neutral, never judges the answer
        "en": "That's okay. Take a breath, and continue whenever you're ready.",
        "fil": "Ayos lang po. Huminga nang malalim, at magpatuloy kapag handa ka na.",
    },
    "laugh": {       # only when mirroring the student's laughter, at most once per session.
        # A real recorded chuckle can replace cache/questions/line.laugh.en.wav / .fil.wav.
        "en": "Haha!",
        "fil": "Haha!",
    },
    "tell_more": {   # follow-up when the answer was very short
        "en": "Could you tell me a bit more? For example, a specific experience or project.",
        "fil": "Maaari mo bang dagdagan pa? Halimbawa, isang tiyak na karanasan o proyekto.",
    },
    "take_your_time": {
        "en": "Take your time.",
        "fil": "Dahan-dahan lang.",
    },
    "next": {
        "en": "Thank you. Next question.",
        "fil": "Salamat. Susunod na tanong.",
    },
    "closing": {
        "en": "That was the last question. Thank you for your answers!",
        "fil": "Iyon na ang huling tanong. Salamat sa iyong mga sagot!",
    },
    "processing": {   # said only while scoring is still running
        "en": "Please give me a moment while I process your results.",
        "fil": "Sandali lang po, pinoproseso ko ang iyong resulta.",
    },
    "ready": {        # said only after every answer has actually been scored
        "en": "Your results are ready.",
        "fil": "Handa na ang iyong resulta.",
    },
    "keep": {
        "en": "Do you want to keep your report on this device? Say keep it, or delete it.",
        "fil": "Gusto mo bang itago ang iyong report sa device na ito? Sabihin ang itago, o burahin.",
    },
    "declined": {
        "en": "No problem. Come back anytime you want to practice.",
        "fil": "Walang problema. Bumalik ka anumang oras na gusto mong magsanay.",
    },
}

# Generative mode only (English sessions, coach/session/generative.py): spoken live or from cache key gen.<name>.
# Not part of LINES, so they never affect whether a Filipino session is possible.
GEN_LINES: dict[str, str] = {
    "ask_name": "Before we begin, may I know your first name?",
    "name_retry": "Sorry, I didn't catch that. Could you say just your first name?",
    "name_skip": "No problem. Let's begin.",
    # said only if the next question is not written yet (covers the AI's thinking time, like a person would)
    "think_1": "Okay, let me think about my next question.",
    "think_2": "Alright, give me a second.",
    "think_3": "Hmm, let me see.",
}

# On-screen text (not spoken). "both" is shown while the student's language is still unknown.
TEXT: dict[str, dict[str, str]] = {
    "start_voice": {"en": "Say “start” to begin.", "fil": "Sabihin ang “simulan” para magsimula.",
                    "both": "Say “start” or “simulan” to begin."},
    "start_touch": {"en": "Tap Start to begin practicing.", "fil": "Pindutin ang Simulan para magsimula.",
                    "both": "Tap the screen to start practicing."},
    "consent_title": {"en": "Do you agree to continue?", "fil": "Pumapayag ka ba?",
                      "both": "Do you agree to continue? / Pumapayag ka ba?"},
    "keep_title": {"en": "Keep your report on this device?", "fil": "Itago ang report sa device na ito?",
                   "both": "Keep your report on this device? / Itago ang report?"},
    "name_title": {"en": "Did I get your name right?", "fil": "Tama ba ang pangalan mo?",
                   "both": "Did I get your name right?"},
    "saved": {"en": "Your report is saved.", "fil": "Na-save na ang iyong report."},
    "processing": {"en": "Processing your results… {done} of {total}",
                   "fil": "Pinoproseso ang iyong resulta… {done} sa {total}"},
}

# Button labels per command.
LABELS: dict[str, dict[str, str]] = {
    "start": {"en": "Start", "fil": "Simulan", "both": "Start"},
    "yes": {"en": "Yes", "fil": "Oo", "both": "Yes / Oo"},
    "no": {"en": "No", "fil": "Hindi", "both": "No / Hindi"},
    "keep": {"en": "Keep it", "fil": "Itago", "both": "Keep it / Itago"},
    "delete": {"en": "Delete it", "fil": "Burahin", "both": "Delete it / Burahin"},
}


def text(key: str, lang: str | None) -> str:
    t = TEXT[key]
    return t.get(lang or "both") or t.get("both") or t["en"]


def label(command: str, lang: str | None) -> str:
    return LABELS.get(command, {}).get(lang or "both", command)
