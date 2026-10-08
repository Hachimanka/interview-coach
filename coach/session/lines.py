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
    "program": {
        "en": "What is your program? Say computer, electrical, or electronics.",
        "fil": "Ano ang iyong programa? Sabihin ang computer, electrical, o electronics.",
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
    "take_your_time": {
        "en": "Take your time.",
        "fil": "Dahan-dahan lang.",
    },
    "next": {
        "en": "Thank you. Next question.",
        "fil": "Salamat. Susunod na tanong.",
    },
    "closing": {
        "en": "That was the last question. Thank you! Your report is ready.",
        "fil": "Iyon na ang huling tanong. Salamat! Handa na ang iyong report.",
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

# On-screen text (not spoken). "both" is shown while the student's language is still unknown.
TEXT: dict[str, dict[str, str]] = {
    "start_voice": {"both": "Say “start” or “simulan” to begin."},
    "start_touch": {"both": "Tap the screen or sit in front of me to start practicing."},
    "consent_title": {"en": "Do you agree to continue?", "fil": "Pumapayag ka ba?",
                      "both": "Do you agree to continue? / Pumapayag ka ba?"},
    "program_title": {"en": "Choose your program", "fil": "Piliin ang iyong programa",
                      "both": "Choose your program / Piliin ang iyong programa"},
    "keep_title": {"en": "Keep your report on this device?", "fil": "Itago ang report sa device na ito?",
                   "both": "Keep your report on this device? / Itago ang report?"},
    "saved": {"en": "Your report is saved.", "fil": "Na-save na ang iyong report."},
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
