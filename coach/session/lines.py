"""Fixed robot lines, pre-synthesized with the questions (cache key: line.<name>)."""

LINES: dict[str, dict[str, str]] = {
    "greeting": {
        "en": "Hello! I am your practice interviewer. This session is only for practice, and you can stop at any time.",
        "fil": "Kumusta! Ako ang iyong practice interviewer. Ang session na ito ay para lamang sa pagsasanay, at maaari kang huminto anumang oras.",
    },
    "consent": {
        "en": "Your answers will be recorded on this device to give you feedback. Do you agree to continue?",
        "fil": "Ire-record ang iyong mga sagot sa device na ito para mabigyan ka ng feedback. Pumapayag ka bang magpatuloy?",
    },
    "instructions": {
        "en": "I will ask you a few questions. Answer out loud, and press Done when you finish each answer.",
        "fil": "Magtatanong ako ng ilang tanong. Sumagot nang malakas, at pindutin ang Done kapag tapos ka na sa bawat sagot.",
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
    "declined": {
        "en": "No problem. Come back anytime you want to practice.",
        "fil": "Walang problema. Bumalik ka anumang oras na gusto mong magsanay.",
    },
}
