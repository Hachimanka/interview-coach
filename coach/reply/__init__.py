"""coach.reply: replies to what the student said (topics, feelings, laughter), approved lines first.

The local AI (llama.cpp) only REPHRASES an approved line so it refers to the student's words; every
output is checked by guard.py and the approved line is spoken as-is if the check fails or AI is slow.
"""
