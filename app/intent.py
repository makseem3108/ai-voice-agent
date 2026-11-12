"""Tiny, deterministic check: did the caller just decline further help or say goodbye?

The language model decides *when it wants* to end a call, but models wrap up
too eagerly. Before the server actually hangs up it therefore checks the
caller's own last words with plain pattern matching. No model is involved, so
the check is predictable and testable.

``caller_reply_kind`` returns:

* ``"goodbye"`` - an explicit farewell ("bye", "have a nice day", "talk later")
* ``"decline"`` - "no / that's all / nothing else / I'm good"
* ``None``      - anything else: a statement, a question, an answer
"""

from __future__ import annotations

import re
from typing import Optional

GOODBYE = "goodbye"
DECLINE = "decline"

_GOODBYE = re.compile(
    r"\b("
    r"bye|goodbye|good bye|bye bye|see you|see ya|talk to you later|talk later|take care|"
    r"have a (good|nice|great|lovely) (day|one|evening|night|weekend)|good night|"
    r"i have to go|i've got to go|i gotta go|i need to go|hang up|end the call"
    r")\b"
)
#: Phrases that mean "nothing more" wherever they appear in the reply.
_DECLINE_ANYWHERE = re.compile(
    r"\b("
    r"that'?s all|that is all|that'?s it|that is it|that will be all|that'?ll be all|"
    r"nothing else|nothing more|nothing further|no more questions|no other questions|"
    r"no thanks|no thank you|no,? i'?m good|i'?m good|i am good|i'?m all set|all set|all good|"
    r"i'?m done|we'?re done|not really|no need|that'?s everything|that is everything"
    r")\b"
)
#: A bare "no" only counts when the reply starts with it and is short;
#: "No, I own a house" or "there is no roof" are answers, not a decline.
_BARE_NO = re.compile(r"^(no|nope|nah|no no)\b")
_THANKS_ONLY = re.compile(r"^(ok(ay)?|alright|all right|great|perfect|fine|good|cool|sure)?\s*(thanks|thank you)( very much| so much| a lot)?$")
_QUESTION_WORDS = re.compile(r"\b(what|when|where|which|who|why|how|can you|could you|do you|does|is there|are there)\b")

MAX_BARE_NO_WORDS = 4


def _normalise(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return " ".join(text.split())


def caller_reply_kind(text: str) -> Optional[str]:
    """Classify the caller's last utterance (see module docstring)."""
    norm = _normalise(text or "")
    if not norm:
        return None
    if _GOODBYE.search(norm):
        return GOODBYE
    # A reply that still asks something is not a decline, whatever else it contains.
    asks = "?" in (text or "") or bool(_QUESTION_WORDS.search(norm))
    if _DECLINE_ANYWHERE.search(norm) and not asks:
        return DECLINE
    words = norm.split()
    if _BARE_NO.match(norm) and len(words) <= MAX_BARE_NO_WORDS and not asks:
        return DECLINE
    if _THANKS_ONLY.match(norm):
        return DECLINE
    return None
