"""Text helpers for the speech pipeline: sentence splitting of a token stream."""

from __future__ import annotations

import re

#: Words ending in a period that do not end a sentence.
_ABBREVIATIONS = frozenset(
    {"mr", "mrs", "ms", "dr", "prof", "st", "vs", "etc", "e.g", "i.e", "approx", "a.m", "p.m", "inc", "ltd"}
)
#: A sentence ends at . ! or ? (optionally followed by a closing quote/bracket) and then whitespace.
_BOUNDARY = re.compile(r"""[.!?]+["')\]]*\s+""")
_SOFT_BREAK = re.compile(r"[,;:]\s+")
_MARKUP = re.compile(r"[*_#`~|<>]+")


def clean_for_speech(text: str) -> str:
    """Remove characters a speech engine would read out or stumble on."""
    return " ".join(_MARKUP.sub("", text).split())


class SentenceSplitter:
    """Cuts a stream of LLM text fragments into complete sentences.

    The LLM sends a few characters at a time. We want to hand each sentence to
    the speech engine the moment it is complete, not wait for the whole reply.
    ``feed()`` returns the sentences completed by the new fragment;
    ``flush()`` returns whatever is left when the stream ends.

    A sentence is only cut when its terminator is followed by whitespace, so
    "3.5" and "example.com" stay whole, and a short list of abbreviations
    ("Dr. Smith") does not split. Very long clauses without a full stop are
    cut at a comma once they exceed ``max_chars`` so speech never stalls.
    """

    def __init__(self, max_chars: int = 220) -> None:
        self._buf = ""
        self._max_chars = max_chars

    def feed(self, fragment: str) -> list[str]:
        self._buf += fragment
        sentences: list[str] = []
        search_from = 0
        while True:
            match = _BOUNDARY.search(self._buf, search_from)
            if match is None:
                break
            candidate = self._buf[: match.end()]
            if self._ends_with_abbreviation(candidate):
                search_from = match.end()
                continue
            sentence = candidate.strip()
            self._buf = self._buf[match.end() :]
            search_from = 0
            if sentence:
                sentences.append(sentence)
        while len(self._buf) > self._max_chars:
            cut = None
            for soft in _SOFT_BREAK.finditer(self._buf, 0, self._max_chars):
                cut = soft.end()
            if cut is None:
                cut = self._buf.rfind(" ", 0, self._max_chars) + 1 or self._max_chars
            sentences.append(self._buf[:cut].strip())
            self._buf = self._buf[cut:]
        return sentences

    def flush(self) -> list[str]:
        rest, self._buf = self._buf.strip(), ""
        return [rest] if rest else []

    @staticmethod
    def _ends_with_abbreviation(candidate: str) -> bool:
        stripped = candidate.rstrip()
        if not stripped.endswith("."):
            return False
        words = stripped[:-1].split()
        if not words:
            return False
        last = words[-1].lstrip("\"'([").lower()
        # single letters ("J. Smith") and known abbreviations
        return last in _ABBREVIATIONS or (len(last) == 1 and last.isalpha() and last not in ("a", "i"))
