"""Tiny, deterministic check: did the caller just decline further help or say goodbye?

The language model decides *when it wants* to end a call, but models wrap up
too eagerly. Before the server actually hangs up it therefore checks the
caller's own last words with plain pattern matching. No model is involved, so
the check is predictable and testable.

``caller_reply_kind`` returns:

* ``"goodbye"`` - an explicit farewell ("bye", "have a nice day", "talk later")
* ``"decline"`` - "no / that's all / nothing else / I'm good"
* ``None``      - anything else: a statement, a question, an answer

``interrupts_closing`` answers a narrower question: the closing line is being
spoken and the caller said something - is it a real new request (then the bot
stops and listens), or just "ok / thanks / bye" or talk that is not meant for
the bot (then the closing line simply finishes)?
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

GOODBYE = "goodbye"
DECLINE = "decline"

_GOODBYE = re.compile(
    r"\b("
    r"bye|goodbye|good bye|bye bye|see you|see ya|talk to you later|talk later|take care|"
    r"have a (good|nice|great|lovely) (day|one|evening|night|weekend)|good night|"
    r"i have to go|i've got to go|i gotta go|i need to go|(?<!don't )(?<!do not )(?<!dont )hang up|end the call"
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


# --------------------------------------------------------------------------- Hebrew / Arabic
# Same four ideas as in English (goodbye, "nothing more", a bare "no", a bare
# "thanks") plus what counts as a question or a new request. Arabic patterns
# are written in a folded spelling (ا for أ إ آ, ه for ة, ي for ى, no tashkeel)
# and the caller's words are folded the same way before matching.
_AR_FOLD = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ة": "ه", "ى": "ي", "ـ": None, "؟": "?"})


def _normalise_any(text: str) -> str:
    """Lower-case, strip marks and punctuation, keep letters of any script."""
    text = unicodedata.normalize("NFKC", text or "").translate(_AR_FOLD).lower()
    text = "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[^\w' ]+", " ", unicodedata.normalize("NFC", text), flags=re.UNICODE)
    return " ".join(text.split())


@dataclass(frozen=True)
class _Words:
    goodbye: "re.Pattern[str]"
    decline_anywhere: "re.Pattern[str]"
    bare_no: "re.Pattern[str]"
    thanks_only: "re.Pattern[str]"
    question: "re.Pattern[str]"
    new_request: "re.Pattern[str]"


def _w(alternatives: str) -> "re.Pattern[str]":
    return re.compile(rf"(?<!\w)({alternatives})(?!\w)")


_HEBREW = _Words(
    goodbye=_w("להתראות|ביי|ביי ביי|יום טוב|יום נעים|ערב טוב|לילה טוב|שבוע טוב|נדבר|אני חייב ללכת|אני חייבת ללכת|תנתק|תנתקי|לסיים את השיחה"),
    decline_anywhere=_w("זה הכל|זה הכול|זהו|לא תודה|אין צורך|לא צריך|לא צריכה|שום דבר|שום דבר אחר|הכל טוב|הכול טוב|אני מסודר|אני מסודרת|אין לי שאלות|אין לי עוד שאלות|זה מספיק"),
    bare_no=re.compile(r"^(לא|לא לא)(?!\w)"),
    thanks_only=re.compile(r"^(אוקיי|אוקי|או קיי|בסדר|בסדר גמור|טוב|מעולה|סבבה|יופי|מצוין)?\s*(תודה|תודה רבה|תודה רבה לך|תודה לך)$"),
    question=_w("מה|מתי|איפה|איזה|איזו|מי|למה|איך|כמה|האם|אפשר|יש לכם|אתם"),
    new_request=_w("רגע|שנייה|שניה|חכה|חכי|עוד שאלה|עוד משהו|יש לי שאלה|יש לי עוד|רציתי לשאול|רציתי לדעת|תגיד|תגידי|מה עם|בעצם|אל תנתק|אל תנתקי"),
)
_ARABIC = _Words(
    goodbye=_w("مع السلامه|باي|باي باي|يلا باي|يلا سلام|الله معك|الله معكم|بخاطرك|في امان الله|تصبح علي خير|تصبحي علي خير|بشوفك|لازم اروح|سكر الخط|سكري الخط"),
    decline_anywhere=_w("لا شكرا|بس هيك|هيك بس|ما في شي|ما في اشي|مافي شي|ما في شي تاني|خلص|خلاص|ولا شي|ولا اشي|ما بدي|ما بدي شي|بكفي|هذا كل شي|هاد كل شي|لا خلص|تمام هيك"),
    bare_no=re.compile(r"^(لا|لاء|لا لا)(?!\w)"),
    thanks_only=re.compile(r"^(تمام|ماشي|طيب|اوكي|اوكيه|منيح|حلو|ممتاز)?\s*(شكرا|شكرا الك|شكرا كتير|شكرا جزيلا|يسلمو|يسلموا|مشكور|مشكوره|يعطيك العافيه)$"),
    question=_w("شو|ايش|متي|ايمتي|وين|مين|ليش|كيف|قديش|اديش|كم|هل|ممكن|بقدر|فيني|في عندكم|عندكم"),
    new_request=_w("لحظه|استني|استنا|دقيقه|ثانيه|سؤال|عندي سؤال|بدي اسال|بدي اعرف|شي تاني|اشي تاني|كمان شي|كمان اشي|طيب و|لا تسكر|لا تسكري|احكيلي|خبرني|خبريني"),
)
_BY_LANGUAGE = {"he": _HEBREW, "ar": _ARABIC}


def _kind_in(words: _Words, text: str) -> Optional[str]:
    norm = _normalise_any(text)
    if not norm:
        return None
    if words.goodbye.search(norm):
        return GOODBYE
    asks = "?" in (text or "") or "؟" in (text or "") or bool(words.question.search(norm))
    if words.decline_anywhere.search(norm) and not asks:
        return DECLINE
    if words.bare_no.match(norm) and len(norm.split()) <= MAX_BARE_NO_WORDS and not asks:
        return DECLINE
    if words.thanks_only.match(norm):
        return DECLINE
    return None


def caller_reply_kind(text: str, language: str = "en") -> Optional[str]:
    """Classify the caller's last utterance (see module docstring)."""
    words = _BY_LANGUAGE.get((language or "en").split("-")[0].lower())
    if words is not None:
        return _kind_in(words, text)
    norm = _normalise(text or "")
    if not norm:
        return None
    if _GOODBYE.search(norm):
        return GOODBYE
    # A reply that still asks something is not a decline, whatever else it contains.
    asks = "?" in (text or "") or bool(_QUESTION_WORDS.search(norm))
    if _DECLINE_ANYWHERE.search(norm) and not asks:
        return DECLINE
    words_en = norm.split()
    if _BARE_NO.match(norm) and len(words_en) <= MAX_BARE_NO_WORDS and not asks:
        return DECLINE
    if _THANKS_ONLY.match(norm):
        return DECLINE
    return None


#: Things a caller says when they do want something more from the bot.
_NEW_REQUEST = re.compile(
    r"\b("
    r"wait|hold on|hang on|one more|another (question|thing)|one (other|last) (question|thing)|"
    r"i have a question|i (also )?(want|wanted|need|would like|'d like) to (ask|know)|"
    r"can i ask|let me ask|tell me|what about|how about|actually|before you go|don'?t hang up|sorry but"
    r")\b"
)


def interrupts_closing(text: str, language: str = "en") -> bool:
    """Should these words, said while the closing line is playing, stop it?

    Only a real new request does: a question, or a phrase like "wait" or "one
    more thing". A goodbye, a thank-you, "ok", an echo of the bot or a remark
    to someone else in the room does not - the bot finishes the line and
    hangs up. Decided from the transcript alone.
    """
    words = _BY_LANGUAGE.get((language or "en").split("-")[0].lower())
    if words is not None:
        norm = _normalise_any(text)
        if not norm:
            return False
        if words.new_request.search(norm):
            return True
        if _kind_in(words, text) is not None:
            return False
        return "?" in (text or "") or "؟" in (text or "") or bool(words.question.search(norm))
    norm = _normalise(text or "")
    if not norm:
        return False
    if _NEW_REQUEST.search(norm):
        return True
    if caller_reply_kind(text) is not None:
        return False  # "no thanks", "bye", "ok thank you"
    return "?" in (text or "") or bool(_QUESTION_WORDS.search(norm))
