"""Read an amount of money out of a short phrase, in English, Hebrew or Arabic.

The language model only writes down what the caller said ("about 500 shekels",
"חמש מאות שקל", "خمسمية شيكل"). Turning that into a number and a currency is
done here, in plain code, so it is predictable and testable and the model's
reply format stays simple.

The parser is deliberately modest. It understands digits and ordinary number
words up to the thousands. If a phrase holds two different numbers ("between
400 and 500") it returns no amount rather than guess; the caller's words are
kept by whoever called us.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional, Union

Number = Union[int, float]

_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
# Arabic letters with several spellings are folded to one, so "مئة", "مية" and "ميه" match.
_ARABIC_FOLD = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ة": "ه", "ى": "ي", "ـ": None})

CURRENCIES: dict[str, tuple[str, ...]] = {
    "ILS": ("shekel", "shekels", "sheqel", "sheqels", "nis", "ils", "₪", "שקל", "שקלים", "שח", 'ש"ח', "ש״ח", "شيكل", "شيقل", "شواكل", "شواقل", "شيكلات"),
    "USD": ("dollar", "dollars", "usd", "$", "bucks", "דולר", "דולרים", "دولار", "دولارات"),
    "EUR": ("euro", "euros", "eur", "€", "אירו", "יורו", "يورو"),
}

# word -> value. "unit" words add; MULT_100 / MULT_1000 multiply what came before.
_UNITS: dict[str, int] = {}
_HUNDRED_WORDS: set[str] = set()  # "hundred": multiplies the preceding 1-99, or is 100 alone
_THOUSAND_WORDS: set[str] = set()


def _add(value: int, *words: str) -> None:
    for word in words:
        _UNITS[word] = value


# ---- English
for _i, _w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
):
    _add(_i, _w)
for _i, _w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split(), 2):
    _add(_i * 10, _w)
_HUNDRED_WORDS |= {"hundred", "hundreds"}
_THOUSAND_WORDS |= {"thousand", "thousands", "grand"}

# ---- Hebrew (masculine and feminine forms; "חמש עשרה" = 5 + 10)
_add(1, "אחד", "אחת")
_add(2, "שניים", "שתיים", "שני", "שתי", "שנים", "שתים")
_add(3, "שלוש", "שלושה")
_add(4, "ארבע", "ארבעה")
_add(5, "חמש", "חמישה", "חמשה")
_add(6, "שש", "שישה", "ששה")
_add(7, "שבע", "שבעה")
_add(8, "שמונה")
_add(9, "תשע", "תשעה")
_add(10, "עשר", "עשרה")
for _i, _w in enumerate("עשרים שלושים ארבעים חמישים שישים שבעים שמונים תשעים".split(), 2):
    _add(_i * 10, _w)
_add(60, "ששים")
_add(50, "חמשים")
_add(100, "מאה")
_add(200, "מאתיים", "מאתים")
_HUNDRED_WORDS |= {"מאות"}
_add(2000, "אלפיים")  # two thousand
_THOUSAND_WORDS |= {"אלף", "אלפים"}

# ---- Arabic, written and Levantine spoken forms (after folding, see _ARABIC_FOLD)
_add(1, "واحد", "واحده")
_add(2, "اثنين", "اثنان", "اتنين", "تنين", "ثنتين", "تنتين")
_add(3, "ثلاث", "ثلاثه", "تلات", "تلاته")
_add(4, "اربع", "اربعه")
_add(5, "خمس", "خمسه")
_add(6, "ست", "سته")
_add(7, "سبع", "سبعه")
_add(8, "ثمان", "ثماني", "ثمانيه", "تمان", "تمانه", "تماني", "تمانيه")
_add(9, "تسع", "تسعه")
_add(10, "عشر", "عشره")
for _v, _w in ((11, "حداعش احدعش احدعشر"), (12, "اطنعش اثنعش اتنعش اثناعشر"), (13, "تلطعش ثلاثطعش تلتطعش"), (14, "اربعطعش"),
               (15, "خمسطعش"), (16, "ستطعش"), (17, "سبعطعش"), (18, "تمنطعش ثمنطعش"), (19, "تسعطعش")):
    _add(_v, *_w.split())
for _v, _w in ((20, "عشرين عشرون"), (30, "ثلاثين ثلاثون تلاتين"), (40, "اربعين اربعون"), (50, "خمسين خمسون"), (60, "ستين ستون"),
               (70, "سبعين سبعون"), (80, "ثمانين ثمانون تمانين"), (90, "تسعين تسعون")):
    _add(_v, *_w.split())
_add(200, "ميتين", "مئتين", "مائتين", "مئتان", "مائتان")
for _v, _stems in ((300, "ثلاث تلات تلت ثلث"), (400, "اربع"), (500, "خمس"), (600, "ست"), (700, "سبع"), (800, "ثمان ثماني تمان تمن"), (900, "تسع")):
    for _stem in _stems.split():
        _add(_v, *(_stem + _end for _end in ("ميه", "مئه", "مائه")))
_HUNDRED_WORDS |= {"ميه", "مئه", "مائه"}  # "خمس ميه" = 5 x 100; alone = 100
_add(2000, "الفين", "الفان")
_THOUSAND_WORDS |= {"الف", "الاف", "تلاف"}

_SKIP = {"and", "a", "و", "ו"}  # may sit inside a number: "a hundred and fifty"
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?k?\b|[^\W\d_]+|[₪$€]", re.UNICODE)


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_ARABIC_DIGITS).lower()
    # drop Hebrew points and Arabic vowel marks
    text = "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")
    text = unicodedata.normalize("NFC", text).translate(_ARABIC_FOLD)
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)  # 1,200 -> 1200
    return text.replace('ש"ח', "שח").replace("ש״ח", "שח").replace("-", " ")


# Look-ups happen on normalised text, so the word lists are normalised the same way
# (for example "مائة" and "مئة" lose their hamza and final ة there).
_UNITS = {_normalise(word): value for word, value in _UNITS.items()}
_HUNDRED_WORDS = {_normalise(word) for word in _HUNDRED_WORDS}
_THOUSAND_WORDS = {_normalise(word) for word in _THOUSAND_WORDS}


def _overlaps(current: Number, value: int) -> bool:
    """Would adding ``value`` write over digits ``current`` already has? ("two hundred, three hundred")

    Then they are two numbers, not one. Small-before-tens is the exception:
    Arabic says "five and twenty", Hebrew and Arabic say "five ten" for 15.
    """
    if current % 100 < 10 and 10 <= value < 100:
        return False  # the tens place is still free
    return current % (10 ** len(str(value))) != 0


def _word_value(token: str) -> Optional[str]:
    """The lexicon key for a token, also trying it without a leading "and" (ו / و)."""
    for candidate in (token, token[1:] if token[:1] in ("ו", "و") and len(token) > 2 else ""):
        if candidate and (candidate in _UNITS or candidate in _HUNDRED_WORDS or candidate in _THOUSAND_WORDS):
            return candidate
    return None


def _numbers(tokens: list[str]) -> list[Number]:
    """Every separate number in the token list, in order."""
    found: list[Number] = []
    total: Number = 0
    current: Number = 0
    active = False
    last_was_digits = False

    def close() -> None:
        nonlocal total, current, active, last_was_digits
        if active and total + current > 0:
            found.append(total + current)
        total, current, active, last_was_digits = 0, 0, False, False

    for index, token in enumerate(tokens):
        if _NUMBER_RE.match(token):
            thousands = token.endswith("k")
            value = float(token.rstrip("k")) * (1000 if thousands else 1)
            if active and (last_was_digits or current or total):
                close()  # "150 200" are two numbers, not one
            current, active, last_was_digits = value, True, True
            continue
        key = _word_value(token)
        if key is None:
            if active and token in _SKIP and index + 1 < len(tokens) and (
                _word_value(tokens[index + 1]) is not None
            ):
                continue
            close()
            continue
        if key in _THOUSAND_WORDS:
            total += (current or 1) * 1000
            current = 0
        elif key in _HUNDRED_WORDS:
            if 0 < current < 100:
                current *= 100  # "five hundred"
            else:
                current += 100  # "hundred" on its own
        else:
            value = _UNITS[key]
            if active and (last_was_digits or (current and _overlaps(current, value))):
                close()  # "500 five", "two hundred ... three hundred": separate numbers
            current += value
        active, last_was_digits = True, False
    close()
    return found


def detect_currency(text: str) -> Optional[str]:
    normalised = _normalise(text)
    tokens = set(_TOKEN_RE.findall(normalised))
    for code, words in CURRENCIES.items():
        for word in words:
            folded = _normalise(word)
            if folded in tokens or any(t[:1] in ("ו", "و", "ב", "ب", "ל") and t[1:] == folded for t in tokens):
                return code
    return None


def parse_amount(text: str) -> tuple[Optional[Number], Optional[str]]:
    """``(amount, currency code)`` found in ``text``; either can be None.

    >>> parse_amount("about 500 shekels")
    (500, 'ILS')
    >>> parse_amount("חמש מאות שקל")
    (500, 'ILS')
    """
    if not isinstance(text, str) or not text.strip():
        return None, None
    tokens = _TOKEN_RE.findall(_normalise(text))
    values = list(dict.fromkeys(v for v in _numbers(tokens) if 0 < v <= 1e9))
    amount: Optional[Number] = None
    if len(values) == 1:
        amount = int(values[0]) if float(values[0]).is_integer() else round(float(values[0]), 2)
    return amount, detect_currency(text)
