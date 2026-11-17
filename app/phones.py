"""Read a phone number out of what the caller said, and check it is complete.

Speech-to-text gives digits ("0501234567"), digit words ("zero five zero ...",
"אפס חמש ...", "صفر خمسة ...") or a mix. This module collects the digits in
order and checks them against Israeli numbering:

* mobile:   05X followed by 7 digits  (10 digits in all)
* landline: 0X  followed by 7 digits  (9 digits; X = 2, 3, 4, 8 or 9), or 07X + 7 digits (10)
* ``+972`` / ``972`` in front is accepted and turned into the leading 0.

A dropped digit is common on a bad line, so the agent reads an incomplete
number back and asks once more (see :mod:`app.agent`).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Optional

_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_DIGIT_WORDS: dict[str, str] = {}
for _digit, _words in enumerate(
    [
        "zero oh אפס صفر",
        "one אחד אחת واحد وحده",
        "two שתיים שניים שתים اثنين اتنين تنين",
        "three שלוש שלושה ثلاثه ثلاث تلاته تلات",
        "four ארבע ארבעה اربعه اربع",
        "five חמש חמישה خمسه خمس",
        "six שש שישה سته ست",
        "seven שבע שבעה سبعه سبع",
        "eight שמונה ثمانيه ثماني تمانيه تماني تمنيه",
        "nine תשע תשעה تسعه تسع",
    ]
):
    for _word in _words.split():
        _DIGIT_WORDS[_word] = str(_digit)
_FOLD = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ة": "ه", "ى": "ي"})
_TOKEN_RE = re.compile(r"\d+|[^\W\d_]+", re.UNICODE)
_MOBILE_RE = re.compile(r"^05\d{8}$")
_LANDLINE_RE = re.compile(r"^0(?:[23489]\d{7}|7\d{8})$")


def extract_digits(text: str) -> str:
    """All digits in ``text`` in order, spoken digit words included."""
    text = unicodedata.normalize("NFKC", text or "").translate(_ARABIC_DIGITS).lower()
    text = "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")
    text = unicodedata.normalize("NFC", text).translate(_FOLD)
    digits: list[str] = []
    for token in _TOKEN_RE.findall(text):
        if token.isdigit():
            digits.append(token)
        else:
            word = token[1:] if token[:1] in ("ו", "و") and token[1:] in _DIGIT_WORDS else token
            if word in _DIGIT_WORDS:
                digits.append(_DIGIT_WORDS[word])
    number = "".join(digits)
    if number.startswith("972") and len(number) >= 11:
        number = "0" + number[3:]
    return number


def is_valid_israeli(number: str) -> bool:
    return bool(_MOBILE_RE.match(number) or _LANDLINE_RE.match(number))


def phone_value(text: Any) -> Optional[dict[str, Any]]:
    """``{"text", "number", "valid"}`` for a phone slot, or None if nothing was said."""
    if text is None or isinstance(text, (bool, list, dict)):
        return None
    clean = " ".join(str(text).split())[:60]
    if not clean or clean.lower() in {"unknown", "none", "null", "n/a"}:
        return None
    number = extract_digits(clean)
    return {"text": clean, "number": number or None, "valid": is_valid_israeli(number)}


def digit_groups(number: str) -> list[str]:
    """Split a number the way people say it: mobile 3-3-4, landline 2-3-4.

    ``0501234567`` -> ``050, 123, 4567``; ``049991234`` -> ``04, 999, 1234``.
    Anything else (a length we do not expect) goes in threes.
    """
    digits = "".join(ch for ch in number if ch.isdigit())
    if len(digits) == 10:
        cuts = (3, 6)
    elif len(digits) == 9:
        cuts = (2, 5)
    else:
        return [digits[i : i + 3] for i in range(0, len(digits), 3)]
    return [digits[: cuts[0]], digits[cuts[0] : cuts[1]], digits[cuts[1] :]]


def spoken_digits(number: str) -> str:
    """``0501`` -> ``0 5 0 1``: digits apart, so the voice reads them one by one."""
    return " ".join(number)
