"""Everything the program itself says or matches, per language.

A profile decides *what* the business says (greeting, closings, questions).
A few phrases belong to the program, not to a business: the apology when
something fails, "one moment", "is there anything else?", the neutral goodbye.
They live here, once per language, together with the rule the model is given
about how to speak that language.

Adding a language = adding one :class:`LanguagePack` here, its decline /
goodbye words in :mod:`app.intent`, number words in :mod:`app.amounts`, and
profiles that use it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .phones import digit_groups


@dataclass(frozen=True)
class LanguagePack:
    code: str
    #: English name, used in the instructions to the model.
    name: str
    apology: str
    one_moment: str
    anything_else: str
    #: Said when the caller says goodbye before we know whether we can help. ``{business}`` is filled in.
    neutral_closing: str
    #: Said when the call reaches the profile's hard time limit.
    time_limit: str
    #: Said when speech recognition returned nothing usable in this language.
    didnt_catch: str
    #: Closing when we gave up on an answer that decides the outcome: a person will call back.
    callback_closing: str
    #: How the model should speak in this language (part of the system prompt). Empty for English.
    speaking_rule: str
    #: Any wording of "is there anything else?" - only the server may ask that.
    anything_else_pattern: str
    #: The words for 0-9, for reading a phone number back digit by digit.
    digits: tuple[str, ...]
    #: A complete phone number is read back for the caller to confirm, worded by the
    #: program (``{digits}`` is filled in).
    phone_confirm: str
    #: An incomplete number is not read back: one short fixed request to say it again
    #: (pre-generated, so it starts at once and is over before the caller answers).
    phone_incomplete: str

    def read_digits(self, number: str) -> str:
        """Digit words in the groups people use (050, 123, 4567): a short pause between groups only."""
        return ", ".join(" ".join(self.digits[int(ch)] for ch in group) for group in digit_groups(number))

    def asks_anything_else(self, text: str) -> bool:
        return bool(re.search(self.anything_else_pattern, text, re.IGNORECASE))


ENGLISH = LanguagePack(
    code="en",
    name="English",
    apology="Sorry, I'm having a little trouble on my end. Could you say that again?",
    one_moment="One moment, please.",
    anything_else="Is there anything else I can help you with?",
    neutral_closing="Thank you for calling {business}. Goodbye!",
    time_limit="I'm sorry, but we've reached the time limit for this call. Thank you for calling {business}. Goodbye!",
    didnt_catch="Sorry, I didn't catch that. Could you say it again?",
    callback_closing="I'm sorry, I'm having trouble understanding. Someone from {business} will call you back. Goodbye!",
    speaking_rule="",
    anything_else_pattern=r"\b(anything|something) (else|more)\b|\bany(thing)? other (question|thing)s?\b|\bother questions\b|\bany more questions\b",
    digits=("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"),
    phone_confirm="I heard {digits}. Is that right?",
    phone_incomplete="I didn't catch the whole number. Could you say it again, slowly?",
)

HEBREW = LanguagePack(
    code="he",
    name="Hebrew",
    apology="סליחה, יש לי תקלה קטנה. אפשר לחזור על זה?",
    one_moment="רגע אחד, בבקשה.",
    anything_else="אפשר לעזור בעוד משהו?",
    neutral_closing="תודה שהתקשרתם ל{business}. להתראות!",
    time_limit="מצטערת, הגענו למגבלת הזמן של השיחה. תודה שהתקשרתם ל{business}. להתראות!",
    didnt_catch="סליחה, לא שמעתי טוב. אפשר לחזור על זה?",
    callback_closing="מצטערת, הקו לא כל כך ברור. מישהו מ{business} יחזור אליכם בטלפון. להתראות!",
    speaking_rule=(
        "Use everyday spoken Israeli Hebrew, in Hebrew letters, without nikud. "
        "The voice is a woman's: speak about yourself in the feminine. "
        "You do not know the caller's gender: prefer wording that fits anyone (plural or neutral forms). "
        "Write numbers, times and prices as Hebrew words, not digits."
    ),
    anything_else_pattern=r"עוד משהו|משהו נוסף|דבר נוסף|עוד שאלות|שאלות נוספות|משהו אחר",
    digits=("אפס", "אחת", "שתיים", "שלוש", "ארבע", "חמש", "שש", "שבע", "שמונה", "תשע"),
    phone_confirm="שמעתי {digits}. נכון?",
    phone_incomplete="לא שמעתי את כל המספר. אפשר לומר אותו שוב, לאט?",
)

ARABIC = LanguagePack(
    code="ar",
    name="Arabic",
    apology="آسفة، في عندي مشكلة صغيرة. ممكن تعيد لو سمحت؟",
    one_moment="لحظة لو سمحت.",
    anything_else="في إشي تاني ممكن أساعدك فيه؟",
    neutral_closing="شكراً لاتصالك ب{business}. مع السلامة!",
    time_limit="آسفة، لازم ننهي المكالمة هلأ. شكراً لاتصالك ب{business}. مع السلامة!",
    didnt_catch="آسفة، ما سمعت منيح. ممكن تعيد لو سمحت؟",
    callback_closing="آسفة، الصوت مش واضح عندي. رح نتصل فيك من {business}. مع السلامة!",
    speaking_rule=(
        "Use spoken Levantine Arabic as people talk on the phone in Jordan and Palestine "
        "(شو، بدي، هلأ، منيح، كتير، إشي), in Arabic letters, without tashkeel. "
        "Never use formal Modern Standard Arabic. "
        "The voice is a woman's: speak about yourself in the feminine. "
        "Write numbers, times and prices as Arabic words, not digits."
    ),
    anything_else_pattern=r"(شي|إشي|اشي|شيء|حاجة|سؤال)\s+(تاني|ثاني|تانية|ثانية|آخر|اخر)|كمان (شي|إشي|اشي)|غير هيك",
    digits=("صفر", "واحد", "اتنين", "تلاتة", "أربعة", "خمسة", "ستة", "سبعة", "تمانية", "تسعة"),
    phone_confirm="سمعت {digits}. مزبوط؟",
    phone_incomplete="ما سمعت النمرة كاملة. ممكن تعيدها كلها، شوي شوي؟",
)

PACKS: dict[str, LanguagePack] = {pack.code: pack for pack in (ENGLISH, HEBREW, ARABIC)}


def base_language(language: str) -> str:
    """``ar-JO`` -> ``ar``."""
    return (language or "en").split("-")[0].lower()


def pack(language: str) -> LanguagePack:
    """The phrases for a language code; English for anything we have no pack for."""
    return PACKS.get(base_language(language), ENGLISH)
