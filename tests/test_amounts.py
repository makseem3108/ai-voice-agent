"""Amounts of money read by code from what the caller said, in English, Hebrew and Arabic."""

from __future__ import annotations

from typing import Optional

import pytest

from app.amounts import detect_currency, parse_amount


@pytest.mark.parametrize(
    "phrase, amount, currency",
    [
        # ---- English
        ("500 shekels", 500, "ILS"),
        ("about five hundred shekels per month", 500, "ILS"),
        ("over five hundred checkups per month", 500, None),  # the mishearing from the real call: number kept, currency unknown
        ("1,200 NIS", 1200, "ILS"),
        ("twelve hundred dollars", 1200, "USD"),
        ("a hundred and fifty euros", 150, "EUR"),
        ("three thousand two hundred", 3200, None),
        ("twenty-five", 25, None),
        ("about 300 a month", 300, None),
        ("450.50", 450.5, None),
        ("2k", 2000, None),
        ("$120", 120, "USD"),
        # ---- Hebrew
        ("חמש מאות שקל", 500, "ILS"),
        ("בערך 500 שקלים בחודש", 500, "ILS"),
        ("מאה וחמישים ש\"ח", 150, "ILS"),
        ("אלף מאתיים שקלים", 1200, "ILS"),
        ("אלפיים שקל", 2000, "ILS"),
        ("שלוש מאות עשרים וחמש", 325, None),
        ("חמש עשרה", 15, None),
        ("שמונים דולר", 80, "USD"),
        ("₪800", 800, "ILS"),
        # ---- Arabic (written and Levantine spoken forms)
        ("خمسمية شيكل", 500, "ILS"),
        ("خمس مية شيكل", 500, "ILS"),
        ("خمسمائة شيكل", 500, "ILS"),
        ("حوالي ٥٠٠ شيكل بالشهر", 500, "ILS"),
        ("مية وخمسين دولار", 150, "USD"),
        ("ألف وخمسمية", 1500, None),
        ("ألفين شيكل", 2000, "ILS"),
        ("تلتمية وعشرين شيكل", 320, "ILS"),
        ("خمسة عشر", 15, None),
        ("خمسة وعشرين دينار", 25, None),  # Arabic says the unit first; dinar is not a currency we map
        ("مئة وخمسة وعشرين شيقل", 125, "ILS"),
        ("ميتين يورو", 200, "EUR"),
    ],
)
def test_amount_phrases(phrase: str, amount: float, currency: Optional[str]) -> None:
    assert parse_amount(phrase) == (amount, currency)


@pytest.mark.parametrize(
    "phrase",
    ["quite a lot", "I don't know", "לא יודע", "ما بعرف", "", "   ", "between 400 and 500", "150 or 200 shekels", "بين ميتين وتلتمية"],
)
def test_no_single_number_means_no_amount(phrase: str) -> None:
    assert parse_amount(phrase)[0] is None  # never guess between two numbers


def test_same_number_said_twice_is_fine_and_types_are_tidy() -> None:
    assert parse_amount("500, yes 500 shekels") == (500, "ILS")
    amount, _ = parse_amount("500")
    assert isinstance(amount, int)
    assert parse_amount(None) == (None, None)  # type: ignore[arg-type]
    assert parse_amount("0 shekels")[0] is None


def test_currency_words() -> None:
    assert detect_currency("in shekels") == "ILS" and detect_currency("בשקלים") == "ILS"
    assert detect_currency("بالدولار") is None  # glued article forms are not guessed
    assert detect_currency("دولار") == "USD" and detect_currency("euros") == "EUR"
    assert detect_currency("no currency here") is None
