"""Slot filling: the facts a call must collect, and the decisions made from them.

A *slot* is one piece of information the agent needs ("does the caller own the
home?"). The language model's only jobs are to **extract** slot values from
what the caller says and to **phrase** replies. Everything that decides how
the call goes is plain code in this module, so it is predictable and testable:

* is the caller a fit?            -> :func:`compute_outcome`
* what do we still need to ask?   -> :func:`missing_slots`
* what should happen this turn?   -> :func:`make_plan`

Outcome rules:

* any slot holds one of its ``disqualifying`` values  -> ``not_fit``
* every ``required`` slot has a valid value           -> ``qualified``, or a
  profile-defined variant of it (e.g. ``qualified_no_callback``) when the
  slots match that variant's ``when`` condition
* otherwise                                           -> keep collecting
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional, Union

SLOT_TYPES = ("yes_no", "choice", "text", "number", "money")
OUTCOME_QUALIFIED = "qualified"
OUTCOME_NOT_FIT = "not_fit"
#: The caller ended the call before we knew either way.
OUTCOME_INCOMPLETE = "incomplete"

MAX_TEXT_CHARS = 120
MAX_SLOTS = 12
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
_UNKNOWN = {"", "unknown", "n/a", "na", "none", "null", "not sure", "unsure", "not given", "not provided", "?"}
_NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)?")

_CURRENCY_RE = re.compile(r"^[A-Z]{3}$")
#: Extra reply fields the model fills for a "money" slot, next to the raw text.
MONEY_AMOUNT_SUFFIX = "_amount"
MONEY_CURRENCY_SUFFIX = "_currency"
RESERVED_OUTCOMES = (OUTCOME_QUALIFIED, OUTCOME_NOT_FIT, OUTCOME_INCOMPLETE)

#: Text, a number, or for "money" slots ``{"text": ..., "amount": ..., "currency": ...}``.
SlotValue = Union[str, int, float, dict[str, Any]]


@dataclass(frozen=True)
class SlotDef:
    name: str
    type: str
    #: What to ask the caller. Slots without a question are only inferred, never asked.
    question: Optional[str] = None
    required: bool = True
    values: tuple[str, ...] = ()  # for type "choice"
    disqualifying: tuple[str, ...] = ()
    #: Why a disqualifying value means we cannot help (given to the model to phrase kindly).
    not_fit_reason: str = ""
    #: Hint for extraction: what counts as which value.
    description: str = ""
    #: For type "money": the currency assumed when the caller names none (ISO code).
    currency: str = ""

    @property
    def allowed(self) -> Optional[tuple[str, ...]]:
        """Closed set of values, or None for free text / numbers / money."""
        if self.type == "yes_no":
            return ("yes", "no")
        if self.type == "choice":
            return self.values
        return None


def parse_slots(data: Any) -> tuple[tuple[SlotDef, ...], list[str]]:
    """Validate the ``slots`` section of a profile. Returns (slots, problems)."""
    problems: list[str] = []
    if data is None:
        return (), ["missing required field 'slots'"]
    if not isinstance(data, list) or not data:
        return (), ["'slots' must be a non-empty list"]
    if len(data) > MAX_SLOTS:
        problems.append(f"'slots' has {len(data)} entries; keep it to {MAX_SLOTS} or fewer for a phone call")
    slots: list[SlotDef] = []
    seen: set[str] = set()
    known_keys = {"name", "type", "question", "required", "values", "disqualifying", "not_fit_reason", "description", "currency"}
    for index, raw in enumerate(data, 1):
        where = f"slot #{index}"
        if not isinstance(raw, dict):
            problems.append(f"{where} must be a mapping of fields")
            continue
        name = raw.get("name")
        if not isinstance(name, str) or not _NAME_RE.match(name):
            problems.append(f"{where}: 'name' must be lowercase letters, digits and underscores (got {name!r})")
            continue
        where = f"slot '{name}'"
        if name in seen:
            problems.append(f"{where} is defined twice")
            continue
        seen.add(name)
        unknown = sorted(str(k) for k in raw if k not in known_keys)
        if unknown:
            problems.append(f"{where}: unknown field(s): {', '.join(unknown)}")
        kind = raw.get("type")
        if kind not in SLOT_TYPES:
            problems.append(f"{where}: 'type' must be one of {', '.join(SLOT_TYPES)} (got {kind!r})")
            continue
        required = raw.get("required", True)
        if not isinstance(required, bool):
            problems.append(f"{where}: 'required' must be true or false")
            required = True
        question = raw.get("question")
        if question is not None and (not isinstance(question, str) or not question.strip()):
            problems.append(f"{where}: 'question' must be non-empty text")
            question = None
        if required and not question:
            problems.append(f"{where}: a required slot needs a 'question' to ask the caller")

        values = raw.get("values", [])
        if kind == "choice":
            if not isinstance(values, list) or len(values) < 2 or any(not isinstance(v, str) or not v.strip() for v in values):
                problems.append(f"{where}: a 'choice' slot needs 'values' with at least two text options")
                values = []
        elif values:
            problems.append(f"{where}: 'values' is only allowed for type 'choice'")
            values = []
        clean_values = tuple(v.strip().lower() for v in values)

        slot_allowed = ("yes", "no") if kind == "yes_no" else clean_values if kind == "choice" else None
        disqualifying = raw.get("disqualifying", [])
        if not isinstance(disqualifying, list) or any(not isinstance(v, str) for v in disqualifying):
            problems.append(f"{where}: 'disqualifying' must be a list of values")
            disqualifying = []
        clean_disq = tuple(v.strip().lower() for v in disqualifying)
        if clean_disq:
            if slot_allowed is None:
                problems.append(f"{where}: 'disqualifying' needs a 'yes_no' or 'choice' slot")
            else:
                bad = [v for v in clean_disq if v not in slot_allowed]
                if bad:
                    problems.append(f"{where}: disqualifying value(s) {', '.join(bad)} are not allowed values ({', '.join(slot_allowed)})")
                if set(clean_disq) >= set(slot_allowed):
                    problems.append(f"{where}: every allowed value is disqualifying, so nobody could ever qualify")
        reason = raw.get("not_fit_reason", "")
        if not isinstance(reason, str):
            problems.append(f"{where}: 'not_fit_reason' must be text")
            reason = ""
        if clean_disq and not reason.strip():
            problems.append(f"{where}: 'not_fit_reason' is required when 'disqualifying' is set")
        description = raw.get("description", "")
        if not isinstance(description, str):
            problems.append(f"{where}: 'description' must be text")
            description = ""
        currency = raw.get("currency", "")
        if kind != "money":
            if currency:
                problems.append(f"{where}: 'currency' is only allowed for type 'money'")
            currency = ""
        elif not isinstance(currency, str) or not _CURRENCY_RE.match(currency):
            problems.append(f"{where}: a 'money' slot needs 'currency', a three-letter code such as ILS or USD")
            currency = ""
        slots.append(
            SlotDef(
                name=name,
                type=str(kind),
                question=" ".join(question.split()) if question else None,
                required=required,
                values=clean_values,
                disqualifying=clean_disq,
                not_fit_reason=" ".join(reason.split()),
                description=" ".join(description.split()),
                currency=currency,
            )
        )
    for slot in slots:
        if slot.type == "money":
            for extra in (slot.name + MONEY_AMOUNT_SUFFIX, slot.name + MONEY_CURRENCY_SUFFIX):
                if extra in seen:
                    problems.append(f"slot '{extra}' clashes with a field of the money slot '{slot.name}'; rename it")
    if slots and not any(s.required for s in slots):
        problems.append("at least one slot must be required, otherwise every caller qualifies immediately")
    return tuple(slots), problems


# --------------------------------------------------------------------------- values
def normalise_value(slot: SlotDef, raw: Any) -> Optional[SlotValue]:
    """A valid value for ``slot``, or None if ``raw`` is unknown or not acceptable."""
    if slot.type == "money":
        if isinstance(raw, dict):
            return money_value(slot, raw.get("text"), raw.get("amount"), raw.get("currency"))
        return money_value(slot, raw, None, None)
    if raw is None or isinstance(raw, (list, dict)):
        return None
    if slot.type == "number":
        if isinstance(raw, bool):
            return None
        if isinstance(raw, (int, float)):
            return raw
        match = _NUMBER_RE.search(str(raw))
        if match is None:
            return None
        number = float(match.group(0).replace(",", "."))
        return int(number) if number.is_integer() else number
    if isinstance(raw, bool):
        text = "yes" if raw else "no"
    else:
        text = " ".join(str(raw).split())
    if text.lower() in _UNKNOWN:
        return None
    allowed = slot.allowed
    if allowed is not None:
        lowered = text.lower()
        if slot.type == "yes_no":
            lowered = {"true": "yes", "false": "no", "y": "yes", "n": "no"}.get(lowered, lowered)
        return lowered if lowered in allowed else None
    return text[:MAX_TEXT_CHARS]


def _amount(raw: Any) -> Optional[Union[int, float]]:
    if raw is None or isinstance(raw, (bool, list, dict)):
        return None
    if not isinstance(raw, (int, float)):
        match = _NUMBER_RE.search(str(raw).replace(",", ""))
        if match is None:
            return None
        raw = float(match.group(0))
    if raw != raw or raw <= 0 or raw > 1e9:  # NaN, zero/negative or absurd
        return None
    return int(raw) if float(raw).is_integer() else round(float(raw), 2)


def money_value(slot: SlotDef, text: Any, amount: Any, currency: Any) -> Optional[dict[str, Any]]:
    """``{"text", "amount", "currency"}`` for a money slot, or None if nothing usable was said.

    ``text`` is what the caller said and is always kept. ``amount`` and
    ``currency`` are filled when the model could turn it into a number; the
    slot's own currency is assumed if the caller named none.
    """
    clean_text = None
    if text is not None and not isinstance(text, (bool, list, dict)):
        clean_text = " ".join(str(text).split())[:MAX_TEXT_CHARS]
        if clean_text.lower() in _UNKNOWN:
            clean_text = None
    number = _amount(amount)
    if clean_text is None and number is None:
        return None
    code = None
    if number is not None:
        code = str(currency).strip().upper() if isinstance(currency, str) else ""
        if not _CURRENCY_RE.match(code):
            code = slot.currency or None
    return {"text": clean_text if clean_text is not None else f"{number} {code or ''}".strip(), "amount": number, "currency": code}


def display_value(value: Any) -> str:
    """One slot value as short text for logs."""
    if isinstance(value, dict):
        if value.get("amount") is not None:
            return f"{value['amount']} {value.get('currency') or '?'} (\"{value.get('text')}\")"
        return f"\"{value.get('text')}\" (amount not understood)"
    return str(value)


def empty_slots(defs: tuple[SlotDef, ...]) -> dict[str, Optional[SlotValue]]:
    return {slot.name: None for slot in defs}


def merge_slots(
    defs: tuple[SlotDef, ...], current: dict[str, Optional[SlotValue]], update: Any
) -> tuple[dict[str, Optional[SlotValue]], list[str]]:
    """Apply the model's extracted values. Returns (new slots, rejected entries).

    A valid new value replaces the old one (callers correct themselves); an
    unknown/null value never erases something we already know. Anything the
    model invents - unknown slot names, values outside the allowed set - is
    rejected and reported, never stored.
    """
    merged = dict(current)
    rejected: list[str] = []
    if not isinstance(update, dict):
        return merged, ([] if update is None else ["slots is not an object"])
    by_name = {slot.name: slot for slot in defs}
    money_extras = {
        slot.name + suffix
        for slot in defs
        if slot.type == "money"
        for suffix in (MONEY_AMOUNT_SUFFIX, MONEY_CURRENCY_SUFFIX)
    }
    for name, raw in update.items():
        if name in money_extras:
            continue  # read together with the money slot itself
        slot = by_name.get(str(name))
        if slot is None:
            rejected.append(f"{name}: unknown slot")
            continue
        if slot.type == "money" and not isinstance(raw, dict):
            value: Optional[SlotValue] = money_value(
                slot, raw, update.get(slot.name + MONEY_AMOUNT_SUFFIX), update.get(slot.name + MONEY_CURRENCY_SUFFIX)
            )
            old = merged.get(slot.name)
            if (
                isinstance(value, dict)
                and isinstance(old, dict)
                and value["amount"] is None
                and old.get("amount") is not None
                and value["text"] == old.get("text")
            ):
                continue  # same words as before; keep the number we already understood
        else:
            value = normalise_value(slot, raw)
        if value is None:
            if raw is not None and str(raw).strip().lower() not in _UNKNOWN:
                rejected.append(f"{name}: {raw!r} is not a valid value")
            continue
        merged[slot.name] = value
    return merged, rejected


# --------------------------------------------------------------------------- decisions
def disqualifier(defs: tuple[SlotDef, ...], slots: dict[str, Optional[SlotValue]]) -> Optional[SlotDef]:
    """The first slot whose value rules the caller out, if any."""
    for slot in defs:
        if slot.disqualifying and slots.get(slot.name) in slot.disqualifying:
            return slot
    return None


def missing_slots(defs: tuple[SlotDef, ...], slots: dict[str, Optional[SlotValue]]) -> list[SlotDef]:
    """Required slots we still have to ask about, in profile order."""
    return [s for s in defs if s.required and s.question and slots.get(s.name) is None]


@dataclass(frozen=True)
class OutcomeVariant:
    """A named flavour of "qualified", e.g. a valid lead who does not want a call back."""

    name: str
    #: Slot values that select this variant (all must match).
    when: tuple[tuple[str, str], ...]
    #: What the model is told when it announces this outcome to the caller.
    announce: str
    #: The fixed closing line for this outcome.
    closing: str

    def matches(self, slots: dict[str, Optional[SlotValue]]) -> bool:
        return all(slots.get(name) == value for name, value in self.when)


def parse_outcomes(data: Any, defs: tuple[SlotDef, ...]) -> tuple[tuple[OutcomeVariant, ...], list[str]]:
    """Validate the optional ``outcomes`` section of a profile."""
    if data is None:
        return (), []
    if not isinstance(data, dict):
        return (), ["'outcomes' must be a mapping of outcome name to its settings"]
    by_name = {slot.name: slot for slot in defs}
    variants: list[OutcomeVariant] = []
    problems: list[str] = []
    for name, raw in data.items():
        where = f"outcome '{name}'"
        if not isinstance(name, str) or not _NAME_RE.match(name):
            problems.append(f"{where}: the name must be lowercase letters, digits and underscores")
            continue
        if name in RESERVED_OUTCOMES:
            problems.append(f"{where}: this name is built in; choose another, such as '{name}_other'")
            continue
        if not isinstance(raw, dict):
            problems.append(f"{where} must be a mapping with 'when', 'announce' and 'closing'")
            continue
        unknown = sorted(str(k) for k in raw if k not in ("when", "announce", "closing"))
        if unknown:
            problems.append(f"{where}: unknown field(s): {', '.join(unknown)}")
        when = raw.get("when")
        conditions: list[tuple[str, str]] = []
        if not isinstance(when, dict) or not when:
            problems.append(f"{where}: 'when' must map at least one slot to a value")
        else:
            for slot_name, value in when.items():
                slot = by_name.get(str(slot_name))
                if isinstance(value, bool):
                    value = "yes" if value else "no"
                clean = str(value).strip().lower()
                if slot is None:
                    problems.append(f"{where}: 'when' refers to unknown slot '{slot_name}'")
                elif slot.allowed is None:
                    problems.append(f"{where}: 'when' needs a 'yes_no' or 'choice' slot, and '{slot_name}' is neither")
                elif clean not in slot.allowed:
                    problems.append(f"{where}: '{clean}' is not an allowed value of '{slot_name}' ({', '.join(slot.allowed)})")
                elif clean in slot.disqualifying:
                    problems.append(f"{where}: '{slot_name}={clean}' is disqualifying, so this outcome could never happen")
                else:
                    conditions.append((slot.name, clean))
        texts: dict[str, str] = {}
        for key, limit in (("announce", 400), ("closing", 400)):
            value = raw.get(key)
            if not isinstance(value, str) or not value.strip():
                problems.append(f"{where}: '{key}' must be non-empty text")
            elif len(value) > limit:
                problems.append(f"{where}: '{key}' is too long ({len(value)} > {limit} characters)")
            else:
                texts[key] = " ".join(value.split())
        if conditions and len(texts) == 2:
            variants.append(OutcomeVariant(name, tuple(conditions), texts["announce"], texts["closing"]))
    return tuple(variants), problems


def compute_outcome(
    defs: tuple[SlotDef, ...],
    slots: dict[str, Optional[SlotValue]],
    variants: tuple[OutcomeVariant, ...] = (),
) -> Optional[str]:
    if disqualifier(defs, slots) is not None:
        return OUTCOME_NOT_FIT
    if not missing_slots(defs, slots):
        for variant in variants:
            if variant.matches(slots):
                return variant.name
        return OUTCOME_QUALIFIED
    return None


COLLECT = "collect"  # ask for the next missing slot
ANNOUNCE = "announce"  # outcome just decided: tell the caller, then ask "anything else?"
ANSWER = "answer"  # outcome already announced: just answer; the caller ends the call
FINAL = "final"  # the caller said goodbye: extract what they said, say nothing; code closes the call


@dataclass(frozen=True)
class Plan:
    """What the code has decided this turn should do."""

    kind: str
    outcome: Optional[str] = None
    target: Optional[SlotDef] = None  # collect: the slot to ask for now
    next: Optional[SlotDef] = None  # collect: the slot to ask if `target` gets answered
    reason: Optional[SlotDef] = None  # announce not_fit: the slot that rules the caller out
    note: Optional[str] = None  # announce a qualified variant: what to tell the caller


def make_plan(
    defs: tuple[SlotDef, ...],
    slots: dict[str, Optional[SlotValue]],
    announced: Optional[str],
    variants: tuple[OutcomeVariant, ...] = (),
) -> Plan:
    """Decide this turn from the slots and from what the caller was already told."""
    outcome = compute_outcome(defs, slots, variants)
    if outcome is None:
        missing = missing_slots(defs, slots)
        return Plan(COLLECT, target=missing[0], next=missing[1] if len(missing) > 1 else None)
    if announced != outcome:
        note = next((v.announce for v in variants if v.name == outcome), None)
        return Plan(ANNOUNCE, outcome=outcome, reason=disqualifier(defs, slots), note=note)
    return Plan(ANSWER, outcome=outcome)


def plans_consistent(before: Plan, after: Plan) -> bool:
    """Is a reply written under ``before`` still right now that we know more (``after``)?

    The model is briefed before it has read the caller's answer. A collect
    reply stays valid if the next slot to ask is the one it was told to ask
    now, or the one it was told to ask if that got answered. Any change of
    outcome, or the caller answering several things at once, needs a new reply.
    """
    if before.kind != after.kind or before.outcome != after.outcome:
        return False
    if before.kind == COLLECT:
        return after.target in (before.target, before.next)
    return True


# --------------------------------------------------------------------------- structured output
def reply_schema(defs: tuple[SlotDef, ...]) -> dict[str, Any]:
    """JSON schema for the model's reply: ``{"slots": {...}, "say": "..."}``.

    Written for strict structured output: every property required, no extra
    properties, unknown values expressed as null. ``slots`` comes first so the
    facts are extracted before the sentence is phrased.
    """
    properties: dict[str, Any] = {}
    for slot in defs:
        allowed = slot.allowed
        if allowed is not None:
            properties[slot.name] = {"type": ["string", "null"], "enum": [*allowed, None]}
        elif slot.type == "number":
            properties[slot.name] = {"type": ["number", "null"]}
        elif slot.type == "money":
            properties[slot.name] = {"type": ["string", "null"]}
            properties[slot.name + MONEY_AMOUNT_SUFFIX] = {"type": ["number", "null"]}
            properties[slot.name + MONEY_CURRENCY_SUFFIX] = {"type": ["string", "null"]}
        else:
            properties[slot.name] = {"type": ["string", "null"]}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["slots", "say"],
        "properties": {
            "slots": {
                "type": "object",
                "additionalProperties": False,
                "required": list(properties),
                "properties": properties,
            },
            "say": {"type": "string"},
        },
    }
