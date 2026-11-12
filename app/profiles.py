"""Business profiles: one YAML file per (fictional) business the agent can play.

A profile is data, not code: swapping ``ACTIVE_PROFILE`` changes who the agent
is without touching the program. Files are validated on load so a typo fails
at start-up with a readable message instead of mid-call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .slots import OutcomeVariant, SlotDef, parse_outcomes, parse_slots

DEFAULT_MAX_CALL_SECONDS = 300
MIN_CALL_SECONDS = 30
MAX_CALL_SECONDS = 3600

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_]{0,62}$")
_TEXT_FIELDS = (
    "business_name",
    "agent_role",
    "goal",
    "greeting",
    "knowledge",
    "tone",
    "closing_qualified",
    "closing_not_fit",
)
_ALL_FIELDS = set(_TEXT_FIELDS) | {"slots", "outcomes", "max_call_seconds", "keyterms", "language", "stt", "tts"}
_LANGUAGE_RE = re.compile(r"^[a-z]{2,3}(-[A-Z]{2})?$")
_PROVIDER_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_MAX_LEN = {"greeting": 400, "closing_qualified": 400, "closing_not_fit": 400, "knowledge": 6000}
MAX_KEYTERMS = 50
MAX_KEYTERM_CHARS = 50


class ProfileError(ValueError):
    """A profile is missing, unreadable or invalid. The message is user-facing."""


@dataclass(frozen=True)
class SttConfig:
    """Which speech-to-text adapter hears the caller. Empty values mean "the default"."""

    provider: str = ""  # adapter name, see app/providers.py
    model: str = ""
    language: str = ""  # language code sent to the service, if it differs from the profile's


@dataclass(frozen=True)
class TtsConfig:
    """Which text-to-speech adapter speaks. Empty values mean "the default"."""

    provider: str = ""
    voice: str = ""


@dataclass(frozen=True)
class Profile:
    name: str
    business_name: str
    agent_role: str
    goal: str
    greeting: str
    #: The facts to collect and the rules that decide the outcome (see app/slots.py).
    slots: tuple[SlotDef, ...]
    knowledge: str
    tone: str
    #: Said when the caller's request can go ahead (e.g. "an advisor will call you back").
    closing_qualified: str
    #: Said when we cannot help this caller; must not promise any follow-up.
    closing_not_fit: str
    max_call_seconds: int = DEFAULT_MAX_CALL_SECONDS
    #: Domain words sent to speech-to-text so they are recognised more reliably.
    keyterms: tuple[str, ...] = ()
    #: Named variants of "qualified", each with its own announcement and closing line.
    outcomes: tuple[OutcomeVariant, ...] = ()
    #: The language of the call, e.g. "en", "he", "ar".
    language: str = "en"
    stt: SttConfig = SttConfig()
    tts: TtsConfig = TtsConfig()


def validate_name(name: str) -> str:
    """Profile names become file names, so only allow a safe character set."""
    if not _NAME_RE.match(name or ""):
        raise ProfileError(
            f"invalid profile name {name!r}: use lowercase letters, digits and underscores only"
        )
    return name


def parse_profile(name: str, data: Any) -> Profile:
    """Validate already-parsed YAML and build a :class:`Profile`."""
    where = f"profile '{name}'"
    if not isinstance(data, dict):
        raise ProfileError(f"{where}: the file must contain a mapping of fields, got {type(data).__name__}")

    problems: list[str] = []
    unknown = sorted(str(k) for k in data if k not in _ALL_FIELDS)
    if unknown:
        problems.append(f"unknown field(s): {', '.join(unknown)}")

    text: dict[str, str] = {}
    for key in _TEXT_FIELDS:
        value = data.get(key)
        if value is None:
            problems.append(f"missing required field '{key}'")
        elif not isinstance(value, str) or not value.strip():
            problems.append(f"'{key}' must be non-empty text")
        elif len(value) > _MAX_LEN.get(key, 600):
            problems.append(f"'{key}' is too long ({len(value)} > {_MAX_LEN.get(key, 600)} characters)")
        else:
            text[key] = " ".join(value.split()) if key != "knowledge" else value.strip()

    slots, slot_problems = parse_slots(data.get("slots"))
    problems += slot_problems
    outcomes, outcome_problems = parse_outcomes(data.get("outcomes"), slots)
    problems += outcome_problems

    seconds = data.get("max_call_seconds", DEFAULT_MAX_CALL_SECONDS)
    if isinstance(seconds, bool) or not isinstance(seconds, int):
        problems.append("'max_call_seconds' must be a whole number")
    elif not MIN_CALL_SECONDS <= seconds <= MAX_CALL_SECONDS:
        problems.append(f"'max_call_seconds' must be between {MIN_CALL_SECONDS} and {MAX_CALL_SECONDS}, got {seconds}")

    keyterms = data.get("keyterms", [])
    clean_keyterms: tuple[str, ...] = ()
    if not isinstance(keyterms, list):
        problems.append("'keyterms' must be a list of words or short phrases")
    elif any(not isinstance(k, str) or not k.strip() for k in keyterms):
        problems.append("every entry in 'keyterms' must be non-empty text")
    elif len(keyterms) > MAX_KEYTERMS:
        problems.append(f"'keyterms' has {len(keyterms)} entries; the limit is {MAX_KEYTERMS}")
    elif any(len(k) > MAX_KEYTERM_CHARS or "," in k or ";" in k for k in keyterms):
        problems.append(f"each keyterm must be at most {MAX_KEYTERM_CHARS} characters without commas or semicolons")
    else:
        clean_keyterms = tuple(dict.fromkeys(" ".join(k.split()) for k in keyterms))

    language = data.get("language", "en")
    if not isinstance(language, str) or not _LANGUAGE_RE.match(language):
        problems.append("'language' must be a language code such as en, he or ar-JO")
        language = "en"
    stt_fields = _provider_section(data, "stt", ("provider", "model", "language"), problems)
    tts_fields = _provider_section(data, "tts", ("provider", "voice"), problems)

    if problems:
        raise ProfileError(f"{where} is invalid: " + "; ".join(problems))
    return Profile(
        name=name,
        slots=slots,
        outcomes=outcomes,
        max_call_seconds=seconds,
        keyterms=clean_keyterms,
        language=language,
        stt=SttConfig(**stt_fields),
        tts=TtsConfig(**tts_fields),
        **text,
    )


def _provider_section(data: dict[str, Any], key: str, allowed: tuple[str, ...], problems: list[str]) -> dict[str, str]:
    """Read the optional ``stt`` / ``tts`` mapping of a profile."""
    section = data.get(key)
    if section is None:
        return {}
    if not isinstance(section, dict):
        problems.append(f"'{key}' must be a mapping, for example {{provider: ...}}")
        return {}
    unknown = sorted(str(k) for k in section if k not in allowed)
    if unknown:
        problems.append(f"'{key}': unknown field(s): {', '.join(unknown)} (allowed: {', '.join(allowed)})")
    fields: dict[str, str] = {}
    for name in allowed:
        value = section.get(name)
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip() or len(value) > 80:
            problems.append(f"'{key}.{name}' must be short non-empty text")
        elif name == "provider" and not _PROVIDER_RE.match(value):
            problems.append(f"'{key}.provider' must be an adapter name in lowercase, such as flux or aura")
        else:
            fields[name] = value.strip()
    return fields


def load_profile(name: str, directory: Path) -> Profile:
    """Read and validate ``<directory>/<name>.yaml``."""
    validate_name(name)
    path = directory / f"{name}.yaml"
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        available = ", ".join(sorted(p.stem for p in directory.glob("*.yaml"))) or "none"
        raise ProfileError(f"profile '{name}' not found at {path} (available: {available})") from None
    except OSError as exc:
        raise ProfileError(f"profile '{name}': cannot read {path}: {exc}") from None
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ProfileError(f"profile '{name}': {path} is not valid YAML: {exc}") from None
    return parse_profile(name, data)
