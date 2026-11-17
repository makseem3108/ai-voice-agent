#!/usr/bin/env python3
"""Compare speech-to-text services on one recorded test call, sentence by sentence.

    .venv/bin/python scripts/stt_sentences.py --latest --dry-run     # cut only, sends nothing
    .venv/bin/python scripts/stt_sentences.py --latest               # the comparison
    .venv/bin/python scripts/stt_sentences.py data/recordings/<call-uuid>.wav --keyterm موعد

The test call: read the sentences of scripts/stt_sentences_ar.txt to the Arabic
line, with a pause of about two seconds after each one (recording must be on:
DEBUG_RECORD_AUDIO=true). This script then

1. cuts the recording (caller audio only) into sentences at the pauses,
2. sends every piece to three recognisers - Deepgram Nova-3 Arabic, the same
   with the profile's keyterms, and Azure Speech ar-IL,
3. prints, per sentence, what was expected and what each one heard, with a
   similarity score (1.00 = the same letters).

It runs offline, after the call: nothing here touches the phone line or the
language model. Cost: Deepgram is billed for the speech sent, twice (two
settings); Azure for the same speech once. --dry-run shows those seconds first.
Keys are read from .env and only ever sent in request headers.
"""

from __future__ import annotations

import argparse
import asyncio
import audioop
import difflib
import io
import re
import sys
import unicodedata
import wave
from pathlib import Path
from typing import Optional

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import load_settings  # noqa: E402
from app.profiles import load_profile  # noqa: E402

RATE = 8000  # a call recording: 8 kHz, 16-bit, mono
FRAME = 320  # bytes in 20 ms
DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"
AZURE_URL = "https://{region}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1"
AZURE_MAX_SECONDS = 55.0  # the short-audio endpoint takes up to 60 s


# ----------------------------------------------------------------- cutting into sentences
def read_recording(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, RATE):
            raise SystemExit(f"{path}: expected 8 kHz 16-bit mono (a call recording)")
        return wav.readframes(wav.getnframes())


def split_sentences(audio: bytes, *, min_gap_s: float = 1.0, min_speech_s: float = 0.25, pad_s: float = 0.3, min_rms: int = 250) -> list[tuple[float, float]]:
    """(start, end) in seconds of every stretch of speech separated by a pause of ``min_gap_s`` or more.

    Loudness is measured per 20 ms. "Speech" is anything clearly above the
    quietest part of the recording (three times its usual level, and at least
    ``min_rms``). Stretches shorter than ``min_speech_s`` are clicks, not words.
    """
    levels = [audioop.rms(audio[i : i + FRAME], 2) for i in range(0, len(audio) - FRAME + 1, FRAME)]
    if not levels:
        return []
    quiet = sorted(levels)[len(levels) // 5]  # the 20th percentile: line noise
    threshold = max(min_rms, quiet * 3)
    spans: list[list[int]] = []
    for index, level in enumerate(levels):
        if level < threshold:
            continue
        if spans and (index - spans[-1][1]) * 0.02 < min_gap_s:
            spans[-1][1] = index + 1
        else:
            spans.append([index, index + 1])
    total = len(levels) * 0.02
    result = []
    for first, last in spans:
        loud = sum(1 for level in levels[first:last] if level >= threshold) * 0.02
        if loud >= min_speech_s:
            result.append((max(0.0, first * 0.02 - pad_s), min(total, last * 0.02 + pad_s)))
    return result


def piece(audio: bytes, start: float, end: float) -> bytes:
    return audio[int(start * RATE) * 2 : int(end * RATE) * 2]


def wav_bytes(pcm: bytes, rate: int = RATE) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm)
    return buffer.getvalue()


# ----------------------------------------------------------------- comparing text
_FOLD = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ة": "ه", "ى": "ي", "ـ": None})
_EASTERN_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def normalise(text: str) -> str:
    """Letters and digits only, Arabic spelling variants folded, so punctuation does not count as a difference."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).translate(_FOLD).translate(_EASTERN_DIGITS)
    return " ".join(re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE).lower().split())


def similarity(expected: str, heard: str) -> float:
    """0.0 - 1.0: how much of the expected sentence came back, letter by letter."""
    a, b = normalise(expected), normalise(heard)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def read_sentences(path: Path) -> list[str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line and not line.startswith("#")]


# ----------------------------------------------------------------- the three recognisers
def _problem(response: httpx.Response) -> str:
    # Status only: an error body or URL is never printed (it could carry account details).
    return f"ERROR: HTTP {response.status_code}"


async def deepgram(client: httpx.AsyncClient, key: str, pcm: bytes, language: str, keyterms: tuple[str, ...]) -> str:
    params: list[tuple[str, str]] = [("model", "nova-3"), ("language", language), ("smart_format", "false")]
    params += [("keyterm", term) for term in keyterms]
    try:
        response = await client.post(
            DEEPGRAM_URL, params=params, content=wav_bytes(pcm), headers={"Authorization": f"Token {key}", "Content-Type": "audio/wav"}
        )
    except httpx.HTTPError as exc:
        return f"ERROR: {type(exc).__name__}"
    if response.status_code != 200:
        return _problem(response)
    try:
        return str(response.json()["results"]["channels"][0]["alternatives"][0]["transcript"])
    except (ValueError, KeyError, IndexError, TypeError):
        return "ERROR: unreadable reply"


async def azure(client: httpx.AsyncClient, key: str, region: str, pcm: bytes, locale: str) -> str:
    if len(pcm) / 2 / RATE > AZURE_MAX_SECONDS:
        return f"ERROR: piece longer than {AZURE_MAX_SECONDS:.0f} s (pause longer between sentences, or lower --gap)"
    wide, _ = audioop.ratecv(pcm, 2, 1, RATE, 16000, None)  # the endpoint expects 16 kHz
    try:
        response = await client.post(
            AZURE_URL.format(region=region),
            params={"language": locale, "format": "simple", "profanity": "raw"},
            content=wav_bytes(wide, 16000),
            headers={"Ocp-Apim-Subscription-Key": key, "Content-Type": "audio/wav; codecs=audio/pcm; samplerate=16000", "Accept": "application/json"},
        )
    except httpx.HTTPError as exc:
        return f"ERROR: {type(exc).__name__}"
    if response.status_code != 200:
        return _problem(response)
    try:
        data = response.json()
    except ValueError:
        return "ERROR: unreadable reply"
    if data.get("RecognitionStatus") != "Success":
        return f"({data.get('RecognitionStatus') or 'nothing recognised'})"
    return str(data.get("DisplayText") or "")


# ----------------------------------------------------------------- main
async def main() -> int:
    parser = argparse.ArgumentParser(description="Compare speech-to-text services on one recorded test call, sentence by sentence.")
    parser.add_argument("recording", nargs="?", help="a call recording (data/recordings/<uuid>.wav)")
    parser.add_argument("--latest", action="store_true", help="use the newest recording")
    parser.add_argument("--sentences", default=str(ROOT / "scripts" / "stt_sentences_ar.txt"), help="the sentences that were read, one per line")
    parser.add_argument("--profile", default="clinic_ar", help="take the keyterms from this profile (default clinic_ar)")
    parser.add_argument("--keyterm", action="append", default=[], help="an extra keyterm; repeat for several")
    parser.add_argument("--language", default="ar", help="Deepgram language (default ar)")
    parser.add_argument("--azure-locale", default="ar-IL", help="Azure Speech locale (default ar-IL)")
    parser.add_argument("--gap", type=float, default=1.0, help="a pause this long (seconds) separates two sentences (default 1.0)")
    parser.add_argument("--dry-run", action="store_true", help="only cut the recording and show what would be sent")
    args = parser.parse_args()

    settings = load_settings(ROOT / ".env")
    if args.latest:
        files = sorted(settings.recordings_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime)
        if not files:
            print(f"no recordings in {settings.recordings_dir} (is DEBUG_RECORD_AUDIO on?)")
            return 2
        path = files[-1]
    elif args.recording:
        path = Path(args.recording)
    else:
        parser.error("give a recording or --latest")
    audio = read_recording(path)
    expected = read_sentences(Path(args.sentences))
    spans = split_sentences(audio, min_gap_s=args.gap)
    speech = sum(end - start for start, end in spans)
    keyterms = tuple(dict.fromkeys([*load_profile(args.profile, ROOT / "profiles").keyterms, *args.keyterm]))

    print(f"{path.name}: {len(audio) / 2 / RATE:.0f} s recorded, {len(spans)} pieces of speech ({speech:.0f} s), {len(expected)} sentences expected")
    print(f"would send: Deepgram {2 * speech:.0f} s (2 settings x {speech:.0f} s), Azure {speech:.0f} s; {3 * len(spans)} requests")
    if len(spans) != len(expected):
        print(f"NOTE: {len(spans)} pieces but {len(expected)} sentences - the 'expected' lines below may be shifted. Try another --gap (now {args.gap}).")
    if args.dry_run:
        for number, (start, end) in enumerate(spans, 1):
            print(f"  {number:2d}  {start:6.1f}s - {end:6.1f}s  ({end - start:.1f} s)")
        return 0
    if not settings.deepgram_api_key:
        print("DEEPGRAM_API_KEY is not set in .env")
        return 2
    have_azure = bool(settings.azure_speech_key and settings.azure_speech_region)
    if not have_azure:
        print("AZURE_SPEECH_KEY / AZURE_SPEECH_REGION are not set in .env: the Azure column is skipped")

    names = [f"nova-3 {args.language}", f"nova-3 {args.language} + keyterms", f"azure {args.azure_locale}"]
    scores: dict[str, list[float]] = {name: [] for name in names}
    exact: dict[str, int] = {name: 0 for name in names}
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        for number, (start, end) in enumerate(spans, 1):
            pcm = piece(audio, start, end)
            heard = await asyncio.gather(
                deepgram(client, settings.deepgram_api_key, pcm, args.language, ()),
                deepgram(client, settings.deepgram_api_key, pcm, args.language, keyterms),
                azure(client, settings.azure_speech_key, settings.azure_speech_region, pcm, args.azure_locale) if have_azure else asyncio.sleep(0, "(skipped)"),
            )
            wanted: Optional[str] = expected[number - 1] if number <= len(expected) else None
            print(f"\n{number:2d}. {start:.1f}s - {end:.1f}s")
            print(f"    {'expected':<24} {wanted or '(no sentence for this piece)'}")
            for name, text in zip(names, heard):
                score = similarity(wanted, text) if wanted and not text.startswith(("ERROR", "(")) else None
                if score is not None:
                    scores[name].append(score)
                    exact[name] += int(normalise(wanted or "") == normalise(text))
                print(f"    {name:<24} {text or '(nothing)'}{'' if score is None else f'   [{score:.2f}]'}")
            await asyncio.sleep(0.3)  # gentle on the free tiers

    print("\nSummary (similarity to the expected sentence, 1.00 = the same letters):")
    for name in names:
        values = scores[name]
        if values:
            print(f"  {name:<24} average {sum(values) / len(values):.2f}   exact {exact[name]}/{len(values)}   worst {min(values):.2f}")
        else:
            print(f"  {name:<24} no results")
    print("Hebrew words written in Arabic letters (or the other way round) lower the score although the word was heard: read sentences 7-10 by eye.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
