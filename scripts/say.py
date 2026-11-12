#!/usr/bin/env python3
"""Synthesize a test sentence with Azure and save it as a WAV file to listen to.

    .venv/bin/python scripts/say.py --lang he
    .venv/bin/python scripts/say.py --lang ar "مرحبا، كيف بقدر أساعدك؟"
    .venv/bin/python scripts/say.py --voice ar-LB-LaylaNeural --lang ar
    .venv/bin/python scripts/say.py --all          # the five voices suggested for he / ar

The audio is exactly what a caller would hear: 8 kHz, 16-bit, mono - phone
quality, so do not expect it to sound like a podcast. Files go to ``data/tmp``
(git-ignored) and are deleted automatically once they are older than 7 days;
the clean-up runs every time this script or the server starts.

Uses AZURE_SPEECH_KEY / AZURE_SPEECH_REGION from .env (never printed). Each
run is one request per voice and counts towards the free tier's 20 requests
per minute and 500,000 characters per month; the monthly total is shown.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from app.audiosocket import SAMPLE_RATE_HZ  # noqa: E402
from app.config import load_settings  # noqa: E402
from app.recording import cleanup_old_recordings  # noqa: E402
from app.speech import TTSError  # noqa: E402
from app.tts_azure import FREE_TIER_CHARS_PER_MONTH, AzureTTS, RateLimiter, UsageMeter, default_voice  # noqa: E402

#: Test sentences for a fictional dental clinic.
SENTENCES = {
    "he": "שלום, הגעתם למרפאת השיניים הרבור. במה אפשר לעזור?",
    "ar": "مرحبا، وصلت لعيادة هاربر لطب الأسنان. كيف بقدر أساعدك؟",
    "en": "Hello, you have reached Harbor Street Dental. How can I help?",
}
#: The voices suggested in the phase 2c review.
SUGGESTED = [
    ("he", "he-IL-HilaNeural"),
    ("he", "he-IL-AvriNeural"),
    ("ar", "ar-JO-SanaNeural"),
    ("ar", "ar-LB-LaylaNeural"),
    ("ar", "ar-JO-TaimNeural"),
]


def write_wav(path: Path, pcm: bytes) -> None:
    """Wrap raw 8 kHz 16-bit mono PCM in a WAV header. The file is readable only by its owner."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as handle, wave.open(handle, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE_HZ)
        wav.writeframes(pcm)


async def synthesize_to_file(tts: AzureTTS, text: str, directory: Path) -> dict[str, Any]:
    """One request: stream the audio, time it, save it. Raises :class:`TTSError`."""
    started = time.monotonic()
    first_ms = None
    audio = bytearray()
    async for chunk in tts.stream(text):
        if first_ms is None:
            first_ms = (time.monotonic() - started) * 1000.0
        audio += chunk
    if not audio:
        raise TTSError("Azure returned no audio")
    total_ms = (time.monotonic() - started) * 1000.0
    path = directory / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{tts.voice}.wav"
    write_wav(path, bytes(audio))
    return {
        "path": path,
        "voice": tts.voice,
        "chars": len(text),
        "first_audio_ms": first_ms,
        "total_ms": total_ms,
        "audio_s": len(audio) / (SAMPLE_RATE_HZ * 2),
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description="Synthesize a test sentence with Azure TTS and save it as a WAV file.")
    parser.add_argument("text", nargs="?", help="what to say (default: a test sentence in the chosen language)")
    parser.add_argument("--lang", choices=sorted(SENTENCES), default="he", help="language of the default sentence and voice (default he)")
    parser.add_argument("--voice", help="Azure voice name, e.g. he-IL-AvriNeural (default: the profile default for --lang)")
    parser.add_argument("--all", action="store_true", help="say the default sentence with each of the five suggested he/ar voices")
    args = parser.parse_args()

    settings = load_settings(ROOT / ".env")
    if not (settings.azure_speech_key and settings.azure_speech_region):
        print("AZURE_SPEECH_KEY and AZURE_SPEECH_REGION must be set in .env")
        return 2
    removed = cleanup_old_recordings(settings.tmp_audio_dir)
    if removed:
        print(f"removed {removed} test file(s) older than 7 days from {settings.tmp_audio_dir}")

    jobs = SUGGESTED if args.all else [(args.lang, args.voice or default_voice(args.lang) or "")]
    limiter = RateLimiter(settings.azure_tts_max_per_minute)
    usage = UsageMeter(settings.usage_dir)
    failed = 0
    async with httpx.AsyncClient(timeout=httpx.Timeout(20.0)) as http:
        for language, voice in jobs:
            text = args.text if (args.text and not args.all) else SENTENCES[language]
            tts = AzureTTS(http, settings.azure_speech_key, settings.azure_speech_region, voice, limiter=limiter, usage=usage)
            try:
                result = await synthesize_to_file(tts, text, settings.tmp_audio_dir)
            except TTSError as exc:
                failed += 1
                print(f"{voice}: FAILED - {exc}")
                continue
            print(
                f"{voice}: {result['audio_s']:.1f} s of audio, first audio after {result['first_audio_ms']:.0f} ms, "
                f"done in {result['total_ms']:.0f} ms, {result['chars']} characters\n    {result['path']}"
            )
    month = usage.month()
    print(f"\nAzure usage in {month['month']}: {month['chars']} of {FREE_TIER_CHARS_PER_MONTH} characters, {month['requests']} requests")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
