#!/usr/bin/env python3
"""Play one recorded call through several speech-to-text settings and compare the transcripts.

    .venv/bin/python scripts/stt_compare.py data/recordings/<call-uuid>.wav
    .venv/bin/python scripts/stt_compare.py --latest --keyterm תור --keyterm "קופת חולים"

Why: to choose between "Arabic only", "Arabic + Hebrew keyterms" and
"multi-language" for callers who mix Arabic and Hebrew, the settings must hear
exactly the same audio. A recording of one real call (DEBUG_RECORD_AUDIO=true,
caller audio only, deleted after 7 days) is streamed to Deepgram once per
setting, at the same time and at real speed, through the same adapter the
phone line uses. Nothing is sent to the language model, so this costs no Groq
tokens - only Deepgram audio time (recording length x number of settings).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
import wave
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import load_settings  # noqa: E402
from app.speech import END_OF_TURN, SttError, TurnEvent  # noqa: E402
from app.stt_deepgram import AudioClock, DeepgramParams  # noqa: E402
from app.stt_nova import NovaStream  # noqa: E402

FRAME_BYTES = 320  # 20 ms at 8 kHz, 16-bit


def read_recording(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 8000):
            raise SystemExit(f"{path}: expected 8 kHz 16-bit mono (a call recording)")
        return wav.readframes(wav.getnframes())


def settings_to_compare(language: str, hints: list[str], keyterms: list[str], base: Any) -> dict[str, DeepgramParams]:
    common = dict(endpointing_ms=base.nova_endpointing_ms, utterance_end_ms=base.nova_utterance_end_ms, smart_format=False)
    return {
        f"{language} only": DeepgramParams(language=language, **common),
        f"{language} + keyterms": DeepgramParams(language=language, keyterms=tuple(keyterms), **common),
        f"multi [{','.join(hints)}]": DeepgramParams(language="multi", language_hints=tuple(hints), **common),
        f"multi [{','.join(hints)}] + keyterms": DeepgramParams(language="multi", language_hints=tuple(hints), keyterms=tuple(keyterms), **common),
    }


async def transcribe(api_key: str, params: DeepgramParams, audio: bytes, turn_timeout_s: float) -> list[dict[str, Any]]:
    """Stream the recording at real speed; return one row per caller turn."""
    turns: list[dict[str, Any]] = []
    clock = AudioClock()

    def on_event(event: Any, wall: float) -> None:
        if isinstance(event, SttError):
            turns.append({"start": 0.0, "end": 0.0, "text": f"ERROR: {event.message}", "ended_by": "-", "languages": []})
        elif isinstance(event, TurnEvent) and event.event == END_OF_TURN:
            details = event.details or {}
            heard = clock.wall_time_at(event.speech_end)
            turns.append(
                {
                    "start": event.audio_window_start,
                    "end": event.speech_end,
                    "text": event.transcript,
                    "ended_by": details.get("ended_by", ""),
                    "languages": details.get("languages", []),
                    "lost": details.get("lost_words", []),
                    "wait_ms": None if heard is None else round((wall - heard) * 1000.0),
                }
            )

    stream = NovaStream(api_key, params, on_event, clock, turn_timeout_s=turn_timeout_s)
    stream.start()
    data = audio + bytes(8000 * 2 * 3)  # three seconds of silence so the last turn can end
    started = time.monotonic()
    for index in range(0, len(data), FRAME_BYTES):
        stream.send_audio(data[index : index + FRAME_BYTES], time.time())
        await asyncio.sleep(max(0.0, started + (index // FRAME_BYTES + 1) * 0.02 - time.monotonic()))
    await stream.close()
    return turns


async def main() -> int:
    parser = argparse.ArgumentParser(description="Compare speech-to-text settings on one recorded call.")
    parser.add_argument("recording", nargs="?", help="a call recording (data/recordings/<uuid>.wav)")
    parser.add_argument("--latest", action="store_true", help="use the newest recording")
    parser.add_argument("--language", default="ar", help="the line's language (default ar)")
    parser.add_argument("--hint", action="append", default=[], help="languages for multi mode (default: ar and he)")
    parser.add_argument("--keyterm", action="append", default=[], help="a word to listen for; repeat for several")
    args = parser.parse_args()

    settings = load_settings(ROOT / ".env")
    if not settings.deepgram_api_key:
        print("DEEPGRAM_API_KEY is not set in .env")
        return 2
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
    seconds = len(audio) / 16000
    configs = settings_to_compare(args.language, args.hint or ["ar", "he"], args.keyterm, settings)
    print(f"{path.name}: {seconds:.0f} s of caller audio, {len(configs)} settings at once (about {seconds + 5:.0f} s)")
    if args.keyterm:
        print("keyterms:", ", ".join(args.keyterm))
    results = await asyncio.gather(
        *(transcribe(settings.deepgram_api_key, params, audio, settings.nova_turn_timeout_ms / 1000.0) for params in configs.values())
    )
    for (name, _), turns in zip(configs.items(), results):
        print(f"\n== {name}: {len(turns)} turns")
        for turn in turns:
            detected = f" detected={'+'.join(turn['languages'])}" if turn["languages"] else ""
            lost = f" lost={turn['lost']}" if turn.get("lost") else ""
            print(f"  {turn['start']:6.1f}s-{turn['end']:6.1f}s  {turn['text']}   [{turn['ended_by']}, wait {turn.get('wait_ms')} ms{detected}{lost}]")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
