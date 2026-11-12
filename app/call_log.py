"""Per-call record: one JSON object per line in ``data/calls/<uuid>.jsonl``.

Transcripts are personal data, so the directory is created owner-only (0700)
and files 0600. Writing never raises into the call: on an I/O error the
recorder logs once and turns itself off.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, TextIO

from .slots import display_value

log = logging.getLogger("voice_agent.calllog")


def format_record(record: dict[str, Any]) -> Optional[str]:
    """Human-readable log text for a transcript record (None = do not log)."""
    kind = record.get("type")
    if kind == "interim":
        return f"[interim] {record['text']}"
    if kind == "final":
        latency = record.get("latency_ms")
        suffix = "" if latency is None else f"  (stt latency {latency:.0f} ms)"
        return f"[final]   {record['text']}{suffix}"
    if kind == "speech_end":
        return f"[speech_end] after {record['silence_s']:.2f}s silence"
    if kind == "utterance_end":
        return f"[utterance_end] last word ended at {record['last_word_end']:.2f}s"
    if kind == "speech_start":
        return f"[speech_start] at {record['audio_t']:.2f}s"
    if kind == "user":
        if record.get("ignored") == "closing":
            return f"[user] {record['text']} (said during the closing line; not a new request, so the call ends as planned)"
        return f"[user] {record['text']}"
    if kind == "bot":
        note = " (interrupted)" if record.get("interrupted") else ""
        label = f"[bot:{record['kind']}]" if record.get("kind") else "[bot] "
        return f"{label} {record['text']}{note}"
    if kind == "turn":
        def ms(key: str) -> str:
            value = record.get(key)
            return "n/a" if value is None else f"{value:.0f}"

        return (
            f"[turn {record['turn']}] stt_ms={ms('stt_ms')} llm_first_token_ms={ms('llm_first_token_ms')} "
            f"tts_first_audio_ms={ms('tts_first_audio_ms')} total_ms={ms('total_ms')} "
            f"eager={'yes' if record.get('eager') else 'no'} llm_path={record.get('llm_path') or 'n/a'}"
        )
    if kind == "barge_in":
        dropped = record.get("dropped_frames", 0)
        paused = record.get("local_pause_ms")
        how = "" if paused is None else f"; bot was already paused {paused:.0f} ms earlier by local voice detection"
        return f"[barge_in] caller interrupted; dropped {dropped} queued frames ({dropped * 0.02:.1f}s of speech){how}"
    if kind == "barge_pause":
        return f"[barge_pause] caller voice detected locally (rms={record.get('rms', 0):.0f}); bot paused, waiting for STT to confirm"
    if kind == "barge_resume":
        return f"[barge_resume] STT did not confirm within {record.get('paused_ms', 0):.0f} ms; bot resumed"
    if kind == "stt_event":
        details = record.get("details")
        if record.get("event") == "EndOfTurn" and isinstance(details, dict):
            # Nova-3: what ended the turn, and whether words were dropped between guess and final text
            lost = details.get("lost_words") or []
            return (
                f"[stt_turn] ended_by={details.get('ended_by')} pieces={len(details.get('segments') or [])}"
                + (f" words in the interim guess but not in the final text: {', '.join(lost)}" if lost else " no words lost between interim and final")
            )
        if record.get("event") != "StartOfTurn":
            return None  # Update / Eager / EndOfTurn details are kept in the call file only
        return f"[stt] StartOfTurn (audio {record.get('audio_end', 0):.2f}s, reported {record.get('lag_ms') or 0:.0f} ms after that audio arrived)"
    if kind == "ending":
        return f"[ending] outcome={record.get('outcome')}; hanging up after {record.get('silence_s', 0):g}s of silence unless the caller speaks"
    if kind == "hangup_extended":
        return (
            f"[hangup_extended] sound on the line (rms={record.get('rms')}) {record.get('waited_ms', 0):.0f} ms after the closing line; "
            f"hanging up in {record.get('hangup_in_ms', 0):.0f} ms" + (" (at the limit)" if record.get("capped") else "")
        )
    if kind == "hangup_cancelled":
        return "[hangup_cancelled] caller asked something during the closing line; call continues"
    if kind == "ending_blocked":
        why = {
            "caller_did_not_decline": "the caller's last words were not a decline or goodbye",
            "not_fit_without_asking": "not-a-fit ending before asking if there is anything else",
        }.get(str(record.get("reason")), str(record.get("reason")))
        return f"[ending_blocked] model tried to end the call but {why}; staying on the line"
    if kind == "state":
        known = ", ".join(f"{k}={display_value(v)}" for k, v in (record.get("slots") or {}).items() if v is not None) or "none"
        return (
            f"[state] outcome={record.get('outcome') or 'undecided'} phase={record.get('phase')} "
            f"next={record.get('next_slot') or '-'} slots: {known}"
            + (" (replanned)" if record.get("replanned") else "")
            + (f" (will re-check {record['recheck']})" if record.get("recheck") else "")
        )
    if kind == "rtp_stats":
        def ms(key: str) -> str:
            value = record.get(key)
            return "?" if value is None else f"{value:g} ms"

        def part(direction: str, label: str) -> str:
            return (
                f"{label}: packets={record.get(direction + '_packets', '?')} lost={record.get(direction + '_lost', '?')}"
                f" ({record.get(direction + '_loss_pct', '?')}%)"
                f" jitter avg={ms(direction + '_jitter_avg_ms')} worst={ms(direction + '_jitter_max_ms')}"
            )

        return (
            f"[rtp] {part('rx', 'phone->server')} | {part('tx', 'server->phone (reported by phone)')}"
            f" | round trip avg={ms('rtt_avg_ms')} worst={ms('rtt_max_ms')}"
        )
    if kind == "llm_fallback_model":
        return (
            f"[llm_fallback_model] the main model's daily token limit is used up; "
            f"this call is answered by {record.get('model')}"
        )
    if kind == "turn_merge":
        parts = record.get("parts") or ["", ""]
        return (
            f"[turn_merge] the caller went on speaking {record.get('gap_ms', 0):.0f} ms after the turn ended, before the reply "
            f"was audible; answered as one turn: \"{parts[0]}\" + \"{parts[-1]}\""
        )
    if kind == "tts_rate_limited":
        return (
            f"[tts_rate_limited] text-to-speech asked us to slow down (retry after {record.get('retry_after_s', 0):g} s); "
            f"said \"one moment\", retrying once in {record.get('waiting_s', 0):g} s"
        )
    if kind == "tts_usage":
        return (
            f"[tts_usage] {record.get('provider')}: this call {record.get('call_chars', 0)} characters; "
            f"month {record.get('month')}: {record.get('month_chars', 0)} of {record.get('month_limit', 0)} characters, "
            f"{record.get('month_requests', 0)} requests"
        )
    if kind == "llm_json_fallback":
        return "[llm_json_fallback] the model answered in plain text instead of JSON; the text was clean, so it was spoken (slots unchanged)"
    if kind == "eager_skipped":
        return "[eager_skipped] token budget is low; early start disabled until it recovers"
    if kind == "error":
        return f"[error] {record['stage']}: {record['message']}"
    if kind == "max_duration":
        return f"[max_duration] call reached {record['seconds']}s limit; saying goodbye"
    if kind == "stt_error":
        return f"[stt_error] {record['message']}"
    return None


class CallRecorder:
    """Appends records for one call to a JSONL file."""

    def __init__(self, directory: Path, call_id: str, started_wall: Optional[float] = None) -> None:
        self._started = time.time() if started_wall is None else started_wall
        self._file: Optional[TextIO] = None
        self.path = directory / f"{call_id}.jsonl"
        try:
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            self._file = os.fdopen(fd, "a", encoding="utf-8", buffering=1)
        except OSError as exc:
            log.error("cannot write call log %s: %s", self.path, exc)

    def write(self, record: dict[str, Any], wall: Optional[float] = None) -> None:
        if self._file is None:
            return
        now = time.time() if wall is None else wall
        line = {
            "ts": datetime.fromtimestamp(now, timezone.utc).isoformat(timespec="milliseconds"),
            "t": round(now - self._started, 3),  # seconds since the call started
            **record,
        }
        try:
            self._file.write(json.dumps(line, ensure_ascii=False) + "\n")
        except OSError as exc:
            log.error("call log write failed, disabling: %s", exc)
            self.close()

    def close(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except OSError:
                pass
            self._file = None
