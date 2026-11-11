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
