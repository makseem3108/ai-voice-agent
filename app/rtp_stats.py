"""Network quality of a call, as measured by Asterisk (RTP statistics).

Our own logs can only see the audio that reaches this program. Whether the
caller *heard* the bot cleanly depends on the trip from Asterisk to the phone,
and only Asterisk knows that: it counts the packets it sent and received, and
the phone reports back (RTCP) how many arrived and how unevenly.

At hang-up the dialplan writes those numbers to
``<RTP_STATS_DIR>/<call uuid>.txt`` (one line, ``key=value`` pairs). This
module picks the file up, turns it into an ``rtp_stats`` record for the call
log and removes it. If the directory or the file is not there - Asterisk not
configured for it, or a test call without RTP - nothing happens.

What the numbers mean (direction as seen from the server):

* ``rx_*``  phone -> server. Measured by Asterisk itself.
* ``tx_*``  server -> phone: what the caller hears. Lost packets and jitter
  here are *reported by the phone*; if the phone sends no RTCP they stay 0.
* ``rtt``   round trip time, also from RTCP.

Units. Everything is stored in milliseconds, converted from what Asterisk
18.10 reports (checked against its source, ``res/res_rtp_asterisk.c``):

* its own measurements (``avgrxjitter``, ``maxrxjitter``, the round-trip
  values) are in **seconds**;
* the phone's reports (``reported_avgjitter``, ``reported_maxjitter``) are the
  raw RTCP "interarrival jitter", which counts in **audio samples** - 8000 per
  second on a phone line, so 8 samples = 1 ms.

Asterisk also prints a last-instant ``rxjitter`` / ``txjitter``. In this
version they are mislabelled (``txjitter`` is really the receive jitter and
``rxjitter`` is always 0), so they are not used; average and worst say more
about a call anyway.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("voice_agent.rtp")

WAIT_S = 3.0
POLL_S = 0.1
STALE_AFTER_S = 24 * 3600
MAX_FILE_BYTES = 4096
_PAIR_RE = re.compile(r"([A-Za-z_]+)=([-+0-9.eE]+|nan|inf)?")
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

#: RTP clock of the codecs we allow (ulaw / alaw): samples per second.
RTP_CLOCK_HZ = 8000
_SECONDS = 1000.0  # seconds -> ms
_SAMPLES = 1000.0 / RTP_CLOCK_HZ  # samples -> ms

#: Asterisk's name -> (our name, factor to milliseconds or None for a plain count).
_FIELDS: dict[str, tuple[str, Optional[float]]] = {
    "rxcount": ("rx_packets", None),
    "txcount": ("tx_packets", None),
    "lp": ("rx_lost", None),  # "local lost packets": phone -> server
    "rlp": ("tx_lost", None),  # "remote lost packets": server -> phone, reported by the phone
    "avgrxjitter": ("rx_jitter_avg_ms", _SECONDS),
    "maxrxjitter": ("rx_jitter_max_ms", _SECONDS),
    "reported_avgjitter": ("tx_jitter_avg_ms", _SAMPLES),
    "reported_maxjitter": ("tx_jitter_max_ms", _SAMPLES),
    "avgrtt": ("rtt_avg_ms", _SECONDS),
    "maxrtt": ("rtt_max_ms", _SECONDS),
}


def parse_rtp_stats(text: str) -> dict[str, Any]:
    """Asterisk's ``key=value;key=value|...`` text as a tidy dict (times in ms). Never raises."""
    stats: dict[str, Any] = {}
    for key, raw in _PAIR_RE.findall(text or ""):
        field = _FIELDS.get(key.lower())
        if field is None or not raw or field[0] in stats:
            continue
        name, factor = field
        try:
            value = float(raw)
        except ValueError:
            continue
        if value != value or value in (float("inf"), float("-inf")):
            continue
        stats[name] = int(value) if factor is None else round(value * factor, 1)
    for direction in ("rx", "tx"):
        packets, lost = stats.get(f"{direction}_packets"), stats.get(f"{direction}_lost")
        if isinstance(packets, int) and isinstance(lost, int) and packets + lost > 0 and lost >= 0:
            stats[f"{direction}_loss_pct"] = round(100.0 * lost / (packets + lost), 2)
    return stats


def stats_path(directory: Path, call_id: str) -> Optional[Path]:
    """Where Asterisk writes the statistics of a call (None for anything but a UUID)."""
    call_id = call_id.lower()
    return directory / f"{call_id}.txt" if _UUID_RE.match(call_id) else None


async def collect(directory: Optional[Path], call_id: str, wait_s: float = WAIT_S) -> Optional[dict[str, Any]]:
    """Wait briefly for the call's statistics file, read it and delete it."""
    if directory is None or not directory.is_dir():
        return None  # RTP statistics are not set up on this machine
    path = stats_path(directory, call_id)
    if path is None:
        return None
    deadline = time.monotonic() + wait_s
    while not path.exists():
        if time.monotonic() >= deadline:
            return None
        await asyncio.sleep(POLL_S)
    await asyncio.sleep(POLL_S)  # let Asterisk finish writing
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            text = handle.read(MAX_FILE_BYTES)
    except OSError as exc:
        log.warning("cannot read RTP statistics %s: %s", path, exc)
        return None
    try:
        path.unlink()
    except OSError as exc:
        log.warning("cannot remove RTP statistics %s: %s", path, exc)
    stats = parse_rtp_stats(text)
    return stats or None


def cleanup_stale(directory: Optional[Path], now: Optional[float] = None) -> int:
    """Delete statistics files nobody collected (calls that never reached us)."""
    if directory is None or not directory.is_dir():
        return 0
    cutoff = (time.time() if now is None else now) - STALE_AFTER_S
    removed = 0
    try:
        for path in directory.glob("*.txt"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
            except OSError:
                pass
    except OSError:
        pass
    return removed
