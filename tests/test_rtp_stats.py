"""Asterisk's per-call RTP statistics: parsing, pick-up after hang-up, clean-up."""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

from app.call_log import format_record
from app.config import load_settings
from app.rtp_stats import cleanup_stale, collect, parse_rtp_stats, stats_path
from app.server import record_rtp_stats

UUID = "939de6df-0b8b-49f8-aced-85b91c488e9b"
LINE = (
    "ssrc=1627384950;themssrc=987654321;lp=3;rxjitter=0.001200;rxcount=2742;txjitter=0.004500;txcount=1317;rlp=12;rtt=0.085000"
    "|minrxjitter=0.000000;maxrxjitter=0.010000;avgrxjitter=0.002000;stdevrxjitter=0.001000;"
    "reported_minjitter=0.000000;reported_maxjitter=0.020000;reported_avgjitter=0.005000;reported_stdevjitter=0.003000;"
    "|minrxlost=0.000000;maxrxlost=2.000000;avgrxlost=0.300000;stdevrxlost=0.500000;reported_minlost=0.000000;reported_maxlost=5.000000;"
    "|minrtt=0.060000;maxrtt=0.140000;avgrtt=0.085000;stdevrtt=0.010000;\n"
)


def test_parse_both_directions_in_milliseconds() -> None:
    stats = parse_rtp_stats(LINE)
    assert stats["rx_packets"] == 2742 and stats["rx_lost"] == 3 and stats["rx_loss_pct"] == 0.11
    assert stats["tx_packets"] == 1317 and stats["tx_lost"] == 12 and stats["tx_loss_pct"] == 0.9
    assert stats["rx_jitter_avg_ms"] == 2.0 and stats["rx_jitter_max_ms"] == 10.0  # Asterisk's own: seconds
    assert stats["rtt_avg_ms"] == 85.0 and stats["rtt_max_ms"] == 140.0
    assert "ssrc" not in stats and "themssrc" not in stats  # identifiers are not kept
    assert "rx_jitter" not in stats and "tx_jitter" not in stats  # the mislabelled last-instant values are dropped
    json.dumps(stats)


def test_real_call_values_phone_reports_jitter_in_samples() -> None:
    """The 5G call of 2026-11-12: Asterisk printed 52 / 181 for the phone's reports - samples at 8 kHz."""
    real = (
        "ssrc=1;themssrc=2;lp=0;rxjitter=0.000000;rxcount=2630;txjitter=0.017197;txcount=1561;rlp=0;rtt=0.081222"
        "|minrxjitter=0.000000;maxrxjitter=0.021375;avgrxjitter=0.013190;stdevrxjitter=0.004;"
        "reported_minjitter=10.000000;reported_maxjitter=181.000000;reported_avgjitter=52.000000;reported_stdevjitter=40.0;"
        "|minrxlost=0;maxrxlost=0;avgrxlost=0;stdevrxlost=0;reported_minlost=0;reported_maxlost=0;reported_avglost=0;reported_stdevlost=0;"
        "|minrtt=0.079;maxrtt=0.095092;avgrtt=0.087505;stdevrtt=0.004;"
    )
    stats = parse_rtp_stats(real)
    assert (stats["rx_jitter_avg_ms"], stats["rx_jitter_max_ms"]) == (13.2, 21.4)
    assert (stats["tx_jitter_avg_ms"], stats["tx_jitter_max_ms"]) == (6.5, 22.6)  # 52 and 181 samples / 8
    assert (stats["rtt_avg_ms"], stats["rtt_max_ms"]) == (87.5, 95.1)
    assert format_record({"type": "rtp_stats", **stats}) == (
        "[rtp] phone->server: packets=2630 lost=0 (0.0%) jitter avg=13.2 ms worst=21.4 ms | "
        "server->phone (reported by phone): packets=1561 lost=0 (0.0%) jitter avg=6.5 ms worst=22.6 ms | "
        "round trip avg=87.5 ms worst=95.1 ms"
    )


def test_parse_is_forgiving() -> None:
    assert parse_rtp_stats("") == {} and parse_rtp_stats("garbage; more=garbage") == {}
    assert parse_rtp_stats("rxcount=;lp=nan;txcount=10;rlp=0") == {"tx_packets": 10, "tx_lost": 0, "tx_loss_pct": 0.0}
    assert parse_rtp_stats("rxcount=0;lp=0") == {"rx_packets": 0, "rx_lost": 0}  # no packets: no percentage
    assert parse_rtp_stats(None) == {}  # type: ignore[arg-type]


def test_only_uuid_file_names_are_used(tmp_path: Path) -> None:
    assert stats_path(tmp_path, UUID.upper()) == tmp_path / f"{UUID}.txt"
    assert stats_path(tmp_path, "../../etc/passwd") is None and stats_path(tmp_path, "") is None


def test_collect_reads_and_removes_the_file(tmp_path: Path) -> None:
    async def scenario() -> None:
        path = tmp_path / f"{UUID}.txt"

        async def asterisk_writes_later() -> None:
            await asyncio.sleep(0.15)
            path.write_text(LINE)

        writer = asyncio.get_running_loop().create_task(asterisk_writes_later())
        stats = await collect(tmp_path, UUID, wait_s=2.0)
        await writer
        assert stats is not None and stats["tx_lost"] == 12
        assert not path.exists()

    asyncio.run(scenario())


def test_collect_gives_up_quietly(tmp_path: Path) -> None:
    async def scenario() -> None:
        started = time.monotonic()
        assert await collect(tmp_path, UUID, wait_s=0.2) is None  # no file (e.g. an internal test call)
        assert time.monotonic() - started < 1.0
        assert await collect(tmp_path / "missing", UUID) is None  # not set up: returns at once
        assert await collect(None, UUID) is None
        (tmp_path / f"{UUID}.txt").write_text("nothing useful\n")
        assert await collect(tmp_path, UUID, wait_s=0.2) is None

    asyncio.run(scenario())


def test_stats_are_appended_to_the_call_log(tmp_path: Path) -> None:
    async def scenario() -> None:
        rtp_dir, calls = tmp_path / "rtp", tmp_path / "calls"
        rtp_dir.mkdir()
        (rtp_dir / f"{UUID}.txt").write_text(LINE)
        settings = load_settings(tmp_path / "none.env", environ={"RTP_STATS_DIR": str(rtp_dir), "CALLS_DIR": str(calls)})
        await record_rtp_stats(settings, UUID, time.time() - 30, "939de6df")
        (line,) = (calls / f"{UUID}.jsonl").read_text().splitlines()
        record = json.loads(line)
        assert record["type"] == "rtp_stats" and record["tx_loss_pct"] == 0.9 and 29 < record["t"] < 32
        await record_rtp_stats(settings, UUID, time.time(), "939de6df")  # nothing left: no second record, no error
        assert len((calls / f"{UUID}.jsonl").read_text().splitlines()) == 1

    asyncio.run(scenario())


def test_stale_files_are_removed(tmp_path: Path) -> None:
    old, fresh = tmp_path / "old.txt", tmp_path / "fresh.txt"
    old.write_text("x")
    fresh.write_text("x")
    os.utime(old, (time.time() - 2 * 86400, time.time() - 2 * 86400))
    assert cleanup_stale(tmp_path) == 1 and fresh.exists() and not old.exists()
    assert cleanup_stale(tmp_path / "missing") == 0 and cleanup_stale(None) == 0


def test_default_directory() -> None:
    assert load_settings(Path("/nonexistent.env"), environ={}).rtp_stats_dir == Path("/var/lib/voice-agent-rtp")
