"""AudioSocket TCP server (asyncio).

Asterisk opens one TCP connection per call (dialplan ``AudioSocket()``), sends
the call UUID, then streams caller audio in 20 ms frames.

Modes, chosen at start-up from ``.env``:

* **agent** (default when both API keys are set): speech-to-text -> Groq LLM
  -> text-to-speech; the bot greets and converses. Which speech services are
  used comes from the profile (see ``app/providers.py``; Flux + Aura by default).
* **stt** (phase 2a fallback): live transcription only, caller hears silence.
* ``ECHO_MODE=true`` additionally plays the caller's audio back (debugging).

Run with ``python -m app.server`` from the project root.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from .agent import Agent, fixed_phrases
from .audio_out import Pacer
from .audiosocket import (
    Frame,
    FrameParser,
    FrameType,
    audio_duration_s,
    encode_frame,
    parse_uuid,
)
from .call_log import CallRecorder, format_record
from .config import Settings, load_settings
from .llm_groq import DAILY_TOKEN_LIMIT, GroqLLM, TokenLedger
from .phrase_cache import PhraseCache
from .profiles import Profile, ProfileError, load_profile
from .prompt_builder import build_system_prompt
from . import rtp_stats
from .recording import RETENTION_DAYS, InboundRecorder, cleanup_old_recordings
from .providers import check_providers, describe_stt, make_stt, make_tts, stt_eager
from .speech import SpeechEvent, STTProvider, TTSProvider
from .tts_azure import FREE_TIER_CHARS_PER_MONTH
from .stt_deepgram import AudioClock, DeepgramParams, DeepgramStream, SttEvent, TranscriptTracker
from .vad import BargeInDetector

log = logging.getLogger("voice_agent")

READ_CHUNK = 4096
#: Exit code for "configuration is wrong"; systemd is told not to restart on it.
EXIT_CONFIG = 78
#: Deepgram drops idle HTTPS connections after ~5 s; stay comfortably below that.
#: Only TTS providers that name a `keep_warm_url` are touched.
KEEP_WARM_INTERVAL_S = 3.0
GROQ_WARM_EVERY = 6  # every 6th tick = every 18 s


#: Tasks that outlive their call (collecting Asterisk's RTP statistics after hang-up).
_after_call: set["asyncio.Task[None]"] = set()


async def record_rtp_stats(settings: Settings, call_id: str, started_wall: float, tag: str) -> None:
    """After hang-up: add Asterisk's network statistics for the call to its log. Never raises."""
    try:
        stats = await rtp_stats.collect(settings.rtp_stats_dir, call_id)
        if not stats:
            return
        record = {"type": "rtp_stats", **stats}
        recorder = CallRecorder(settings.calls_dir, call_id, started_wall)
        recorder.write(record)
        recorder.close()
        line = format_record(record)
        if line:
            log.info("[%s] %s", tag, line)
    except Exception:
        log.exception("could not record RTP statistics")


def format_llm_day(totals: dict[str, dict[str, int]]) -> str:
    """``gpt-oss-120b:41.2k/200k,gpt-oss-20b:3.0k/200k`` - our own count for the last 24 hours."""
    if not totals:
        return "none"
    return ",".join(
        f"{model.split('/')[-1]}:{entry['tokens'] / 1000.0:.1f}k/{DAILY_TOKEN_LIMIT // 1000}k" for model, entry in sorted(totals.items())
    )


def tts_usage_record(tts: Optional[TTSProvider], call_chars: int) -> Optional[dict[str, Any]]:
    """Quota bookkeeping for text-to-speech services that have a monthly allowance (Azure)."""
    meter = getattr(tts, "usage", None)
    if tts is None or meter is None:
        return None
    month = meter.month()
    return {
        "type": "tts_usage",
        "provider": tts.name,
        "voice": tts.voice,
        "call_chars": call_chars,
        "month": month["month"],
        "month_chars": month["chars"],
        "month_requests": month["requests"],
        "month_limit": getattr(meter, "monthly_limit", FREE_TIER_CHARS_PER_MONTH),
    }


@dataclass
class Services:
    """Things shared by all calls, built once at start-up."""

    settings: Settings
    profile: Optional[Profile] = None
    system_prompt: str = ""
    http: Optional[httpx.AsyncClient] = None
    llm: Optional[GroqLLM] = None
    tts: Optional[TTSProvider] = None
    #: The local port this line listens on (None = the main one, settings.port).
    port: Optional[int] = None
    phrases: Optional[PhraseCache] = None

    async def keep_warm(self) -> None:
        """Keep TLS connections to the LLM and TTS hosts open while a call is active.

        A cold HTTPS connection costs 200-300 ms of handshakes, which would land
        on the caller's wait for a reply. Deepgram closes a connection after
        about 5 s without traffic (measured), so we touch it every few seconds
        (the TTS provider says which URL, or None for "do not ping");
        Groq keeps connections much longer, so it is touched less often. These
        requests carry no credentials and their response is irrelevant - only
        the open connection matters.
        """
        if self.http is None:
            return
        tick = 0
        while True:
            tts_url = self.tts.keep_warm_url if self.tts is not None else None
            urls = [tts_url] if tts_url else []
            if tick % GROQ_WARM_EVERY == 0:
                urls.append("https://api.groq.com/")
            for url in urls:
                try:
                    await self.http.get(url, timeout=3.0)
                except httpx.HTTPError:
                    pass
            tick += 1
            await asyncio.sleep(KEEP_WARM_INTERVAL_S)

    @property
    def agent_ready(self) -> bool:
        return self.profile is not None and self.llm is not None and self.phrases is not None

    @property
    def mode(self) -> str:
        if self.agent_ready:
            mode = "agent"
        elif self.settings.stt_active:
            mode = "stt"
        else:
            mode = "no-stt"
        return mode + ("+echo" if self.settings.echo_mode else "")


def build_services(
    settings: Settings, profile_name: Optional[str] = None, port: Optional[int] = None, shared: Optional[Services] = None
) -> Services:
    """Create the clients for one line (one profile on one port).

    The main line uses ``ACTIVE_PROFILE`` on ``PORT``. Extra lines
    (``EXTRA_PROFILES``) pass their profile, their port and the main line's
    services as ``shared``: they reuse its HTTP connections and its view of
    the LLM token budget. Raises :class:`ProfileError` for a bad profile.
    """
    services = Services(settings, port=port)
    if not settings.agent_active:
        return services
    services.profile = load_profile(profile_name or settings.active_profile, settings.profiles_dir)
    check_providers(services.profile, settings)
    services.system_prompt = build_system_prompt(services.profile)
    if shared is not None and shared.http is not None:
        services.http = shared.http
    else:
        services.http = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=4.0, read=10.0, write=4.0, pool=4.0),
            limits=httpx.Limits(max_keepalive_connections=8, keepalive_expiry=120.0),
        )
    # One account: every line shares the per-minute budget and the 24-hour token ledger
    # (which also knows when the main model is refused for the day).
    ledger = shared.llm.ledger if shared is not None and shared.llm is not None else TokenLedger(settings.usage_dir / "groq_tokens.json")
    services.llm = GroqLLM(
        services.http,
        settings.groq_api_key,
        settings.groq_model,
        fallback_model=settings.groq_fallback_model or None,
        ledger=ledger,
    )
    services.llm.fallback_language = services.profile.language
    if shared is not None and shared.llm is not None:
        services.llm.budget = shared.llm.budget
    services.tts = make_tts(services.profile, settings, services.http)
    services.phrases = PhraseCache(services.tts, settings.tts_cache_dir)
    return services


def build_all_services(settings: Settings) -> list[Services]:
    """The main line first, then one per ``EXTRA_PROFILES`` entry.

    A broken extra profile is logged and skipped: it must not take the main
    line down with it.
    """
    main_line = build_services(settings)
    lines = [main_line]
    if not main_line.agent_ready:
        return lines
    for name, port in settings.extra_profiles:
        if port == settings.port:
            log.error("extra profile '%s' is not started: port %d is the main line's port", name, port)
            continue
        try:
            lines.append(build_services(settings, name, port, shared=main_line))
        except ProfileError as exc:
            log.error("extra profile '%s' is not started: %s", name, exc)
    return lines


@dataclass
class CallStats:
    """Per-call counters, reported in one log line when the call ends."""

    started: float = field(default_factory=time.monotonic)
    started_wall: float = field(default_factory=time.time)
    call_uuid: Optional[uuid.UUID] = None
    frames_in: int = 0
    frames_out: int = 0
    unknown_frames: int = 0
    _last_arrival: Optional[float] = None
    _expected_gap_s: float = 0.0
    _jitter_sum_s: float = 0.0
    _jitter_samples: int = 0

    def audio_in(self, payload: bytes, now: float) -> None:
        """Count an inbound audio frame and update the jitter estimate.

        Jitter here = how far each inter-arrival gap is from the gap the audio
        itself implies (20 ms of audio should arrive 20 ms after the previous
        frame). We report the mean of the absolute deviations.
        """
        self.frames_in += 1
        if self._last_arrival is not None:
            gap = now - self._last_arrival
            self._jitter_sum_s += abs(gap - self._expected_gap_s)
            self._jitter_samples += 1
        self._last_arrival = now
        self._expected_gap_s = audio_duration_s(payload)

    @property
    def avg_jitter_ms(self) -> float:
        if not self._jitter_samples:
            return 0.0
        return 1000.0 * self._jitter_sum_s / self._jitter_samples

    def duration_s(self, now: Optional[float] = None) -> float:
        return (time.monotonic() if now is None else now) - self.started


def _ms(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.0f}"


class CallSession:
    """One AudioSocket connection = one call."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        services: Services,
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._services = services
        self._settings = services.settings
        self._parser = FrameParser()
        self._loop = asyncio.get_running_loop()
        self._pending: set[asyncio.TimerHandle] = set()
        self.stats = CallStats()
        self._tag = "--------"
        self._recorder: Optional[CallRecorder] = None
        self._clock = AudioClock()
        # phase 2a (transcription only)
        self._tracker = TranscriptTracker(self._clock)
        self._stt: Optional[DeepgramStream] = None
        # phase 2b (conversational agent)
        self._agent_stt: Optional[STTProvider] = None
        self._agent: Optional[Agent] = None
        self._agent_start: Optional[asyncio.Task[None]] = None
        self._prewarm: Optional[asyncio.Task[None]] = None
        self._end_reason: Optional[str] = None
        self._audio_recorder: Optional[InboundRecorder] = None

    async def run(self) -> None:
        reason = "eof"
        try:
            reason = await self._read_loop()
        except (ConnectionError, asyncio.IncompleteReadError) as exc:
            reason = f"connection-error:{type(exc).__name__}"
        except asyncio.CancelledError:
            reason = "server-shutdown"
            raise
        finally:
            self._cancel_pending()
            await self._shutdown_call()
            self._finish(self._end_reason or reason)
            if self.stats.call_uuid is not None:
                task = self._loop.create_task(
                    record_rtp_stats(self._settings, str(self.stats.call_uuid), self.stats.started_wall, self._tag)
                )
                _after_call.add(task)
                task.add_done_callback(_after_call.discard)

    async def _shutdown_call(self) -> None:
        try:
            for task in (self._agent_start, self._prewarm):
                if task is not None:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
            if self._agent is not None:
                await self._agent.close()
            await self._close()
            for stream in (self._agent_stt, self._stt):
                if stream is not None:
                    await stream.close()  # flush: last results arrive here
            if self._audio_recorder is not None:
                self._audio_recorder.close()
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("[%s] error while closing the call", self._tag)

    # ------------------------------------------------------------------ call start / end
    def _start_call(self, call_uuid: uuid.UUID) -> None:
        self.stats.call_uuid = call_uuid
        self._tag = str(call_uuid)[:8]
        s, services = self._settings, self._services
        self._recorder = CallRecorder(s.calls_dir, str(call_uuid), self.stats.started_wall)
        profile = services.profile.name if services.profile and services.agent_ready else None
        log.info("call started uuid=%s mode=%s%s", call_uuid, services.mode, f" profile={profile}" if profile else "")
        self._record({"type": "call_start", "uuid": str(call_uuid), "mode": services.mode, "profile": profile})
        # Old debug recordings are removed whether or not recording is on right now.
        cleanup_old_recordings(s.recordings_dir)
        if s.debug_record_audio:
            self._audio_recorder = InboundRecorder(s.recordings_dir, str(call_uuid))
            log.warning("[%s] DEBUG_RECORD_AUDIO is on: caller audio is being recorded (kept %d days)", self._tag, RETENTION_DAYS)

        if services.agent_ready:
            self._start_agent()
        elif s.stt_active:
            params = DeepgramParams(
                model=s.deepgram_model,
                language=s.deepgram_language,
                endpointing_ms=s.deepgram_endpointing_ms,
                utterance_end_ms=s.deepgram_utterance_end_ms,
            )
            self._stt = DeepgramStream(s.deepgram_api_key, params, self._on_stt_event, self._clock)
            self._stt.start()
        elif s.stt_enabled:
            log.warning("[%s] DEEPGRAM_API_KEY is not set - transcription is off", self._tag)

    def _start_agent(self) -> None:
        s, services = self._settings, self._services
        assert services.profile and services.llm and services.tts and services.phrases
        pacer = Pacer(self._send_audio_frame, on_frame=self._on_frame_sent)
        llm = services.llm
        detector: Optional[BargeInDetector] = None
        if s.barge_vad_enabled:
            try:
                detector = BargeInDetector(
                    mode=s.barge_vad_mode, min_rms=s.barge_min_rms, min_speech_ms=s.barge_min_speech_ms
                )
            except Exception as exc:  # missing/broken webrtcvad must not break calls
                log.error("[%s] local voice detection unavailable (%s); barge-in relies on STT only", self._tag, exc)
        # The agent is created first because the speech-to-text stream reports to it.
        stt = make_stt(services.profile, s, self._on_speech_event, self._clock)
        self._agent = Agent(
            profile=services.profile,
            system_prompt=services.system_prompt,
            llm=services.llm,
            tts=services.tts,
            phrases=services.phrases,
            pacer=pacer,
            clock=self._clock,
            record=self._log_record,
            hangup=self._agent_hangup,
            eager=stt.supports_eager,
            merge_turns=bool(getattr(stt, "merges_turns", False)),
            eager_allowed=lambda: llm.budget.allows_eager(s.groq_eager_min_budget),
            detector=detector,
            barge_confirm_s=s.barge_confirm_ms / 1000.0,
            end_silence_s=s.end_silence_ms / 1000.0,
            end_silence_max_s=s.end_silence_max_ms / 1000.0,
        )
        self._agent_stt = stt
        stt.start()
        self._prewarm = self._loop.create_task(services.keep_warm())
        self._agent_start = self._loop.create_task(self._agent.start())

    def _finish(self, reason: str) -> None:
        s = self.stats
        summary: dict[str, Any] = {
            "type": "call_end",
            "reason": reason,
            "duration_s": round(s.duration_s(), 2),
            "frames_in": s.frames_in,
            "frames_out": s.frames_out,
            "avg_jitter_ms": round(s.avg_jitter_ms, 2),
            "stt_audio_s": round(self._clock.total_seconds, 2),
        }
        if self._agent is not None:
            a = self._agent.summary()
            summary.update(a)
            summary["stt_failed"] = bool(self._agent_stt and self._agent_stt.failed)
            turns_by = getattr(self._agent_stt, "turns_by", None)
            if turns_by is not None:
                summary["stt_turns_ended_by"] = dict(turns_by)  # Nova-3: speech_final / utterance_end / timeout
            self._record(summary)
            summary["llm_tokens_24h"] = llm_day = self._services.llm.ledger.last_24h() if self._services.llm else {}
            usage = tts_usage_record(self._services.tts, a["tts_chars"])
            if usage is not None:
                self._log_record(usage)
            log.info(
                "call ended uuid=%s duration=%.2fs turns=%d total_ms_avg=%s total_ms_max=%s barge_ins=%d "
                "false_pauses=%d outcome=%s replans=%d llm_paths=%s llm_24h=%s stt_audio_s=%.2f llm_tokens=%d llm_cached=%d eager_skipped=%d tts_chars=%d errors=%d "
                "frames_in=%d frames_out=%d reason=%s",
                s.call_uuid or "unknown",
                s.duration_s(),
                a["turns"],
                _ms(a["total_ms_avg"]),
                _ms(a["total_ms_max"]),
                a["barge_ins"],
                a["false_pauses"],
                a["outcome"] or "none",
                a["replans"],
                "/".join(f"{name}:{count}" for name, count in a["llm_paths"].items()),
                format_llm_day(llm_day),
                self._clock.total_seconds,
                a["llm_tokens"],
                a["llm_cached_tokens"],
                a["eager_skipped"],
                a["tts_chars"],
                a["errors"],
                s.frames_in,
                s.frames_out,
                reason,
            )
        else:
            t = self._tracker
            summary.update(
                {
                    "finals": t.finals,
                    "stt_latency_avg_ms": None if t.avg_latency_ms is None else round(t.avg_latency_ms, 1),
                    "stt_latency_max_ms": None if t.max_latency_ms is None else round(t.max_latency_ms, 1),
                    "deepgram_reported_s": t.deepgram_duration,
                    "deepgram_request_id": t.request_id or None,
                    "stt_failed": bool(self._stt and self._stt.failed),
                }
            )
            self._record(summary)
            log.info(
                "call ended uuid=%s duration=%.2fs finals=%d stt_latency_avg_ms=%s stt_latency_max_ms=%s "
                "audio_sent_s=%.2f frames_in=%d frames_out=%d avg_jitter_ms=%.2f reason=%s",
                s.call_uuid or "unknown",
                s.duration_s(),
                t.finals,
                _ms(t.avg_latency_ms),
                _ms(t.max_latency_ms),
                self._clock.total_seconds,
                s.frames_in,
                s.frames_out,
                s.avg_jitter_ms,
                reason,
            )
        if self._recorder is not None:
            self._recorder.close()

    # ------------------------------------------------------------------ inbound frames
    async def _read_loop(self) -> str:
        while True:
            data = await self._reader.read(READ_CHUNK)
            if not data:
                return "eof"
            now = time.monotonic()
            wall = time.time()
            for frame in self._parser.feed(data):
                done = self._handle(frame, now, wall)
                if done is not None:
                    return done

    def _handle(self, frame: Frame, now: float, wall: float) -> Optional[str]:
        """Process one frame. Returns an end-of-call reason, or None to go on."""
        kind = frame.kind
        if kind is FrameType.AUDIO:
            self.stats.audio_in(frame.payload, now)
            if self._audio_recorder is not None:
                self._audio_recorder.write(frame.payload)
            if self._agent is not None:
                self._agent.on_caller_audio(frame.payload)
            if self._agent_stt is not None:
                self._agent_stt.send_audio(frame.payload, wall)
            elif self._stt is not None:
                self._stt.send_audio(frame.payload, wall)
            if self._settings.echo_mode:
                self._echo_later(frame.payload)
            return None
        if kind is FrameType.UUID:
            try:
                self._start_call(parse_uuid(frame.payload))
            except ValueError as exc:
                log.warning("bad UUID frame: %s", exc)
                return "bad-uuid"
            return None
        if kind is FrameType.HANGUP:
            return "hangup"
        if kind is FrameType.ERROR:
            code = frame.payload[0] if frame.payload else None
            return f"asterisk-error:{code}"
        self.stats.unknown_frames += 1
        log.debug("ignoring unknown frame type 0x%02x (%d bytes)", frame.type, len(frame.payload))
        return None

    # ------------------------------------------------------------------ agent plumbing
    def _on_speech_event(self, event: SpeechEvent, arrival_wall: float) -> None:
        if self._agent is not None:
            self._agent.on_stt_event(event, arrival_wall)

    def _send_audio_frame(self, frame: bytes) -> None:
        if not self._writer.is_closing():
            self._writer.write(encode_frame(FrameType.AUDIO, frame))
            self.stats.frames_out += 1

    def _on_frame_sent(self) -> None:
        if self._agent is not None:
            self._agent.on_frame_sent()

    def _agent_hangup(self, reason: str) -> None:
        """The agent decided the call is over: tell Asterisk to hang up."""
        self._end_reason = reason
        log.info("[%s] hanging up (%s)", self._tag, reason)
        if not self._writer.is_closing():
            self._writer.write(encode_frame(FrameType.HANGUP))
            self._writer.close()

    def _log_record(self, record: dict[str, Any]) -> None:
        """Store a record in the call file and print its log line."""
        try:
            self._record(record)
            text = format_record(record)
            if text is not None:
                log.info("[%s] %s", self._tag, text)
        except Exception:
            log.exception("[%s] failed to log a record", self._tag)

    # ------------------------------------------------------------------ transcription only (phase 2a)
    def _on_stt_event(self, event: SttEvent, arrival_wall: float) -> None:
        """Called by the Deepgram stream for every message. Must never raise."""
        try:
            for record in self._tracker.on_event(event, arrival_wall):
                self._record(record, arrival_wall)
                text = format_record(record)
                if text is not None:
                    log.info("[%s] %s", self._tag, text)
        except Exception:
            log.exception("[%s] failed to handle STT event", self._tag)

    def _record(self, record: dict[str, Any], wall: Optional[float] = None) -> None:
        if self._recorder is not None:
            self._recorder.write(record, wall)

    # ------------------------------------------------------------------ debug echo
    def _echo_later(self, payload: bytes) -> None:
        handle: Optional[asyncio.TimerHandle] = None

        def send() -> None:
            if handle is not None:
                self._pending.discard(handle)
            if self._writer.is_closing():
                return
            self._writer.write(encode_frame(FrameType.AUDIO, payload))
            self.stats.frames_out += 1

        # call_later fires in scheduling order for equal delays, so frame order is kept.
        handle = self._loop.call_later(self._settings.echo_delay_s, send)
        self._pending.add(handle)

    def _cancel_pending(self) -> None:
        for handle in self._pending:
            handle.cancel()
        self._pending.clear()

    async def _close(self) -> None:
        if not self._writer.is_closing():
            self._writer.close()
        try:
            await self._writer.wait_closed()
        except (ConnectionError, OSError):
            pass


async def serve(lines: "Services | list[Services]") -> None:
    """Run the server until SIGTERM / SIGINT. One listening port per line."""
    all_lines = lines if isinstance(lines, list) else [lines]
    services = all_lines[0]
    settings = services.settings

    def handler(line: Services) -> Any:
        async def on_connect(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            await CallSession(reader, writer, line).run()

        return on_connect

    servers: list[asyncio.AbstractServer] = []
    started: list[Services] = []
    for line in all_lines:
        port = line.port or settings.port
        try:
            server = await asyncio.start_server(handler(line), host=settings.host, port=port)
        except OSError as exc:
            if line is services:
                raise
            log.error("extra profile '%s' is not started: cannot listen on port %d (%s)", line.profile.name if line.profile else "?", port, exc)
            continue
        servers.append(server)
        started.append(line)
        bound = ", ".join(f"{s.getsockname()[0]}:{s.getsockname()[1]}" for s in server.sockets or ())
        if line.agent_ready and line.profile is not None:
            log.info(
                "AudioSocket server listening on %s (mode=%s profile=%s stt=%s llm=%s tts=%s eager=%s keyterms=%d local_vad=%s)",
                bound,
                line.mode,
                line.profile.name,
                describe_stt(line.profile, settings),
                settings.groq_model,
                line.tts.voice if line.tts is not None else "-",
                "on" if stt_eager(line.profile, settings) else "off",
                len(line.profile.keyterms),
                "on" if settings.barge_vad_enabled else "off",
            )
        else:
            if settings.agent_enabled:
                log.warning("agent mode is off: DEEPGRAM_API_KEY and GROQ_API_KEY are both required")
            log.info("AudioSocket server listening on %s (mode=%s model=%s)", bound, line.mode, settings.deepgram_model)

    cleanup_old_recordings(settings.recordings_dir)
    cleanup_old_recordings(settings.tmp_audio_dir)  # test audio from scripts/say.py, same 7-day rule
    rtp_stats.cleanup_stale(settings.rtp_stats_dir)
    if settings.debug_record_audio:
        log.warning(
            "DEBUG_RECORD_AUDIO is ON: caller audio of every call is saved to %s and deleted after %d days. "
            "Use for test calls only.",
            settings.recordings_dir,
            RETENTION_DAYS,
        )

    async def warm_all() -> None:
        # Pre-generate greeting / apology / closings so the first call starts instantly.
        # One line after the other: services with a request limit are not hit in parallel.
        for line in started:
            if line.agent_ready and line.phrases is not None and line.profile is not None:
                await line.phrases.warm(fixed_phrases(line.profile))

    warmup: Optional[asyncio.Task[None]] = asyncio.get_running_loop().create_task(warm_all())

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)

    try:
        await stop.wait()
        log.info("shutting down")
    finally:
        if warmup is not None:
            warmup.cancel()
        for server in servers:
            server.close()
        await asyncio.gather(*(server.wait_closed() for server in servers), return_exceptions=True)
        if services.http is not None:
            await services.http.aclose()


def main() -> None:
    try:
        settings = load_settings()
    except ValueError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        sys.exit(EXIT_CONFIG)
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # At DEBUG the websockets library logs handshake headers, which would
    # include the Authorization header. Never let it go below INFO. httpx logs
    # one line per request at INFO; keep that out of the call log too.
    logging.getLogger("websockets").setLevel(logging.INFO)
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    async def run() -> None:
        try:
            lines = build_all_services(settings)
        except ProfileError as exc:
            log.error("cannot start: %s", exc)
            raise SystemExit(EXIT_CONFIG) from None
        await serve(lines)

    asyncio.run(run())


if __name__ == "__main__":
    main()
