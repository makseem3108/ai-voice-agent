"""Daily token limit: switch to the fallback model, and keep our own 24-hour token count."""

from __future__ import annotations

import asyncio
import dataclasses
import json
from pathlib import Path
from typing import Any, Callable, Optional

import httpx
import pytest

from app.call_log import format_record
from app.config import load_settings
from app.llm_groq import DAILY_TOKEN_LIMIT, GroqLLM, LLMError, LLMUsage, TokenLedger, is_daily_limit, retry_after_seconds
from app.server import build_all_services, format_llm_day
from tests.helpers import FakeLLM, Rig, reply, run

BIG, SMALL = "openai/gpt-oss-120b", "openai/gpt-oss-20b"
#: The text Groq really sent on 2026-11-12 (organisation id shortened).
TPD = ("Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01abcdefghijk` service tier `on_demand` "
       "on tokens per day (TPD): Limit 200000, Used 199211, Requested 1611. Please try again in 5m55.104s.")
TPM = ("Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01abcdefghijk` service tier `on_demand` "
       "on tokens per minute (TPM): Limit 8000, Used 7853, Requested 1583. Please try again in 10.7s.")
SCHEMA: dict[str, Any] = {"type": "object"}


class Clock:
    def __init__(self, t: float = 1_794_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def ok(say: str = "Hello.", tokens: int = 1300) -> httpx.Response:
    content = json.dumps({"slots": {}, "say": say})
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}], "usage": {"prompt_tokens": tokens - 100, "completion_tokens": 100, "total_tokens": tokens}})


def limited(text: str) -> httpx.Response:
    return httpx.Response(429, json={"error": {"message": text, "code": "rate_limit_exceeded"}})


def llm(handler: Callable[[httpx.Request], httpx.Response], *, fallback: Optional[str] = SMALL, ledger: Optional[TokenLedger] = None) -> GroqLLM:
    return GroqLLM(httpx.AsyncClient(transport=httpx.MockTransport(handler)), "groq-test-key", BIG, fallback_model=fallback, ledger=ledger)


def ask(client: GroqLLM) -> tuple[dict[str, Any], LLMUsage]:
    async def scenario() -> tuple[dict[str, Any], LLMUsage]:
        usage = LLMUsage()
        return await client.reply_json([{"role": "user", "content": "hi"}], SCHEMA, usage), usage

    return asyncio.run(scenario())


# ====================================================================== reading Groq's message
def test_daily_and_per_minute_limits_are_told_apart() -> None:
    assert is_daily_limit(TPD) is True and is_daily_limit(TPM) is False and is_daily_limit("") is False
    assert retry_after_seconds(TPD) == pytest.approx(355.104)
    assert retry_after_seconds("Please try again in 1h2m3s.") == 3600.0  # never trusted for more than an hour
    assert retry_after_seconds("Please try again in 4s.") == 30.0  # nor for less than half a minute
    assert retry_after_seconds("no hint at all") == 600.0


# ====================================================================== switching models
def test_daily_limit_switches_to_the_fallback_model_and_stays_there() -> None:
    models: list[str] = []
    clock = Clock()

    def handler(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        models.append(model)
        return limited(TPD) if model == BIG else ok("From the small model.")

    client = llm(handler, ledger=TokenLedger(now=clock))
    first, usage = ask(client)
    assert first["say"] == "From the small model." and first["model"] == SMALL and first["path"] == "json"
    assert models == [BIG, SMALL] and usage.requests == 2 and client.model_fallbacks == 1
    second, usage = ask(client)
    assert models == [BIG, SMALL, SMALL] and usage.requests == 1  # no request wasted on the refused model
    assert second["model"] == SMALL
    clock.t += 356  # the time Groq named has passed: the main model is tried again
    ask(client)
    assert models[-2:] == [BIG, SMALL] and client.model_fallbacks == 2


def test_fallback_request_is_the_same_request_with_the_other_model() -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return limited(TPD) if len(bodies) == 1 else ok()

    ask(llm(handler))
    big, small = bodies
    assert (big["model"], small["model"]) == (BIG, SMALL)
    assert {k: v for k, v in big.items() if k != "model"} == {k: v for k, v in small.items() if k != "model"}
    assert small["reasoning_effort"] == "low" and small["response_format"]["type"] == "json_schema"


def test_main_model_replies_carry_no_model_tag() -> None:
    result, _ = ask(llm(lambda r: ok()))
    assert "model" not in result


def test_per_minute_limit_does_not_switch_models() -> None:
    models: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        models.append(json.loads(request.content)["model"])
        return limited(TPM)

    with pytest.raises(LLMError, match="tokens per minute"):
        ask(llm(handler))
    assert models == [BIG]


def test_no_fallback_configured_or_fallback_also_refused_is_an_error() -> None:
    with pytest.raises(LLMError, match="tokens per day"):
        ask(llm(lambda r: limited(TPD), fallback=None))
    with pytest.raises(LLMError, match="org_\\*\\*\\*"):  # and the account id is masked
        ask(llm(lambda r: limited(TPD)))
    assert llm(lambda r: ok(), fallback=BIG).fallback_model is None  # same as the main model: pointless


def test_all_lines_switch_together_through_the_shared_ledger() -> None:
    ledger = TokenLedger()
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        seen.append(model)
        return limited(TPD) if model == BIG else ok()

    english, hebrew = llm(handler, ledger=ledger), llm(handler, ledger=ledger)
    ask(english)
    ask(hebrew)
    assert seen == [BIG, SMALL, SMALL]  # the Hebrew line did not have to find out for itself


# ====================================================================== our own 24-hour count
def test_ledger_counts_tokens_per_model_over_a_rolling_day(tmp_path: Path) -> None:
    clock = Clock()
    path = tmp_path / "usage" / "groq_tokens.json"
    ledger = TokenLedger(path, now=clock)
    ledger.add(BIG, 1300)
    ledger.add(BIG, 1500)
    ledger.add(SMALL, 900)
    assert ledger.last_24h() == {BIG: {"tokens": 2800, "requests": 2}, SMALL: {"tokens": 900, "requests": 1}}
    assert TokenLedger(path, now=clock).last_24h()[BIG]["tokens"] == 2800  # another process (the replay script) sees it
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    clock.t += 23 * 3600
    ledger.add(BIG, 100)
    assert ledger.last_24h()[BIG] == {"tokens": 2900, "requests": 3}
    clock.t += 2 * 3600  # the first hour has now left the window
    assert ledger.last_24h() == {BIG: {"tokens": 100, "requests": 1}}


def test_ledger_survives_a_broken_file_and_works_in_memory(tmp_path: Path) -> None:
    path = tmp_path / "groq_tokens.json"
    path.write_text("not json")
    ledger = TokenLedger(path)
    ledger.add(BIG, 10)
    assert ledger.last_24h() == {BIG: {"tokens": 10, "requests": 1}}
    memory = TokenLedger()
    memory.add(SMALL, 5)
    assert memory.last_24h() == {SMALL: {"tokens": 5, "requests": 1}}


def test_every_answered_request_is_added_under_the_model_that_answered() -> None:
    ledger = TokenLedger()

    def handler(request: httpx.Request) -> httpx.Response:
        return limited(TPD) if json.loads(request.content)["model"] == BIG else ok(tokens=1450)

    ask(llm(handler, ledger=ledger))
    assert ledger.last_24h() == {SMALL: {"tokens": 1450, "requests": 1}}  # the refused request used nothing


def test_summary_text_for_the_call_log() -> None:
    assert format_llm_day({}) == "none"
    totals = {BIG: {"tokens": 41234, "requests": 30}, SMALL: {"tokens": 3000, "requests": 2}}
    assert format_llm_day(totals) == "gpt-oss-120b:41.2k/200k,gpt-oss-20b:3.0k/200k" and DAILY_TOKEN_LIMIT == 200_000


# ====================================================================== settings and server wiring
def test_settings_and_shared_ledger(tmp_path: Path) -> None:
    assert load_settings(tmp_path / "none.env", environ={}).groq_fallback_model == SMALL
    assert load_settings(tmp_path / "none.env", environ={"GROQ_FALLBACK_MODEL": ""}).groq_fallback_model == ""  # switched off

    async def scenario() -> None:
        profiles = Path(__file__).resolve().parent.parent / "profiles"
        env = {"DEEPGRAM_API_KEY": "dg", "GROQ_API_KEY": "groq", "AZURE_SPEECH_KEY": "az", "AZURE_SPEECH_REGION": "northeurope",
               "USAGE_DIR": str(tmp_path / "usage"), "EXTRA_PROFILES": "clinic_he:9093"}
        s = dataclasses.replace(load_settings(tmp_path / "none.env", environ=env), profiles_dir=profiles, tts_cache_dir=tmp_path / "tts")
        english, hebrew = build_all_services(s)
        assert english.llm.fallback_model == SMALL and hebrew.llm.fallback_model == SMALL
        assert hebrew.llm.ledger is english.llm.ledger
        english.llm.ledger.add(BIG, 1234)
        assert (tmp_path / "usage" / "groq_tokens.json").exists()
        await english.http.aclose()

    asyncio.run(scenario())


# ====================================================================== the call log
def test_call_logs_the_fallback_model_once() -> None:
    async def scenario() -> Rig:
        small = {"model": SMALL}
        rig = Rig(FakeLLM([{**reply("Do you own the house?"), **small}, {**reply("Thanks.", owns_home="yes"), **small}]))
        rig.pacer.start()
        await rig.say("Hi.")
        await rig.say("I own it.")
        await rig.agent.close()
        return rig

    rig = run(scenario)
    (record,) = rig.of("llm_fallback_model")  # once per call, not once per turn
    assert record == {"type": "llm_fallback_model", "turn": 1, "model": SMALL}
    assert rig.agent.summary()["llm_fallback_model"] == SMALL and rig.agent.slots["owns_home"] == "yes"
    line = format_record(record)
    assert line == f"[llm_fallback_model] the main model's daily token limit is used up; this call is answered by {SMALL}"


def test_normal_calls_log_nothing_about_models() -> None:
    async def scenario() -> Rig:
        rig = Rig(FakeLLM([reply("Do you own the house?")]))
        rig.pacer.start()
        await rig.say("Hi.")
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.of("llm_fallback_model") == [] and rig.agent.summary()["llm_fallback_model"] is None


def test_replay_budget_is_thirty_requests() -> None:
    import scripts.replay_llm as replay

    assert replay.MAX_REQUESTS == 30
