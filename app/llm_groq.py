"""Groq chat completions (OpenAI-compatible), streamed token by token."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Mapping, Optional

import httpx

from .text_utils import clean_fallback_text

log = logging.getLogger("voice_agent.llm")

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
JSON_REPLY_MAX_TOKENS = 400

#: How a structured reply was obtained (see GroqLLM.reply_json), best first.
PATH_JSON = "json"
PATH_FUNCTION = "function_call"
PATH_TEXT = "text_fallback"


class LLMError(Exception):
    """The language model could not produce a reply. Message is safe to log."""


_ACCOUNT_ID_RE = re.compile(r"\b(org|user|proj|key)_[A-Za-z0-9]{6,}\b")


def _safe(detail: object) -> str:
    """An error text from the provider, shortened and without account identifiers."""
    return _ACCOUNT_ID_RE.sub(r"\1_***", " ".join(str(detail).split()))[:200]


def _reply_from_generation(generated: str) -> Optional[dict[str, Any]]:
    """A usable reply hidden in a rejected generation, if there is one.

    The forced function call is sometimes rejected only because the model
    named the function wrongly ("json" instead of "reply") although its
    arguments are a perfectly good ``{"slots": ..., "say": ...}``.
    """
    try:
        data = json.loads(generated)
    except (ValueError, TypeError):
        return None
    for candidate in (data, data.get("arguments") if isinstance(data, dict) else None):
        if isinstance(candidate, str):
            try:
                candidate = json.loads(candidate)
            except ValueError:
                continue
        if isinstance(candidate, dict) and isinstance(candidate.get("say"), str) and candidate["say"].strip() and isinstance(candidate.get("slots"), dict):
            return {"slots": candidate["slots"], "say": candidate["say"]}
    return None


@dataclass
class LLMUsage:
    """Token counters, added up over every request of a call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    requests: int = 0
    #: Prompt tokens the provider served from its prompt cache (subset of prompt_tokens).
    cached_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class TokenBudget:
    """How much of the per-minute token allowance is left, as reported by Groq.

    Every response carries ``x-ratelimit-limit-tokens`` and
    ``x-ratelimit-remaining-tokens``. We trust those numbers (they already
    account for prompt caching) and, between responses, assume the allowance
    refills evenly over a minute. Shared by all calls: the limit is per account.
    """

    limit: Optional[float] = None
    remaining: Optional[float] = None
    seen_at: float = 0.0

    def update(self, headers: Mapping[str, str], now: Optional[float] = None) -> None:
        try:
            limit = float(headers["x-ratelimit-limit-tokens"])
            remaining = float(headers["x-ratelimit-remaining-tokens"])
        except (KeyError, ValueError, TypeError):
            return
        if limit <= 0:
            return
        self.limit, self.remaining = limit, max(0.0, remaining)
        self.seen_at = time.monotonic() if now is None else now

    def exhausted(self, now: Optional[float] = None) -> None:
        """The provider answered 429: nothing is left right now."""
        if self.limit is not None:
            self.remaining = 0.0
            self.seen_at = time.monotonic() if now is None else now

    def fraction_left(self, now: Optional[float] = None) -> Optional[float]:
        """Estimated share of the allowance still available (None = not known yet)."""
        if self.limit is None or self.remaining is None:
            return None
        elapsed = max(0.0, (time.monotonic() if now is None else now) - self.seen_at)
        return min(self.limit, self.remaining + elapsed * self.limit / 60.0) / self.limit

    def allows_eager(self, min_fraction: float, now: Optional[float] = None) -> bool:
        """Speculative requests are a luxury: only while enough budget is left."""
        left = self.fraction_left(now)
        return left is None or left >= min_fraction


def parse_sse_line(line: str) -> tuple[str, Optional[dict[str, Any]], bool]:
    """Parse one server-sent-events line -> (text delta, usage or None, stream finished)."""
    if not line.startswith("data:"):
        return "", None, False
    data = line[5:].strip()
    if data == "[DONE]":
        return "", None, True
    try:
        obj = json.loads(data)
    except ValueError:
        return "", None, False
    if not isinstance(obj, dict):
        return "", None, False
    usage = obj.get("usage") or (obj.get("x_groq") or {}).get("usage")
    text = ""
    for choice in obj.get("choices") or []:
        delta = (choice.get("delta") or {}).get("content")
        if isinstance(delta, str):
            text += delta
    return text, usage if isinstance(usage, dict) else None, False


#: Groq's free tier: tokens per rolling 24 hours, per model. Used for display only.
DAILY_TOKEN_LIMIT = 200_000
_TRY_AGAIN_RE = re.compile(r"try again in (?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?", re.IGNORECASE)


def is_daily_limit(detail: object) -> bool:
    """Is this 429 text about the per-day token limit (not the per-minute one)?"""
    text = str(detail).lower()
    return "per day" in text or "(tpd)" in text


def retry_after_seconds(detail: object, default: float = 600.0) -> float:
    """How long Groq says to wait ("Please try again in 5m55.104s"), within sane bounds."""
    match = _TRY_AGAIN_RE.search(str(detail))
    if not match or not any(match.groups()):
        return default
    hours, minutes, seconds = (float(g) if g else 0.0 for g in match.groups())
    return min(max(hours * 3600 + minutes * 60 + seconds, 30.0), 3600.0)


class TokenLedger:
    """Our own count of LLM tokens per model over the last 24 hours, kept on disk.

    Groq's daily limit is a rolling 24 hours per model, and its response
    headers do not show how much of it is left - the first sign used to be a
    refused request. So every reply's token usage is added here (in hourly
    buckets, in a small file shared by the server and the replay script) and
    the total is shown at the end of each call. It only knows about requests
    made through this program since the file was started.

    It also remembers which models are currently refused for the day, so all
    lines of the server switch to the fallback model together.
    """

    def __init__(self, path: Optional[Path] = None, now: Callable[[], float] = time.time) -> None:
        self._path = path
        self._now = now
        self._memory: dict[str, dict[str, list[int]]] = {}  # hour -> model -> [tokens, requests]
        self.blocked_until: dict[str, float] = {}  # model -> wall-clock time it may be tried again

    def _hour(self, when: float) -> str:
        return time.strftime("%Y-%m-%dT%H", time.gmtime(when))

    def _load(self) -> dict[str, dict[str, list[int]]]:
        if self._path is None:
            return self._memory
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return {h: {m: [int(v[0]), int(v[1])] for m, v in models.items()} for h, models in data.items()}
        except (OSError, ValueError, TypeError, AttributeError, IndexError):
            return {}

    def _recent(self, buckets: dict[str, dict[str, list[int]]]) -> dict[str, dict[str, list[int]]]:
        oldest = self._hour(self._now() - 24 * 3600)
        return {hour: models for hour, models in buckets.items() if hour > oldest}

    def add(self, model: str, tokens: int) -> None:
        """One answered request used ``tokens`` tokens. Never raises."""
        buckets = self._recent(self._load())
        entry = buckets.setdefault(self._hour(self._now()), {}).setdefault(model, [0, 0])
        entry[0] += max(0, int(tokens))
        entry[1] += 1
        self._memory = buckets
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(buckets, handle)
            os.replace(tmp, self._path)
        except OSError as exc:
            log.warning("could not store the token ledger in %s: %s", self._path, exc)

    def last_24h(self) -> dict[str, dict[str, int]]:
        """``{model: {"tokens": n, "requests": n}}`` for the last 24 hours."""
        totals: dict[str, dict[str, int]] = {}
        for models in self._recent(self._load()).values():
            for model, (tokens, requests) in models.items():
                entry = totals.setdefault(model, {"tokens": 0, "requests": 0})
                entry["tokens"] += tokens
                entry["requests"] += requests
        return totals

    def block(self, model: str, seconds: float) -> None:
        self.blocked_until[model] = self._now() + seconds

    def blocked(self, model: str) -> bool:
        return self._now() < self.blocked_until.get(model, 0.0)


class GroqLLM:
    def __init__(
        self,
        client: httpx.AsyncClient,
        api_key: str,
        model: str,
        max_tokens: int = 200,
        temperature: float = 0.4,
        fallback_model: Optional[str] = None,
        ledger: Optional[TokenLedger] = None,
    ) -> None:
        self._client = client
        self._api_key = api_key
        self.model = model
        #: Used instead of ``model`` while Groq refuses it for the day (its
        #: per-day token limit); has its own daily allowance. None = no fallback.
        self.fallback_model = fallback_model if fallback_model and fallback_model != model else None
        self.ledger = ledger if ledger is not None else TokenLedger()
        self.model_fallbacks = 0
        self._max_tokens = max_tokens
        self._temperature = temperature
        self.budget = TokenBudget()
        #: Language of the call. When set, a rejected JSON reply whose text is
        #: clean and in this language is used as the spoken reply (see reply_json).
        self.fallback_language: Optional[str] = None

    def _body(self, messages: list[dict[str, str]], model: Optional[str] = None) -> dict[str, Any]:
        model = model or self.model
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_completion_tokens": self._max_tokens,
            "temperature": self._temperature,
        }
        # Reasoning models think before answering; for a phone call we want the
        # minimum, and we never want the thinking text in the stream.
        if model.startswith("openai/gpt-oss"):
            body["reasoning_effort"] = "low"
            body["include_reasoning"] = False
        elif model.startswith("qwen/"):
            body["reasoning_effort"] = "none"
        return body

    def _count(self, used: Optional[dict[str, Any]], usage: Optional[LLMUsage], model: Optional[str] = None) -> None:
        if used:
            total = int(used.get("total_tokens") or 0) or int(used.get("prompt_tokens") or 0) + int(used.get("completion_tokens") or 0)
            self.ledger.add(model or self.model, total)
        if used and usage is not None:
            usage.prompt_tokens += int(used.get("prompt_tokens") or 0)
            usage.completion_tokens += int(used.get("completion_tokens") or 0)
            details = used.get("prompt_tokens_details") or {}
            usage.cached_tokens += int(details.get("cached_tokens") or 0)

    async def reply_json(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        usage: Optional[LLMUsage] = None,
        *,
        text_fallback: bool = True,
    ) -> dict[str, Any]:
        """One complete reply as a dict with a non-empty ``say`` string, and how we got it.

        The model is asked for ``{"slots": {...}, "say": "..."}``. Groq's JSON
        mode for this model fails now and then: the model writes a good answer
        as plain sentences and the API rejects it with HTTP 400 "Failed to
        generate JSON" (it happens more on caller questions; Groq documents no
        fix). So there are three ways to get a reply, tried in this order, and
        the one that worked is returned as ``reply["path"]``:

        1. ``json``           - strict structured output (``json_schema``).
        2. ``function_call``  - the same schema offered as a function the model
           must call. A different mechanism, so it usually works when 1 did
           not, and it still delivers the slots.
        3. ``text_fallback``  - the plain sentences the model wrote, if they are
           *clean* (:func:`~app.text_utils.clean_fallback_text`: short, no JSON
           or markup fragments, in the call's language). No ``slots`` then, so
           the facts of this turn have to be picked up on the next one.
           ``text_fallback=False`` skips this for requests whose whole point is
           the slots.

        If none works, :class:`LLMError` is raised and the caller apologises.
        """
        attempts: list[tuple[str, dict[str, Any]]] = [
            (PATH_JSON, {"response_format": {"type": "json_schema", "json_schema": {"name": "reply", "strict": True, "schema": schema}}}),
            (
                PATH_FUNCTION,
                {
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "reply",
                                "description": "Give your reply to the phone system.",
                                "strict": True,
                                "parameters": schema,
                            },
                        }
                    ],
                    "tool_choice": {"type": "function", "function": {"name": "reply"}},
                },
            ),
        ]
        problem = "no attempt made"
        plain_text: Optional[str] = None  # the first clean plain-text answer seen on the way
        for path, extra in attempts:
            response, model = await self._post(messages, extra, usage)
            generated = ""
            if response.status_code != 200:
                try:
                    error = response.json()["error"]
                    detail = error["message"]
                    generated = str(error.get("failed_generation") or "")
                except Exception:
                    detail = response.text[:200]
                problem = f"Groq HTTP {response.status_code}: {_safe(detail)}"
                log.warning(
                    "reply attempt (%s) failed: %s%s",
                    path,
                    problem,
                    f" | model produced: {' '.join(generated.split())[:200]!r}" if generated else "",
                )
                if response.status_code == 429:
                    self.budget.exhausted()
                if response.status_code != 400:
                    if plain_text is not None:
                        break  # e.g. rate limited on the retry: better the clean text than an apology
                    raise LLMError(problem)
                # 400: the output failed validation (or the mode was refused).
                salvaged = _reply_from_generation(generated)
                if salvaged is not None:
                    salvaged["path"] = path
                    return self._tag(salvaged, model)
                # Otherwise try the next way.
            else:
                try:
                    data = response.json()
                    self._count(data.get("usage"), usage, model)
                    message = data["choices"][0]["message"]
                    calls = message.get("tool_calls") or []
                    content = calls[0]["function"]["arguments"] if calls else message.get("content")
                    generated = content if isinstance(content, str) else ""
                    reply = json.loads(content)
                except (ValueError, KeyError, IndexError, TypeError):
                    problem = "the reply was not valid JSON"
                    reply = None
                if isinstance(reply, dict) and isinstance(reply.get("say"), str) and reply["say"].strip():
                    reply["path"] = path
                    return self._tag(reply, model)
                if reply is not None:
                    problem = "the reply had no 'say' text"
                    generated = ""
            if plain_text is None and text_fallback and self.fallback_language and generated:
                plain_text = clean_fallback_text(generated, self.fallback_language)
        if plain_text is not None:
            return self._tag({"say": plain_text, "path": PATH_TEXT}, model)
        raise LLMError(f"no usable reply after retry ({problem})")

    def _tag(self, reply: dict[str, Any], model: str) -> dict[str, Any]:
        """Note on the reply when it did not come from the main model."""
        if model != self.model:
            reply["model"] = model
        return reply

    async def _post(
        self, messages: list[dict[str, str]], extra: dict[str, Any], usage: Optional[LLMUsage]
    ) -> tuple[httpx.Response, str]:
        """Send one request; returns the response and the model that answered.

        If the main model is refused because its **daily** token allowance is
        used up, the same request goes to ``fallback_model`` at once, and the
        main model is left alone until the time Groq names ("try again in
        ..."), so later turns and other calls do not each waste a request on it.
        """
        models = [self.model]
        if self.fallback_model:
            models = [self.fallback_model] if self.ledger.blocked(self.model) else [self.model, self.fallback_model]
        response: Optional[httpx.Response] = None
        model = models[0]
        for model in models:
            body = self._body(messages, model)
            body["stream"] = False
            body.pop("stream_options", None)
            body.update(extra)
            # Room for the slots object, the sentence and the model's (hidden) reasoning.
            body["max_completion_tokens"] = max(self._max_tokens, JSON_REPLY_MAX_TOKENS)
            if usage is not None:
                usage.requests += 1
            try:
                response = await self._client.post(
                    GROQ_CHAT_URL, json=body, headers={"Authorization": f"Bearer {self._api_key}"}
                )
            except httpx.HTTPError as exc:
                raise LLMError(f"Groq request failed: {type(exc).__name__}") from None
            self.budget.update(response.headers)
            if response.status_code == 429 and model == self.model and self.fallback_model:
                try:
                    detail = response.json()["error"]["message"]
                except Exception:
                    detail = response.text[:300]
                if is_daily_limit(detail):
                    wait_s = retry_after_seconds(detail)
                    self.ledger.block(self.model, wait_s)
                    self.model_fallbacks += 1
                    log.warning(
                        "daily token limit of %s reached; using %s for the next %.0f min", self.model, self.fallback_model, wait_s / 60.0
                    )
                    continue
            break
        assert response is not None
        return response, model

    async def stream(self, messages: list[dict[str, str]], usage: Optional[LLMUsage] = None) -> AsyncIterator[str]:
        """Yield text fragments as the model produces them."""
        if usage is not None:
            usage.requests += 1
        try:
            async with self._client.stream(
                "POST",
                GROQ_CHAT_URL,
                json=self._body(messages),
                headers={"Authorization": f"Bearer {self._api_key}"},
            ) as response:
                self.budget.update(response.headers)
                if response.status_code == 429:
                    self.budget.exhausted()
                if response.status_code != 200:
                    detail = (await response.aread())[:300].decode("utf-8", "replace")
                    try:
                        detail = json.loads(detail)["error"]["message"]
                    except Exception:
                        pass
                    raise LLMError(f"Groq HTTP {response.status_code}: {_safe(detail)}")
                # Usage for the request can be reported more than once (x_groq.usage
                # and then usage); keep the last report and add it exactly once.
                last_usage: Optional[dict[str, Any]] = None
                try:
                    async for line in response.aiter_lines():
                        text, used, done = parse_sse_line(line)
                        if used:
                            last_usage = used
                        if text:
                            yield text
                        if done:
                            return
                finally:
                    if last_usage and usage is not None:
                        usage.prompt_tokens += int(last_usage.get("prompt_tokens") or 0)
                        usage.completion_tokens += int(last_usage.get("completion_tokens") or 0)
                        details = last_usage.get("prompt_tokens_details") or {}
                        usage.cached_tokens += int(details.get("cached_tokens") or 0)
        except httpx.HTTPError as exc:
            raise LLMError(f"Groq request failed: {type(exc).__name__}") from None
