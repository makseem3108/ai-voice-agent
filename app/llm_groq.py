"""Groq chat completions (OpenAI-compatible), streamed token by token."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Mapping, Optional

import httpx

log = logging.getLogger("voice_agent.llm")

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
JSON_REPLY_MAX_TOKENS = 400


class LLMError(Exception):
    """The language model could not produce a reply. Message is safe to log."""


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


class GroqLLM:
    def __init__(
        self,
        client: httpx.AsyncClient,
        api_key: str,
        model: str,
        max_tokens: int = 200,
        temperature: float = 0.4,
    ) -> None:
        self._client = client
        self._api_key = api_key
        self.model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self.budget = TokenBudget()

    def _body(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_completion_tokens": self._max_tokens,
            "temperature": self._temperature,
        }
        # Reasoning models think before answering; for a phone call we want the
        # minimum, and we never want the thinking text in the stream.
        if self.model.startswith("openai/gpt-oss"):
            body["reasoning_effort"] = "low"
            body["include_reasoning"] = False
        elif self.model.startswith("qwen/"):
            body["reasoning_effort"] = "none"
        return body

    def _count(self, used: Optional[dict[str, Any]], usage: Optional[LLMUsage]) -> None:
        if used and usage is not None:
            usage.prompt_tokens += int(used.get("prompt_tokens") or 0)
            usage.completion_tokens += int(used.get("completion_tokens") or 0)
            details = used.get("prompt_tokens_details") or {}
            usage.cached_tokens += int(details.get("cached_tokens") or 0)

    async def reply_json(
        self, messages: list[dict[str, str]], schema: dict[str, Any], usage: Optional[LLMUsage] = None
    ) -> dict[str, Any]:
        """One complete reply as a JSON object with at least a non-empty ``say`` string.

        First choice is strict structured output: the provider constrains the
        model to ``schema``, so the reply is valid by construction. (On Groq
        structured output is delivered in one piece, not token by token.) If
        that request is rejected or the reply is unusable we retry in plain
        JSON mode and check the result ourselves ("parse and retry"), up to
        twice: Groq occasionally answers 400 "Failed to generate JSON" for a
        request that works when simply sent again.
        """
        attempts: list[dict[str, Any]] = [
            {"type": "json_schema", "json_schema": {"name": "reply", "strict": True, "schema": schema}},
            {"type": "json_object"},
            {"type": "json_object"},
        ]
        problem = "no attempt made"
        for response_format in attempts:
            body = self._body(messages)
            body["stream"] = False
            body.pop("stream_options", None)
            body["response_format"] = response_format
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
            if response.status_code != 200:
                generated = ""
                try:
                    error = response.json()["error"]
                    detail = error["message"]
                    generated = str(error.get("failed_generation") or "")
                except Exception:
                    detail = response.text[:200]
                problem = f"Groq HTTP {response.status_code}: {' '.join(str(detail).split())[:200]}"
                log.warning(
                    "structured reply attempt (%s) failed: %s%s",
                    response_format["type"],
                    problem,
                    f" | model produced: {' '.join(generated.split())[:200]!r}" if generated else "",
                )
                if response.status_code == 429:
                    self.budget.exhausted()
                if response.status_code == 400:
                    continue  # schema rejected or output failed validation: try plain JSON mode
                raise LLMError(problem)
            try:
                data = response.json()
                self._count(data.get("usage"), usage)
                content = data["choices"][0]["message"]["content"]
                reply = json.loads(content)
            except (ValueError, KeyError, IndexError, TypeError):
                problem = "the reply was not valid JSON"
                continue
            if isinstance(reply, dict) and isinstance(reply.get("say"), str) and reply["say"].strip():
                return reply
            problem = "the reply had no 'say' text"
        raise LLMError(f"no usable reply after retry ({problem})")

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
                    raise LLMError(f"Groq HTTP {response.status_code}: {' '.join(str(detail).split())[:200]}")
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
