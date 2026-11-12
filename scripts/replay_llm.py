#!/usr/bin/env python3
"""Replay realistic turns against the REAL language model and report the pass rate.

Why this exists: unit tests use a fake model, so they cannot tell whether the
real one still answers in valid JSON after a prompt or reply-format change. On
2026-11-12 a format change made every caller *question* in mid-conversation
fail ("Failed to generate JSON") while all unit tests passed.

Run it before deploying any change to the prompt, the per-turn instruction or
the reply format:

    .venv/bin/python scripts/replay_llm.py --runs 3

Each scenario builds exactly the messages the agent would send (system prompt
+ conversation + this turn's instruction) and sends them N times, through the
same client the agent uses. Every run ends in one of four ways:

    json           valid reply from JSON mode (first attempt)
    function_call  JSON mode failed, the forced function call delivered it
    fallback       both failed, the model's clean plain text would be spoken
    apology        nothing usable: the caller would hear the apology

A run **passes** when it is json or function_call *and* the slots are right;
"wrong" means the reply arrived but a slot value was not what was expected.
Exit code 0 means every run passed.

It uses the Groq key from .env (never printed), costs about 1,200 tokens per
run, and paces itself to stay under the free tier's tokens-per-minute limit.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from app.config import load_settings  # noqa: E402
from app.llm_groq import GroqLLM, LLMError, LLMUsage, TokenLedger  # noqa: E402
from app.profiles import Profile, load_profile  # noqa: E402
from app.prompt_builder import ANYTHING_ELSE_QUESTION, build_directive, build_system_prompt  # noqa: E402
from app.slots import FINAL, Plan, empty_slots, make_plan, merge_slots, reply_schema  # noqa: E402
from app.text_utils import clean_fallback_text  # noqa: E402

TOKENS_PER_MINUTE = 6000  # stay below Groq's 8,000 free-tier limit
#: Requests one run may send. The free tier also has a DAILY token limit
#: (200,000 per model, about 140 replies) that the phone line shares: a big
#: replay can use up the day's calls. More than this needs --allow-more.
MAX_REQUESTS = 30


@dataclass
class Scenario:
    name: str
    #: The conversation after the greeting: caller and bot lines alternate, caller first and last.
    lines: list[str]
    known: dict[str, Any] = field(default_factory=dict)
    announced: Optional[str] = None
    #: Slot values the reply must contain after merging (money slots: the amount).
    expect: dict[str, Any] = field(default_factory=dict)
    final: bool = False  # the goodbye extraction request


def english_scenarios() -> list[Scenario]:
    """Scenarios for the lead_qualification profile (English). Clinic profiles: see clinic_scenarios."""
    q_own = "Great, solar panels! Do you own the house, or do you rent?"
    q_bill = "Roughly how much is your monthly electricity bill?"
    return [
        Scenario("question right after the greeting", ["Tell me more about your service."]),
        Scenario(
            "question after one answer (real call 2026-11-12)",
            ["Yes. I'm interested on the solar panels.", q_own, "Alright. Set... tell me more about your service."],
            known={"interest": "solar_panels"},
        ),
        Scenario(
            "same call, with the misheard turn before it",
            [
                "Yes. I'm interested on the solar panels.",
                q_own,
                "with, like, cucumbers in there. Okay. Tell me more.",
                "I'm sorry, I don't have that information. Do you own the house, or do you rent?",
                "Alright. Set... tell me more about your service.",
            ],
            known={"interest": "solar_panels"},
        ),
        Scenario(
            "question after two answers",
            ["Solar panels.", q_own, "I own it.", q_bill, "What does the warranty cover?"],
            known={"interest": "solar_panels", "owns_home": "yes"},
        ),
        Scenario("plain answer", ["Solar panels.", q_own, "I own the house."], known={"interest": "solar_panels"}, expect={"owns_home": "yes"}),
        Scenario(
            "bill amount in words",
            ["Solar panels.", q_own, "I own it.", q_bill, "About five hundred shekels a month."],
            known={"interest": "solar_panels", "owns_home": "yes"},
            expect={"monthly_bill": 500},
        ),
        Scenario(
            "two answers at once",
            ["Both, panels and a battery.", q_own, "I own my house and my name is David."],
            known={"interest": "both"},
            expect={"owns_home": "yes", "name": "David"},
        ),
        Scenario(
            "answer plus a question",
            ["Solar panels.", q_own, "Yes I own it. By the way, how long does the installation take?"],
            known={"interest": "solar_panels"},
            expect={"owns_home": "yes"},
        ),
        Scenario("not a fit (apartment)", ["Solar panels.", q_own, "No. I live in an apartment."], known={"interest": "solar_panels"}, expect={"owns_home": "no"}),
        Scenario(
            "question after the outcome was announced",
            ["Yes, that's fine.", f"Thank you, an advisor will call you on a weekday. {ANYTHING_ELSE_QUESTION}", "How long is the warranty?"],
            known={"interest": "solar_panels", "owns_home": "yes", "monthly_bill": "500 shekels", "name": "David", "callback_ok": "yes"},
            announced="qualified",
        ),
        Scenario(
            "goodbye that carries an answer",
            ["My name is David.", "Is it okay if an advisor calls you back on this number on a weekday?", "Yes. Thank you. Goodbye."],
            known={"interest": "solar_panels", "owns_home": "yes", "monthly_bill": "500 shekels", "name": "David"},
            expect={"callback_ok": "yes"},
            final=True,
        ),
    ]


#: What a caller says in the clinic scenarios, per language.
CLINIC_LINES = {
    "en": {
        "hours": "What are your opening hours?",
        "reason": "I have a toothache.",
        "how_long": "Wait, how long does a check-up take?",
        "two": "It's my first visit, and I'd like Tuesday morning.",
        "braces": "I need braces for my son.",
        "bye": "Zero five two, one two three, four five six seven. Thanks, goodbye.",
    },
    "he": {
        "hours": "מה שעות הפתיחה שלכם?",
        "reason": "יש לי כאב שיניים.",
        "how_long": "רגע, כמה זמן לוקחת בדיקה?",
        "two": "זה ביקור ראשון, ואני רוצה ביום שלישי בבוקר.",
        "braces": "אני צריך יישור שיניים לבן שלי, גשר.",
        "bye": "אפס חמש שתיים, אחת שתיים שלוש, ארבע חמש שש שבע. תודה, להתראות.",
    },
    "ar": {
        "hours": "شو ساعات الدوام عندكم؟",
        "reason": "عندي وجع أسنان.",
        "how_long": "لحظة، قديش بياخد الفحص؟",
        "two": "هاي أول زيارة، وبدي يوم الثلاثا الصبح.",
        "braces": "بدي تقويم أسنان لابني.",
        "bye": "صفر خمسة اثنين، واحد اثنين ثلاثة، أربعة خمسة ستة سبعة. شكراً، مع السلامة.",
    },
}


def clinic_scenarios(profile: Profile) -> list[Scenario]:
    """The same six situations for clinic / clinic_he / clinic_ar ("*" = any non-empty value)."""
    say = CLINIC_LINES[profile.language.split("-")[0]]
    ask = {slot.name: slot.question or "" for slot in profile.slots}
    return [
        Scenario("question right after the greeting", [say["hours"]]),
        Scenario("plain answer", [say["reason"]], expect={"reason": "*"}),
        Scenario("question after one answer", [say["reason"], ask["patient_status"], say["how_long"]], known={"reason": say["reason"]}),
        Scenario(
            "two answers at once",
            [say["reason"], ask["patient_status"], say["two"]],
            known={"reason": say["reason"]},
            expect={"patient_status": "new", "preferred_time": "*"},
        ),
        Scenario("not a fit (braces)", [say["braces"]], expect={"service_offered": "no"}),
        Scenario(
            "goodbye that carries an answer",
            [say["reason"], ask["phone"], say["bye"]],
            known={"reason": say["reason"], "patient_status": "new", "preferred_time": "Tuesday", "name": "Dana"},
            expect={"phone": "*"},
            final=True,
        ),
    ]


def scenarios_for(profile: Profile) -> list[Scenario]:
    return clinic_scenarios(profile) if profile.name.startswith("clinic") else english_scenarios()


def build_messages(profile: Profile, scenario: Scenario) -> tuple[list[dict[str, str]], dict[str, Any]]:
    slots, _ = merge_slots(profile.slots, empty_slots(profile.slots), scenario.known)
    plan = Plan(FINAL) if scenario.final else make_plan(profile.slots, slots, scenario.announced, profile.outcomes)
    history = [{"role": "assistant", "content": profile.greeting}]
    for index, line in enumerate(scenario.lines):
        history.append({"role": "user" if index % 2 == 0 else "assistant", "content": line})
    messages = (
        [{"role": "system", "content": build_system_prompt(profile)}]
        + history
        + [{"role": "system", "content": build_directive(plan, slots)}]
    )
    return messages, slots


def check(profile: Profile, scenario: Scenario, slots: dict[str, Any], reply: dict[str, Any]) -> Optional[str]:
    """None if the reply is right, else what is wrong with it."""
    merged, rejected = merge_slots(profile.slots, slots, reply.get("slots"))
    if rejected:
        return f"invalid slot values: {rejected}"
    for name, wanted in scenario.expect.items():
        got = merged.get(name)
        if isinstance(got, dict):
            got = got.get("amount")
        if wanted == "*":
            if got in (None, ""):
                return f"{name} is empty, expected a value"
        elif str(got).lower() != str(wanted).lower():
            return f"{name}={got!r}, expected {wanted!r}"
    return None


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=int, default=3, help="how many times each scenario is sent (default 3)")
    parser.add_argument("--profile", default="lead_qualification")
    parser.add_argument("--only", default="", help="run only scenarios whose name contains this text")
    parser.add_argument("--show", action="store_true", help="print what the model said in each run")
    parser.add_argument("--model", default="", help="send to this model instead of GROQ_MODEL (e.g. openai/gpt-oss-20b)")
    parser.add_argument("--allow-more", action="store_true", help=f"lift the limit of {MAX_REQUESTS} requests per run (uses up the daily token allowance)")
    args = parser.parse_args()

    settings = load_settings(ROOT / ".env")
    if not settings.groq_api_key:
        print("GROQ_API_KEY is not set in .env")
        return 2
    profile = load_profile(args.profile, ROOT / "profiles")
    scenarios = [s for s in scenarios_for(profile) if args.only.lower() in s.name.lower()]
    schema = reply_schema(profile.slots)
    model = args.model or settings.groq_model
    planned = len(scenarios) * args.runs
    if planned > MAX_REQUESTS and not args.allow_more:
        print(f"{planned} runs planned, more than the budget of {MAX_REQUESTS} requests per run. Use fewer --runs or --only, or --allow-more.")
        return 2
    ledger = TokenLedger(settings.usage_dir / "groq_tokens.json")  # the same count the phone line uses
    print(f"profile={profile.name} language={profile.language} model={model} runs per scenario={args.runs}\n")
    requests_sent = 0

    totals = {"json": 0, "function_call": 0, "fallback": 0, "apology": 0}
    wrong = 0
    first_try_tokens: list[tuple[int, int]] = []  # (prompt, completion) of replies that needed one request
    wrong_language = 0
    latency: dict[str, list[float]] = {key: [] for key in totals}
    async with httpx.AsyncClient(timeout=httpx.Timeout(20.0)) as http:
        llm = GroqLLM(http, settings.groq_api_key, model, ledger=ledger)  # no fallback model: we test exactly this one
        llm.fallback_language = profile.language
        for scenario in scenarios:
            messages, slots = build_messages(profile, scenario)
            results: list[str] = []
            notes: list[str] = []
            times: list[float] = []
            for _ in range(args.runs):
                if requests_sent >= MAX_REQUESTS and not args.allow_more:
                    notes.append(f"stopped: {MAX_REQUESTS} requests sent (retries count too)")
                    break
                usage = LLMUsage()
                started = time.monotonic()
                try:
                    reply = await llm.reply_json(messages, schema, usage, text_fallback=not scenario.final)
                    outcome = {"text_fallback": "fallback"}.get(str(reply.get("path")), str(reply.get("path") or "json"))
                    if outcome != "fallback":
                        problem = check(profile, scenario, slots, reply)
                        if problem:
                            wrong += 1
                            notes.append(f"wrong slots ({outcome}): {problem}")
                            outcome_ok = False
                        else:
                            outcome_ok = True
                    else:
                        outcome_ok = False
                    if usage.requests == 1:
                        first_try_tokens.append((usage.prompt_tokens, usage.completion_tokens))
                    if not scenario.final and clean_fallback_text(str(reply.get("say") or ""), profile.language) is None:
                        wrong_language += 1
                        notes.append(f"say is not clean {profile.language}: {str(reply.get('say'))[:80]!r}")
                    if args.show or outcome == "fallback":
                        notes.append(f"{outcome} said: {str(reply.get('say'))[:100]!r}")
                except LLMError as exc:
                    outcome, outcome_ok = "apology", False
                    notes.append(str(exc)[:80])
                    if "429" in str(exc):
                        # Over the per-minute limit (something else is using the key too):
                        # wait it out, or every following run fails the same way.
                        await asyncio.sleep(30.0)
                requests_sent += usage.requests
                elapsed = (time.monotonic() - started) * 1000.0
                times.append(elapsed)
                latency[outcome].append(elapsed)
                results.append(outcome if outcome_ok or outcome in ("fallback", "apology") else "wrong")
                totals[outcome] += 1
                # pace: spread the tokens just used over the per-minute budget
                await asyncio.sleep(min(40.0, max(2.0, usage.total_tokens / TOKENS_PER_MINUTE * 60.0)))
            passed = sum(1 for r in results if r in ("json", "function_call"))
            avg = f"{sum(times) / len(times):4.0f} ms" if times else "   -   "
            counts = " ".join(f"{k}={results.count(k)}" for k in ("json", "function_call", "fallback", "apology", "wrong") if results.count(k))
            print(f"  {'OK  ' if passed == len(results) else 'FAIL'} {passed}/{len(results)}  {avg}  {scenario.name}  [{counts}]", flush=True)
            for note in dict.fromkeys(notes):
                print(f"            {note}")
    runs = sum(totals.values())

    def avg_ms(key: str) -> str:
        return f"{sum(latency[key]) / len(latency[key]):.0f} ms" if latency[key] else "-"

    print(f"\n{runs} runs:")
    for key, label in (("json", "JSON mode"), ("function_call", "function call (2 requests)"), ("fallback", "plain-text fallback"), ("apology", "apology")):
        print(f"  {label:28} {totals[key]:3d}  ({100.0 * totals[key] / max(runs, 1):3.0f}%)  avg {avg_ms(key)}")
    everything = [t for values in latency.values() for t in values]
    passed_total = totals["json"] + totals["function_call"] - wrong
    print(f"  wrong slot values: {wrong}   overall avg latency: {sum(everything) / max(len(everything), 1):.0f} ms")
    if first_try_tokens:
        prompt_avg = sum(p for p, _ in first_try_tokens) / len(first_try_tokens)
        completion_avg = sum(c for _, c in first_try_tokens) / len(first_try_tokens)
        print(f"  tokens per reply (one request): prompt {prompt_avg:.0f} + completion {completion_avg:.0f} = {prompt_avg + completion_avg:.0f}")
    print(f"  replies not in {profile.language}: {wrong_language}")
    day = ledger.last_24h().get(model, {"tokens": 0, "requests": 0})
    print(f"  requests sent: {requests_sent}; {model} in the last 24 h (our own count): {day['tokens']} tokens, {day['requests']} requests")
    print(f"pass rate (reply with correct slots): {passed_total}/{runs} ({100.0 * passed_total / max(runs, 1):.0f}%)")
    return 0 if passed_total == runs else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
