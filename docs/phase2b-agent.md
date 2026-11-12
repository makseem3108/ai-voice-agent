# Phase 2b - the bot talks

Call extension `700`: the bot greets you, you have a conversation, and it
stops talking the moment you interrupt it.

```
                      +--> Deepgram Flux (speech-to-text + turn detection)
caller audio (20 ms) -+                 |
                                        v  "the caller finished: <text>"
                               conversation engine (app/agent.py)
                                        |  history + system prompt
                                        v
                               Groq LLM (streams the reply word by word)
                                        |  one finished sentence at a time
                                        v
                               Deepgram Aura (text-to-speech, 8 kHz PCM)
                                        |
                                        v
bot audio (20 ms) <---- pacer (app/audio_out.py): one frame every 20 ms
```

## Who the bot is: profiles
The agent's personality and job come from a YAML file in `profiles/`, not
from code. `ACTIVE_PROFILE` in `.env` picks one (default
`lead_qualification`); restart the service after changing it.

| Profile              | Fictional business   | Job                                |
|----------------------|----------------------|------------------------------------|
| `lead_qualification` | Northwind Solar      | qualify a home-solar enquiry       |
| `clinic`             | Harbor Street Dental | take an appointment request        |
| `travel`             | Blue Compass Travel  | collect details for a trip offer   |

Fields: `business_name`, `agent_role`, `goal`, `greeting`, `slots`,
`knowledge` (free text), `tone`, `closing_qualified`, `closing_not_fit`,
`outcomes` (optional), `max_call_seconds` (default 300) and `keyterms`
(optional list).

The file is checked at start-up; a typo or missing field stops the service
with a message that names the profile and every problem found.

### Slots: what the call must find out

A slot is one fact the agent needs. Each slot has:

| Field            | Meaning                                                                  |
|------------------|--------------------------------------------------------------------------|
| `name`           | identifier, e.g. `owns_home`                                             |
| `type`           | `yes_no`, `choice` (with `values`), `text`, `number` or `money`          |
| `currency`       | for `money`: the currency assumed when the caller names none (`ILS`)     |
| `question`       | what to ask the caller; a slot without one is only inferred, never asked |
| `required`       | default true                                                             |
| `disqualifying`  | values that mean we cannot help this caller                              |
| `not_fit_reason` | why (the model phrases it kindly)                                        |
| `description`    | hint for extraction, e.g. "an apartment counts as no"                    |

The lead profile collects `interest`, `owns_home`, `monthly_bill`, `name` and
`callback_ok`; `owns_home: no` (renting or an apartment) is disqualifying.

**Money slots** keep the caller's words and add a number when one was said:

```json
"monthly_bill": {"text": "over five hundred shekels per month", "amount": 500, "currency": "ILS"}
```

The model returns one text field (what the caller said); the number and the
currency are read from it by code (`app/amounts.py`: digits and number words
in English, Hebrew and Arabic). If no single number can be understood -
nothing was said, or two different numbers ("between 400 and 500") - `amount`
and `currency` are null and the text alone fills the slot. An earlier version
asked the model for three fields; that made it stop answering in JSON on
caller questions, so the format is kept as simple as possible.

### Outcome variants

`outcomes` names flavours of "qualified" where the lead is valid but the
follow-up differs. Each has a `when` condition on slot values, an `announce`
instruction for the model and its own fixed `closing` line:

```yaml
outcomes:
  qualified_no_callback:
    when: {callback_ok: "no"}
    announce: The caller does not want a call back. ... Do not promise that anyone will call.
    closing: Thanks, I've noted your details, and we won't call you. ...
```

A variant applies only when every required slot is filled and nothing is
disqualifying. In the lead profile, declining the call back is therefore not a
rejection: the outcome is `qualified_no_callback`.

## Turn flow
### Who decides what

The **code** owns the flow of the call. The **language model** only does two
things: pull facts out of what the caller said, and phrase sentences.

Every model reply is structured output, a JSON object with two parts:

```json
{"slots": {"owns_home": "no", "monthly_bill": null, "name": null}, "say": "..."}
```

Only `say` is sent to text-to-speech. `slots` is validated against the
profile (unknown slots and values outside the allowed set are rejected and
logged) and merged into the call's state. From that state the code decides:

| Decision                | Rule (in `app/slots.py`)                                                |
|-------------------------|-------------------------------------------------------------------------|
| Outcome                 | any disqualifying value -> `not_fit`; all required slots filled -> `qualified` (or a matching variant from `outcomes`); otherwise keep going |
| Next question           | the first required slot still missing; the model is told which one and only words it |
| "Anything else?"        | asked by the server, exactly once, right after the outcome is announced |
| Closing line            | a fixed phrase picked from the outcome - the model never sees or chooses it |
| Hanging up              | only after the caller's own words were a decline or a goodbye           |

### One turn, step by step

1. **Greeting.** Played as soon as the call connects (cached audio).
2. **Listening.** Every 20 ms frame goes to Deepgram Flux (80 ms chunks), which
   transcribes and decides when your turn starts and ends.
3. **Plan.** At `EndOfTurn` the code looks at the slots and makes a plan for
   this reply: *collect* (ask for the next missing slot), *announce* (the
   outcome was just decided: tell the caller) or *answer* (outcome already
   announced: just answer). The plan is sent to the model as a short
   instruction after the conversation.
4. **Reply.** The model returns `{slots, say}`. The slots are merged and the
   plan is made again. If what the caller said changed the plan - they turned
   out not to be a fit, or answered several things at once - the first reply
   no longer fits, is thrown away unheard, and the model is asked once more
   with the new plan (`replanned` in the logs).
5. **Speaking.** `say` is split into sentences, each goes to Aura, and the
   audio is queued for the pacer. After an *announce* reply the server adds
   "Is there anything else I can help you with?" from its phrase cache.

A per-turn `[state]` line shows the slots, the computed outcome, the phase
and the next slot; the same record is in the call file (and is what a CRM
webhook will consume later).

### Structured output and latency

Groq's strict structured output (`json_schema`, `strict: true`) constrains the
model to the reply schema. On Groq a structured reply is delivered in one
piece, not token by token, so the first sentence can no longer start while the
second is still being written. With replies of one or two sentences the whole
reply arrives in roughly 0.4-0.6 s.

In practice JSON mode fails now and then with this model: it writes a good
answer as plain sentences and Groq rejects it (HTTP 400 "Failed to generate
JSON"), mostly on caller questions. Groq documents no cure, so a reply can be
obtained in three ways, tried in order; the `[turn]` line shows which one
answered (`llm_path=`):

| Path             | What it is                                                        | Slots |
|------------------|-------------------------------------------------------------------|-------|
| `json`           | strict JSON mode, the first attempt                               | yes   |
| `function_call`  | the same schema as a function the model must call (second request) | yes   |
| `text_fallback`  | the model's plain sentences, only if clean: under 300 characters, no JSON or markup fragments, in the profile's language | no |

If none works the caller hears the apology. After a `text_fallback` turn that
was asking for a slot which decides the outcome (own or rent, for example),
the next turn settles that slot first - filled from what was already said, or
asked again - before moving on.

`scripts/replay_llm.py --runs 4` sends realistic turns (including caller
questions in mid-conversation) to the real model and prints how many were
answered through each path. Run it before deploying any change to the prompt,
the per-turn instruction or the reply format: the unit tests use a fake model
and cannot see this kind of failure. Last run (2026-11-12, 44 turns): 38 json,
6 function_call, 0 fallback, 0 apology.

### Head start (eager end of turn)

Flux can also say *"they have probably finished"* (`EagerEndOfTurn`) slightly
before it is sure. When that happens we already run the LLM and TTS but
**hold the audio back** - and the slots it extracted are not applied:

- Flux confirms with `EndOfTurn` and the same words -> the audio is released
  immediately and the slots become official.
- The caller keeps talking (`TurnResumed`) -> the guess is thrown away. Nobody
  heard it, and it changed no state.

The cost is extra LLM requests for guesses that get discarded. Set
`FLUX_EAGER_EOT_THRESHOLD=0` to turn the head start off.

## Pacing: why audio is sent one frame every 20 ms

Text-to-speech produces four seconds of speech in about one second. If we
forwarded it as fast as it arrives, Asterisk would play it as a garbled
burst, because this path has no jitter buffer: whatever we send is played
right away.

The pacer therefore releases exactly one 320-byte frame (20 ms of sound)
every 20 ms. It does not "sleep 20 ms after each frame" - small delays would
add up and the speech would slowly fall behind. Instead frame number *n* of
an utterance is due at `start + n x 20 ms` on a monotonic clock, so a late
frame simply makes the next wait shorter and the timing cannot drift. If the
program stalls for more than 100 ms it restarts the schedule instead of
firing a burst to catch up.

## Barge-in: interrupting the bot
Interrupting works in two stages, because the reliable signal is slow.

**Stage 1 - pause (fast, local).** Speech-to-text only reports a new turn once
it recognises words; for a single short word like "wait" that took about two
seconds in a real call. So while the bot is speaking, `app/vad.py` watches the
caller's audio itself. A 20 ms frame counts as caller speech when both are true:

- **WebRTC VAD says it is voice.** This is Google's voice-activity detector
  (the `webrtcvad` library): tiny, fast (about 5 microseconds per frame) and far
  better than a plain volume threshold at telling voice from noise.
- **It is loud enough** (`BARGE_MIN_RMS`, default 500). The bot's own voice
  leaking back through the caller's phone is also voice, so the VAD alone would
  accept it. But echo is much quieter than someone talking into the microphone:
  measured on our own TTS audio, a copy 20 dB quieter still looked like voice
  on two thirds of the frames, with a level under about 370, while direct
  speech is typically 800-3700.

After 120 ms of such frames within 200 ms the pacer is **paused**: the bot goes
quiet within one frame, but nothing is thrown away yet.

**Stage 2 - confirm or resume.**

- Flux reports `StartOfTurn` -> a real interruption. The queued audio is
  dropped, the LLM and TTS requests still in flight are cancelled, and
  `[barge_in]` is logged (with how long the bot had already been paused).
- Nothing from Flux within `BARGE_CONFIRM_MS` (default 2.5 s) -> it was a
  cough, noise or echo. The bot **resumes exactly where it stopped** and
  `[barge_resume]` is logged.

What the bot had said before the interruption stays in the history (marked
interrupted). If the caller only says "wait" or "hold on", the model is told
to answer with a few words and listen, not to launch into a new long reply.

Tuning: on a speakerphone the echo can be louder than assumed. If the bot
pauses on its own voice (`false_pauses` in the summary line), raise
`BARGE_MIN_RMS`; `BARGE_VAD_ENABLED=false` falls back to Flux only.

## Latency budget

Goal: the bot starts answering about 1.2 s after you stop talking.

| Stage                         | Log field              | Typical      | What it is                                         |
|-------------------------------|------------------------|--------------|----------------------------------------------------|
| Turn detection + transcript   | `stt_ms`               | 250-500 ms   | your last word -> `EndOfTurn` arrives              |
| LLM thinks                    | `llm_first_token_ms`   | 300-450 ms   | request sent -> first word of the reply            |
| First sentence -> audio       | `tts_first_audio_ms`   | 200-350 ms   | first sentence sent to Aura -> first audio back    |
| **What you experience**       | `total_ms`             | 0.8-1.3 s    | your last word -> first reply frame sent to Asterisk |

One line per turn:

```
[turn 3] stt_ms=310 llm_first_token_ms=340 tts_first_audio_ms=230 total_ms=905 eager=yes
```

Notes on reading it:

- `total_ms` is measured directly (not added up), using the same audio clock
  as phase 2a: Flux says *"the last word ended at audio second 7.36"*, we know
  when that audio arrived, and we note when the first reply frame leaves.
- With `eager=yes` the LLM and TTS started before `stt_ms` was over, so the
  three parts overlap and add up to more than the total.
- Not included: the delay between your mouth and this server (phone, mobile
  network, Asterisk), and the same on the way back.
- While a call is active the server keeps its HTTPS connections to Groq and
  Deepgram open with a tiny request every few seconds (Deepgram closes an idle
  connection after about 5 s). Otherwise every reply after a pause would pay
  200-300 ms for a new TLS handshake.

## Cost counters (summary line at hangup)

```
call ended uuid=... duration=74.2s turns=6 total_ms_avg=930 total_ms_max=1240 barge_ins=1 stt_audio_s=74.0 llm_tokens=5210 tts_chars=412 ...
```

- `stt_audio_s` - audio sent to Flux (billed per second).
- `llm_tokens` - prompt + completion tokens over all LLM requests, including
  discarded head starts.
- `tts_chars` - characters sent to Aura during the call (billed per
  character). Cached phrases (greeting, apology, goodbye) are not counted
  because they are not synthesized again.

The whole conversation - every user and bot line, each turn's latency,
barge-ins, errors and the summary - is saved in `data/calls/<uuid>.jsonl`.

## Safety nets

- **Call length limit.** After `max_call_seconds` the bot stops whatever it is
  saying, plays a short goodbye and hangs up.
- **Ending the call.** See the next section: two outcomes, and the hang-up
  can be cancelled by the caller.
- **LLM or TTS failure mid-call.** The caller hears a pre-generated apology
  ("Sorry, I'm having a little trouble...") and the bot keeps listening. The
  apology is cached audio, so it works even when TTS is what failed.
- **STT failure.** Logged clearly; the call stays up but the bot can no longer
  hear the caller (no automatic reconnect yet).
- A bug in the reply pipeline is caught per turn; it cannot take down the
  call or the server.

## Ending the call
The model cannot end a call, choose a closing line, or ask "anything else?".
Its instructions say so, and it is not even given the closing texts.

1. **Outcome decided** (from the slots). The model is told to announce it:
   - *qualified*: thank the caller and say what happens next;
   - *not a fit*: explain why in one sentence and mention one thing from the
     knowledge that could still help (for example, that a landlord can
     contact us).
   The server then asks, once, "Is there anything else I can help you with?".
   If the model asks that itself in any wording, the sentence is dropped.
2. **After that the caller leads.** Questions and statements are simply
   answered. The server hangs up only when the caller's own words are a
   decline ("no, that's all", "nothing else", "I'm good", "thank you") or a
   goodbye - plain pattern matching in `app/intent.py`, not the model.
3. **Closing line.** Chosen by code from the outcome: `closing_qualified`,
   `closing_not_fit`, a variant's `closing`, or a neutral "Thank you for
   calling ..." if the outcome is still open. It is cached audio.
4. **Goodbye before the outcome was announced.** The goodbye sentence may
   carry an answer ("Yes. Thank you. Goodbye." after the call-back question).
   So the sentence is first sent to the model for slot extraction only -
   nothing the model says is spoken - then the code computes the outcome and
   picks the closing. This costs one LLM request (about half a second); if the
   model does not answer within 2.5 s the call closes on what is already
   known. After the outcome has been announced, a decline or goodbye closes
   immediately with no request.
5. **The closing line is finished once it has started.** Sound from the
   caller does not pause, cut or restart it. If the caller says something
   during the line or in the `END_SILENCE_MS` window after it (default 1.5 s),
   the hang-up waits for the transcript, which decides: "bye", "thanks", "ok"
   or talk not meant for the bot is ignored and the call ends as planned; only
   a real new request - a question, "wait", "one more thing" - stops the
   closing and is answered (`[hangup_cancelled]`). The rule is plain pattern
   matching (`interrupts_closing` in `app/intent.py`).
6. **Noise cannot hold the line open.** Sound picked up by the local voice
   detector in that window pushes the hang-up out by another
   `END_SILENCE_MS`, but never beyond `END_SILENCE_MAX_MS` (default 3 s) after
   the closing line ended. Every extension is logged as `[hangup_extended]`.

Details that matter:

- A bare "no" before the outcome is announced is just an answer ("Do you own
  the house?" - "No.") and goes to the model like any other reply.
- If the caller corrects themselves ("actually I own the house"), the slot
  changes, the outcome is recomputed and collecting continues.
- If an announcement is interrupted, the facts are kept but the announcement
  is made again on the next turn.
- The only other way a call ends is the `max_call_seconds` limit.

The summary line shows `outcome=qualified`, a variant such as
`qualified_no_callback`, `not_fit`, `incomplete` or `none`.

## Speech recognition accuracy

Phone audio (8 kHz) and accents make recognition harder; in a real call "an
apartment" was heard as "the Toronto apartment". Two measures:

- **Keyterms.** Each profile lists domain words (`apartment`, `rent`, `own`,
  `roof`, `electricity bill`, the business name...). They are sent to Flux as
  `keyterm` parameters when the call connects ("keyterm prompting"), which
  makes those words more likely to be recognised. Keep the list short and
  specific; up to 50 entries.
- **Clarify, don't act.** The model is told that transcripts contain mistakes:
  when something is surprising or out of context it asks a short clarifying
  question, and it never rejects a caller or ends the call over something it
  is not sure it heard correctly.

## Diagnostics

- Every Flux turn event (`StartOfTurn`, `Update`, `EagerEndOfTurn`,
  `TurnResumed`, `EndOfTurn`) is stored in the call file as `stt_event`, with
  its audio timestamps, how long after that audio arrived Flux reported it
  (`lag_ms`), and whether the bot was speaking. `StartOfTurn` also appears in
  the journal as a `[stt]` line.
- The model's exact output, markers included, is stored as `llm_raw`.
- **Debug audio recording** (`DEBUG_RECORD_AUDIO=true`): saves the caller's
  audio of each call to `data/recordings/<uuid>.wav`. **Off by default and for
  test calls only** - it is personal data. Files are owner-only, never
  committed (`data/` is git-ignored), and deleted automatically after 7 days;
  the clean-up runs at start-up and at every call, also when recording has
  been switched off again.

## Choices made (verified against the live APIs)

- **Flux at 8 kHz, no resampling.** Flux accepts `linear16` at 8000 Hz
  directly, so phone audio is forwarded unchanged.
- **Aura at 8 kHz, no resampling.** `encoding=linear16&sample_rate=8000&container=none`
  returns raw PCM over a chunked HTTP response.
- **Groq model `openai/gpt-oss-120b`.** The older Llama models are no longer
  offered. Of the current production models it and `gpt-oss-20b` had the same
  time-to-first-token in our test (~0.3-0.4 s) and the same free-tier limits;
  the larger one follows role and "only use the knowledge" rules more
  reliably. Reasoning is set to `low` and kept out of the output. Change it
  with `GROQ_MODEL`.
- **Free-tier limit to keep in mind:** 8,000 tokens per minute and 1,000
  requests per day. See "Token budget" below.

## Token budget

Every reply sends the whole system prompt plus the conversation so far, and
all of it counts against Groq's per-minute token limit. Three measures:

- **Short prompt.** About 2,800 characters including the profile. With the
  reply schema and the per-turn instruction a request costs roughly 1,100
  tokens at the start of a call, so about seven replies per minute fit.
- **Stable prefix for prompt caching.** The system prompt is built once at
  start-up and is byte-for-byte identical in every request of every call: no
  timestamps, no call IDs, nothing per turn. The conversation is only ever
  appended after it. Groq caches repeated prompt prefixes automatically for
  the gpt-oss models, and its documentation says cached tokens do not count
  towards the rate limit. Measured on this account: caching does happen (in
  256-token blocks) but **not on every request**, so it is treated as a bonus,
  not something to rely on. `llm_cached` in the summary line shows how many
  prompt tokens were served from the cache.
- **Head starts only when affordable.** Each Groq response reports how much of
  the per-minute allowance is left (`x-ratelimit-remaining-tokens`). When less
  than `GROQ_EAGER_MIN_BUDGET` (default 40 %) remains, the eager head start is
  switched off (`[eager_skipped]`), so no tokens are spent on guesses that may
  be thrown away. It comes back by itself as the allowance refills.

If the limit is hit anyway, the caller hears the apology phrase and the bot
keeps listening.

## Configuration (`.env`)

| Key                        | Default               | Purpose                                         |
|----------------------------|-----------------------|-------------------------------------------------|
| `GROQ_API_KEY`             | (empty)               | Required for the agent                          |
| `DEEPGRAM_API_KEY`         | (empty)               | Required (STT and TTS)                          |
| `ACTIVE_PROFILE`           | `lead_qualification`  | Which file in `profiles/` to use                |
| `AGENT_ENABLED`            | `true`                | `false` = transcription only (phase 2a)         |
| `GROQ_MODEL`               | `openai/gpt-oss-120b` |                                                 |
| `GROQ_EAGER_MIN_BUDGET`    | `0.4`                 | Skip head starts below this share of the per-minute token allowance |
| `AURA_MODEL`               | `aura-2-thalia-en`    | Voice                                           |
| `FLUX_MODEL`               | `flux-general-en`     |                                                 |
| `FLUX_EOT_THRESHOLD`       | `0.7`                 | Confidence needed to end a turn (0.5-1.0)       |
| `FLUX_EAGER_EOT_THRESHOLD` | `0.5`                 | Confidence for the head start; `0` = off        |
| `FLUX_EOT_TIMEOUT_MS`      | `3000`                | End the turn after this much silence regardless |
| `BARGE_VAD_ENABLED`        | `true`                | Local voice detection for fast barge-in         |
| `BARGE_VAD_MODE`           | `3`                   | WebRTC VAD strictness (0-3)                     |
| `BARGE_MIN_RMS`            | `500`                 | Loudness gate against the bot's own echo        |
| `BARGE_MIN_SPEECH_MS`      | `120`                 | Voice needed before the bot pauses              |
| `BARGE_CONFIRM_MS`         | `2500`                | Resume if Flux has not confirmed by then        |
| `END_SILENCE_MS`           | `1500`                | Silence after the closing line before hang-up   |
| `END_SILENCE_MAX_MS`       | `3000`                | Upper limit of that wait when there is sound on the line |
| `DEBUG_RECORD_AUDIO`       | `false`               | Record caller audio (test calls only, 7 days)   |

Without both API keys the server falls back to phase 2a behaviour
(transcription only).

## Code layout

| File                    | Role                                                           |
|-------------------------|----------------------------------------------------------------|
| `profiles/*.yaml`       | Business profiles                                              |
| `app/profiles.py`       | Load and validate profiles                                     |
| `app/prompt_builder.py` | Static system prompt + the per-turn instruction                |
| `app/stt_flux.py`       | Flux client: turn events                                       |
| `app/llm_groq.py`       | Groq streaming chat client                                     |
| `app/tts_aura.py`       | Aura client and the fixed-phrase cache                         |
| `app/text_utils.py`     | Sentence splitter for the LLM stream                           |
| `app/audio_out.py`      | Pacer (with pause / resume)                                    |
| `app/vad.py`            | Local voice detector for fast barge-in                         |
| `app/intent.py`         | Did the caller decline / say goodbye? (pattern matching)       |
| `app/slots.py`          | Slot definitions, validation, outcome, next question, plan     |
| `app/recording.py`      | Optional debug recording and its 7-day clean-up                |
| `app/agent.py`          | Conversation engine: state, turns, head start, barge-in, endings |
| `app/server.py`         | AudioSocket server; wires one agent per call                   |
