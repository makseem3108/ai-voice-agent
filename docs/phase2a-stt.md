# Phase 2a - live speech-to-text (Deepgram streaming)

Call extension `700` and talk: your words appear in the service log while you
are still speaking. There is no LLM and no voice reply yet, so the caller
hears silence.

```
phone --RTP--> Asterisk --AudioSocket (TCP, loopback)--> app/server.py --WebSocket (TLS)--> Deepgram
                                                              |                                |
                                                              +<------- transcript events -----+
                                                              v
                                              journal log + data/calls/<uuid>.jsonl
```

## How streaming STT works

A normal ("batch") transcription API takes a finished recording and returns
the text. A **streaming** API is a phone line that stays open: we push audio
in as it happens, and the service pushes text back as soon as it has an
opinion. For a voice agent this is the only workable option, because we
cannot wait for the caller to hang up before finding out what they said.

Every 20 ms Asterisk hands us 320 bytes of audio (16-bit, 8 kHz, mono) and we
forward those bytes unchanged over one WebSocket per call.

### Interim vs final

While you talk, Deepgram keeps guessing:

```
[interim] Hi I'd like
[interim] Hi I'd like to book an
[final]   Hi, I'd like to book an appointment for Tuesday.
```

- An **interim** result is a draft. It arrives fast (a few hundred ms behind
  your voice) but may still change: later words give the model context, so
  "to book in" can become "to book an appointment".
- A **final** result is the committed text for one stretch of audio. It will
  not change again. Punctuation and number formatting (`smart_format`) are
  applied here.

Interims are good for showing liveness or for reacting early; decisions
should be based on finals.

### Endpointing: deciding that you stopped talking

The hard question in a conversation is not "what was said" but "are they
done?". Deepgram answers it in two ways, and we log both:

| Log line          | Deepgram signal                 | Meaning                                                                 |
|-------------------|---------------------------------|-------------------------------------------------------------------------|
| `[speech_start]`  | `SpeechStarted` (VAD event)     | Voice detected after silence.                                           |
| `[speech_end]`    | `speech_final: true` on a result | **Endpointing**: about 400 ms of silence followed speech (`endpointing=400`). The line shows how much silence was actually observed. |
| `[utterance_end]` | `UtteranceEnd`                  | Backup signal: no new *words* for 1000 ms (`utterance_end_ms=1000`). Works on word timings, so it still fires on a noisy line where silence is never detected. |

A shorter endpointing value makes the agent answer faster but cuts people
off when they pause to think; a longer one feels sluggish. 400 ms is a
reasonable starting point for phone conversations.

Note that a `[final]` is not always the end of a sentence: during a long
stretch of speech Deepgram also finalises text every few seconds so it never
has to hold too much audio. Only a final followed by `[speech_end]` means the
caller paused.

## Latency: what we measure and how

**STT latency = how long after the caller stopped talking we have their final words.**

Deepgram does not tell us wall-clock times. It timestamps everything in
*audio seconds since the stream began* ("the last word ended at 7.36 s").
To turn that into real time we keep an **audio clock**:

1. Each audio frame gets the wall-clock time at which it arrived from Asterisk.
2. As frames are forwarded we add up their duration. So we know, for example,
   that audio second 7.36 lies in frame 368, which arrived at 14:02:11.480.
3. When a final arrives we take the end time of its **last word**
   (from Deepgram's word timings), look up when that audio arrived, and subtract:

```
stt_latency = (time the final transcript arrived) - (time we received the audio where the last word ended)
```

What is inside that number:

- Deepgram waiting for silence before it commits (most of it, roughly the 400 ms endpointing),
- Deepgram's processing time,
- the network round trip between this server and Deepgram.

What is **not** inside it: the delay from the caller's mouth to this server
(phone, mobile network, Asterisk). We cannot observe that from here.

Accuracy is limited by the 20 ms frame size and by how precisely Deepgram
places the end of the last word (a few tens of ms).

## Cost

Deepgram bills by audio duration. `audio_sent_s` in the summary line is the
total audio we forwarded, including silence, and matches the duration
Deepgram reports back when the stream closes (`deepgram_reported_s` in the
JSONL file).

## What gets recorded

One log line per event, prefixed with the first 8 characters of the call UUID,
and a summary when the call ends:

```
call ended uuid=<uuid> duration=31.20s finals=4 stt_latency_avg_ms=512 stt_latency_max_ms=690 audio_sent_s=31.14 frames_in=1557 ...
```

The same events are written as JSON lines to `data/calls/<uuid>.jsonl`
(`ts` = UTC time, `t` = seconds since the call started). Transcripts are
personal data: the directory is owner-only and `data/` is git-ignored.

## Why raw WebSocket instead of the official SDK

The app talks to Deepgram with the small `websockets` library rather than
`deepgram-sdk`:

- The streaming protocol is tiny: binary audio out, JSON events in, and three
  control messages (`KeepAlive`, `CloseStream`, `Finalize`). The whole client
  is one file.
- One small dependency instead of the SDK and its HTTP/typing stack, which
  keeps the footprint low on a shared server and the supply chain short.
- We control timing, buffering and error handling exactly, which matters
  for measuring latency honestly.
- The parsing code is pure functions, so it is tested without a network.

The trade-off: if Deepgram changes its API we update the client ourselves.

## Behaviour when things go wrong

- **No API key / `STT_ENABLED=false`**: calls work, nothing is transcribed.
- **Deepgram unreachable or the connection drops mid-call**: one clear
  `[stt_error]` line, transcription stops for that call, the call itself
  continues, the server keeps running. There is no automatic reconnect yet.
- **Quiet line**: if no audio flows for 4 s we send `KeepAlive` so Deepgram
  does not close the stream (it does so after about 10 s of nothing).
- **Hangup**: we send `CloseStream`, wait up to 3 s for the last results, then
  write the summary.
- The API key is sent only in the `Authorization` header, never in the URL,
  and never logged (the `websockets` logger is pinned to INFO because its
  DEBUG output includes handshake headers).

## Configuration (`.env`)

| Key                         | Default  | Purpose                                         |
|-----------------------------|----------|-------------------------------------------------|
| `DEEPGRAM_API_KEY`          | (empty)  | Required for transcription                      |
| `STT_ENABLED`               | `true`   | Switch STT off without removing the key         |
| `DEEPGRAM_MODEL`            | `nova-3` |                                                 |
| `DEEPGRAM_LANGUAGE`         | `en`     |                                                 |
| `DEEPGRAM_ENDPOINTING_MS`   | `400`    | Silence that ends an utterance                  |
| `DEEPGRAM_UTTERANCE_END_MS` | `1000`   | Word-gap backup signal                          |
| `ECHO_MODE`                 | `false`  | Debug: play the caller back after `ECHO_DELAY_MS` |

## Code layout

| File                  | Role                                                             |
|-----------------------|------------------------------------------------------------------|
| `app/stt_deepgram.py` | Deepgram client: message parsing, audio clock, latency, WebSocket |
| `app/call_log.py`     | JSONL recorder and log-line formatting                           |
| `app/server.py`       | Wires AudioSocket frames to the STT stream; per-call summary     |
| `tests/test_stt_deepgram.py` | Parsing, latency and failure handling with a fake WebSocket |
