# Phase 1 - AudioSocket bridge (Asterisk <-> Python)

Extension `700` hands the call's audio to a Python asyncio server over a local
TCP connection. For now the server is a **delayed echo**: you hear yourself
about one second late (extension `600` is Asterisk's own instant echo).

```
softphone --SIP/RTP--> Asterisk --TCP 127.0.0.1:9092 (AudioSocket)--> app/server.py
```

## The AudioSocket protocol

AudioSocket is a deliberately tiny protocol that ships with Asterisk
(`res_audiosocket` + `app_audiosocket`). Asterisk opens **one TCP connection
per call** and both sides exchange frames with a 3-byte header:

| Bytes | Field   | Meaning                                  |
|-------|---------|------------------------------------------|
| 1     | type    | frame type (table below)                 |
| 2     | length  | payload length, unsigned, **big-endian** |
| n     | payload | `length` bytes                           |

| Type   | Name   | Direction         | Payload                                              |
|--------|--------|-------------------|------------------------------------------------------|
| `0x00` | hangup | either            | empty - end the call                                 |
| `0x01` | UUID   | Asterisk -> app   | 16 raw bytes, always the first frame of a connection |
| `0x10` | audio  | both              | signed linear 16-bit little-endian, 8 kHz, mono      |
| `0xff` | error  | Asterisk -> app   | optional 1-byte error code                           |

Things worth knowing:

- Asterisk sends audio in 20 ms frames: 160 samples = **320 bytes**, 50 frames per second.
  It transcodes to and from the caller's codec (G.711 here) itself.
- Audio the app sends back is played to the caller as it arrives; there is no
  jitter buffer, so the app is responsible for pacing.
- TCP is a byte stream: one `read()` can return half a frame or several. The
  parser (`app/audiosocket.py`) buffers partial frames, and uses the length
  field to skip frame types it does not know, so the stream never loses sync.
- There is **no authentication and no encryption**. The server therefore binds
  to `127.0.0.1` only, and that is hard-coded rather than configurable.

## The call UUID

`AudioSocket()` needs a UUID to identify the call. The packaged Asterisk 18.10
has no `UUID()` dialplan function, so extension 700 builds one:

```
Set(H=${MD5(${UNIQUEID}-${EPOCH}-${RAND()}-${RAND()})})
Set(CALL_UUID=${H:0:8}-${H:8:4}-4${H:13:3}-a${H:17:3}-${H:20:12})
```

`UNIQUEID` is already unique per call on this server; hashing it with the time
and two random numbers gives 32 hex digits, which are cut into the
8-4-4-4-12 UUID layout with the version (`4`) and variant (`a`) digits fixed.
No shell command is run from the dialplan (`System()` / `SHELL()` stay
unloaded). The UUID is a correlation ID for logs, not a secret.

## Code layout

| File                 | Role                                                        |
|----------------------|-------------------------------------------------------------|
| `app/audiosocket.py` | protocol only: frame types, encoder, incremental parser     |
| `app/server.py`      | asyncio TCP server, one `CallSession` per call, delayed echo |
| `app/config.py`      | settings from `.env` / environment                          |
| `tests/`             | pytest suite for the parser                                 |

Optional `.env` keys: `AUDIOSOCKET_PORT` (9092), `ECHO_DELAY_MS` (1000), `LOG_LEVEL` (INFO).

## Per-call log line

Written when the call ends:

```
call ended uuid=<uuid> duration=12.34s frames_in=617 frames_out=567 avg_jitter_ms=0.41 reason=hangup
```

`reason` is `hangup` when Asterisk sends a hangup frame and `eof` when it just
closes the connection. Asterisk 18.10 does the latter when the caller hangs up
(and logs a harmless `Failed to forward frame to channel` warning for the echo
frame that was in flight), so `eof` is the normal ending for a real call.

`avg_jitter_ms` is the mean absolute difference between each inter-frame
arrival gap and the gap the audio implies (20 ms). `frames_out` is lower than
`frames_in` by roughly one second's worth (about 50 frames), because the last
second of audio is still waiting in the delay line when the caller hangs up.

## Run / test

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest
sudo systemctl status voice-agent        # unit file: deploy/systemd/voice-agent.service
journalctl -u voice-agent -f
```
