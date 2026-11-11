# AI Voice Agent

> Work in progress - placeholder README.

A self-hosted AI voice agent built on Asterisk (PJSIP).

## Status

- [x] Phase 0 - infrastructure: hardened Asterisk, one softphone endpoint, echo / playback test extensions
- [x] Phase 1 - AudioSocket bridge: ext 700 -> Python asyncio server (delayed echo)
- [x] Phase 2a - live speech-to-text: caller audio -> Deepgram streaming -> transcript log
- [ ] Phase 2b - LLM + text-to-speech
- [ ] Docker setup for easy local runs (`docker/`)

## Layout

| Path        | Purpose                                                              |
|-------------|----------------------------------------------------------------------|
| `asterisk/` | Asterisk config templates (placeholders, no secrets)                 |
| `deploy/`   | fail2ban jail and systemd resource-limit drop-in                     |
| `app/`      | Python application (AudioSocket server, STT)                                 |
| `docker/`   | Docker version (coming later)                                        |
| `docs/`     | Notes and documentation                                              |

See [docs/phase0-infrastructure.md](docs/phase0-infrastructure.md), [docs/phase1-audiosocket.md](docs/phase1-audiosocket.md) and [docs/phase2a-stt.md](docs/phase2a-stt.md).
