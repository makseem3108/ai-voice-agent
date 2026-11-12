# AI Voice Agent

> Work in progress - placeholder README.

A self-hosted AI voice agent built on Asterisk (PJSIP).

## Status

- [x] Phase 0 - infrastructure: hardened Asterisk, one softphone endpoint, echo / playback test extensions
- [x] Phase 1 - AudioSocket bridge: ext 700 -> Python asyncio server (delayed echo)
- [x] Phase 2a - live speech-to-text: caller audio -> Deepgram streaming -> transcript log
- [x] Phase 2b - the bot talks: Flux turn detection -> Groq LLM -> Aura TTS, profile-driven, with barge-in
- [ ] Phase 3 - tools / actions
- [ ] Docker setup for easy local runs (`docker/`)

## Layout

| Path        | Purpose                                                              |
|-------------|----------------------------------------------------------------------|
| `asterisk/` | Asterisk config templates (placeholders, no secrets)                 |
| `deploy/`   | fail2ban jail and systemd resource-limit drop-in                     |
| `app/`      | Python application (AudioSocket server, STT, LLM, TTS, agent)                                 |
| `docker/`   | Docker version (coming later)                                        |
| `profiles/` | Business profiles (YAML) that define who the agent is                |
| `docs/`     | Notes and documentation                                              |

See [docs/phase0-infrastructure.md](docs/phase0-infrastructure.md), [docs/phase1-audiosocket.md](docs/phase1-audiosocket.md), [docs/phase2a-stt.md](docs/phase2a-stt.md) and [docs/phase2b-agent.md](docs/phase2b-agent.md).
