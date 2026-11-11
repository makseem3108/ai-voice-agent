# AI Voice Agent

> Work in progress - placeholder README.

A self-hosted AI voice agent built on Asterisk (PJSIP).

## Status

- [x] Phase 0 - infrastructure: hardened Asterisk, one softphone endpoint, echo / playback test extensions
- [x] Phase 1 - AudioSocket bridge: ext 700 -> Python asyncio server (delayed echo)
- [ ] Phase 2 - speech pipeline
- [ ] Docker setup for easy local runs (`docker/`)

## Layout

| Path        | Purpose                                                              |
|-------------|----------------------------------------------------------------------|
| `asterisk/` | Asterisk config templates (placeholders, no secrets)                 |
| `deploy/`   | fail2ban jail and systemd resource-limit drop-in                     |
| `app/`      | Python application (AudioSocket server)                                 |
| `docker/`   | Docker version (coming later)                                        |
| `docs/`     | Notes and documentation                                              |

See [docs/phase0-infrastructure.md](docs/phase0-infrastructure.md) and [docs/phase1-audiosocket.md](docs/phase1-audiosocket.md).
