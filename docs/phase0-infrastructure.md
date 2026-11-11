# Phase 0 - infrastructure

Native Asterisk 18 (Ubuntu 22.04 package) with a deliberately tiny surface.

## What is configured

- **Modules**: `autoload = no`; only PJSIP, G.711/GSM codecs, `Echo` and `Playback` are loaded.
  chan_sip, IAX2, AMI, ARI and the HTTP server are not running.
- **Transport**: PJSIP over UDP 5060. RTP restricted to UDP 10000-10100.
- **Endpoint `1001`**: softphone behind home NAT (`rtp_symmetric`, `force_rport`, `rewrite_contact`, no direct media).
- **Dialplan** (`from-internal`): `600` echo test, `601` plays hello-world, `700` AudioSocket bridge (phase 1). No outbound routes.
- **Hardening**: no anonymous endpoint, username-only endpoint matching, identical 401 for
  known and unknown users, security log + fail2ban (5 failures -> 24 h ban on the SIP/RTP ports),
  firewall opens only 5060/udp and 10000-10100/udp.
- **Resource limits**: systemd drop-in, `MemoryMax=512M`, `CPUQuota=100%`.

## Rendering the config

Files in `asterisk/` contain placeholders that come from `.env`:

| Placeholder              | `.env` key          |
|--------------------------|---------------------|
| `__PUBLIC_IP__`          | `PUBLIC_IP`         |
| `__SIP_1001_PASSWORD__`  | `SIP_1001_PASSWORD` |

```bash
set -a; . ./.env; set +a
sed -e "s/__PUBLIC_IP__/$PUBLIC_IP/g" -e "s/__SIP_1001_PASSWORD__/$SIP_1001_PASSWORD/g" \
    asterisk/pjsip.conf | sudo install -o root -g asterisk -m 640 /dev/stdin /etc/asterisk/pjsip.conf
```

The other files in `asterisk/` are copied to `/etc/asterisk/` unchanged.
