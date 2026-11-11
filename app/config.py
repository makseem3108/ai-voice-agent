"""Runtime configuration, read from the process environment and ``.env``.

Real environment variables win over ``.env``. Only the keys this app needs are
read; the file may hold other secrets (e.g. SIP credentials) that are ignored.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"

#: The AudioSocket protocol has no authentication or encryption, so the server
#: is loopback-only by design. This is intentionally not configurable.
BIND_HOST = "127.0.0.1"


@dataclass(frozen=True)
class Settings:
    host: str = BIND_HOST
    port: int = 9092
    echo_delay_ms: int = 1000
    log_level: str = "INFO"

    @property
    def echo_delay_s(self) -> float:
        return self.echo_delay_ms / 1000.0


def read_env_file(path: Path) -> dict[str, str]:
    """Parse a simple ``KEY=VALUE`` file. Missing file -> empty dict."""
    values: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return values
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def _int(values: Mapping[str, str], key: str, default: int, lo: int, hi: int) -> int:
    raw = values.get(key)
    if raw is None or raw == "":
        return default
    try:
        number = int(raw)
    except ValueError:
        raise ValueError(f"{key} must be an integer, got {raw!r}") from None
    if not lo <= number <= hi:
        raise ValueError(f"{key} must be between {lo} and {hi}, got {number}")
    return number


def load_settings(
    env_file: Path = DEFAULT_ENV_FILE,
    environ: Optional[Mapping[str, str]] = None,
) -> Settings:
    """Build :class:`Settings` from ``.env`` overlaid with the environment."""
    values = read_env_file(env_file)
    values.update(os.environ if environ is None else environ)
    return Settings(
        port=_int(values, "AUDIOSOCKET_PORT", 9092, 1024, 65535),
        echo_delay_ms=_int(values, "ECHO_DELAY_MS", 1000, 0, 10_000),
        log_level=(values.get("LOG_LEVEL") or "INFO").upper(),
    )
