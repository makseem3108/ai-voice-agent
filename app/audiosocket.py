"""AudioSocket wire protocol: frame model, encoder and incremental parser.

Every frame on the TCP stream is::

    +--------+-----------------+----------------------+
    | type   | payload length  | payload              |
    | 1 byte | 2 bytes, big-   | ``length`` bytes     |
    |        | endian unsigned |                      |
    +--------+-----------------+----------------------+

This module is pure (no I/O), so it can be tested without a socket.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

HEADER_SIZE = 3
MAX_PAYLOAD = 0xFFFF

#: Audio is signed linear 16-bit little-endian, 8 kHz, mono.
SAMPLE_RATE_HZ = 8000
BYTES_PER_SAMPLE = 2


class FrameType(IntEnum):
    """Frame types defined by the AudioSocket protocol."""

    HANGUP = 0x00  # either side: terminate the call, empty payload
    UUID = 0x01  # Asterisk -> app, first frame: 16-byte binary call UUID
    AUDIO = 0x10  # both directions: slin16 @ 8 kHz
    ERROR = 0xFF  # Asterisk -> app: optional 1-byte error code


@dataclass(frozen=True)
class Frame:
    """One protocol frame. ``type`` is kept raw so unknown types survive parsing."""

    type: int
    payload: bytes = b""

    @property
    def kind(self) -> Optional[FrameType]:
        """The known :class:`FrameType`, or ``None`` for a type we do not know."""
        try:
            return FrameType(self.type)
        except ValueError:
            return None

    def encode(self) -> bytes:
        return encode_frame(self.type, self.payload)


def encode_frame(frame_type: int, payload: bytes = b"") -> bytes:
    """Serialize one frame (header + payload)."""
    if not 0 <= frame_type <= 0xFF:
        raise ValueError(f"frame type out of range: {frame_type}")
    if len(payload) > MAX_PAYLOAD:
        raise ValueError(f"payload too large: {len(payload)} > {MAX_PAYLOAD}")
    return bytes((frame_type,)) + len(payload).to_bytes(2, "big") + payload


def parse_uuid(payload: bytes) -> uuid.UUID:
    """Decode the payload of a UUID frame."""
    if len(payload) != 16:
        raise ValueError(f"UUID payload must be 16 bytes, got {len(payload)}")
    return uuid.UUID(bytes=payload)


def audio_duration_s(payload: bytes) -> float:
    """Playback duration of an audio payload in seconds."""
    return len(payload) / (SAMPLE_RATE_HZ * BYTES_PER_SAMPLE)


class FrameParser:
    """Incremental parser: feed it whatever TCP delivers, get whole frames back.

    TCP is a byte stream, so a single read may contain half a frame, exactly
    one frame, or several. Bytes of an incomplete frame are buffered until the
    rest arrives. Unknown frame types are *not* an error: the length header
    tells us how to skip them, so they are returned as frames whose
    :attr:`Frame.kind` is ``None`` and the stream stays in sync.
    """

    def __init__(self) -> None:
        self._buf = bytearray()

    @property
    def pending(self) -> int:
        """Number of buffered bytes that do not yet form a complete frame."""
        return len(self._buf)

    def feed(self, data: bytes) -> list[Frame]:
        """Add ``data`` to the buffer and return every frame now complete."""
        self._buf += data
        frames: list[Frame] = []
        buf = self._buf
        offset = 0
        while len(buf) - offset >= HEADER_SIZE:
            length = int.from_bytes(buf[offset + 1 : offset + HEADER_SIZE], "big")
            end = offset + HEADER_SIZE + length
            if len(buf) < end:
                break
            frames.append(Frame(buf[offset], bytes(buf[offset + HEADER_SIZE : end])))
            offset = end
        if offset:
            del buf[:offset]
        return frames
