"""Tests for the AudioSocket frame parser / encoder."""

from __future__ import annotations

import uuid

import pytest

from app.audiosocket import (
    Frame,
    FrameParser,
    FrameType,
    audio_duration_s,
    encode_frame,
    parse_uuid,
)

CALL_UUID = uuid.UUID("0f8fad5b-d9cb-469f-a165-70867728950e")
AUDIO_20MS = bytes(range(256)) + bytes(64)  # 320 bytes = 20 ms of slin16 @ 8 kHz


# ----------------------------------------------------------------- valid frames
def test_uuid_frame() -> None:
    frames = FrameParser().feed(b"\x01\x00\x10" + CALL_UUID.bytes)
    assert frames == [Frame(0x01, CALL_UUID.bytes)]
    assert frames[0].kind is FrameType.UUID
    assert parse_uuid(frames[0].payload) == CALL_UUID


def test_audio_frame() -> None:
    frames = FrameParser().feed(b"\x10\x01\x40" + AUDIO_20MS)
    assert len(frames) == 1
    assert frames[0].kind is FrameType.AUDIO
    assert frames[0].payload == AUDIO_20MS
    assert audio_duration_s(frames[0].payload) == pytest.approx(0.020)


def test_error_frame_with_code() -> None:
    (frame,) = FrameParser().feed(b"\xff\x00\x01\x02")
    assert frame.kind is FrameType.ERROR
    assert frame.payload == b"\x02"


def test_several_frames_in_one_read() -> None:
    stream = (
        encode_frame(FrameType.UUID, CALL_UUID.bytes)
        + encode_frame(FrameType.AUDIO, AUDIO_20MS)
        + encode_frame(FrameType.AUDIO, AUDIO_20MS[::-1])
    )
    parser = FrameParser()
    frames = parser.feed(stream)
    assert [f.kind for f in frames] == [FrameType.UUID, FrameType.AUDIO, FrameType.AUDIO]
    assert frames[2].payload == AUDIO_20MS[::-1]
    assert parser.pending == 0


def test_length_is_big_endian_and_max_payload_roundtrips() -> None:
    payload = bytes(0xFFFF)
    encoded = encode_frame(FrameType.AUDIO, payload)
    assert encoded[:3] == b"\x10\xff\xff"
    (frame,) = FrameParser().feed(encoded)
    assert frame.payload == payload


def test_encode_roundtrip() -> None:
    frame = Frame(FrameType.AUDIO, AUDIO_20MS)
    assert FrameParser().feed(frame.encode()) == [frame]


def test_encode_rejects_oversized_payload_and_bad_type() -> None:
    with pytest.raises(ValueError):
        encode_frame(FrameType.AUDIO, bytes(0x10000))
    with pytest.raises(ValueError):
        encode_frame(0x100, b"")


def test_parse_uuid_rejects_wrong_length() -> None:
    with pytest.raises(ValueError):
        parse_uuid(b"\x00" * 15)


# ---------------------------------------------------------------- partial reads
def test_empty_feed_returns_nothing() -> None:
    parser = FrameParser()
    assert parser.feed(b"") == []
    assert parser.pending == 0


def test_partial_header_then_rest() -> None:
    encoded = encode_frame(FrameType.AUDIO, AUDIO_20MS)
    parser = FrameParser()
    assert parser.feed(encoded[:2]) == []
    assert parser.pending == 2
    assert parser.feed(encoded[2:]) == [Frame(FrameType.AUDIO, AUDIO_20MS)]
    assert parser.pending == 0


def test_partial_payload_then_rest() -> None:
    encoded = encode_frame(FrameType.AUDIO, AUDIO_20MS)
    parser = FrameParser()
    assert parser.feed(encoded[:100]) == []
    assert parser.feed(encoded[100:200]) == []
    assert parser.feed(encoded[200:]) == [Frame(FrameType.AUDIO, AUDIO_20MS)]


def test_byte_by_byte_stream() -> None:
    stream = (
        encode_frame(FrameType.UUID, CALL_UUID.bytes)
        + encode_frame(FrameType.AUDIO, AUDIO_20MS)
        + encode_frame(FrameType.HANGUP)
    )
    parser = FrameParser()
    frames: list[Frame] = []
    for i in range(len(stream)):
        frames += parser.feed(stream[i : i + 1])
    assert [f.kind for f in frames] == [FrameType.UUID, FrameType.AUDIO, FrameType.HANGUP]
    assert parser.pending == 0


def test_read_boundary_in_the_middle_of_second_frame() -> None:
    first = encode_frame(FrameType.AUDIO, AUDIO_20MS)
    second = encode_frame(FrameType.AUDIO, AUDIO_20MS[::-1])
    parser = FrameParser()
    assert parser.feed(first + second[:50]) == [Frame(FrameType.AUDIO, AUDIO_20MS)]
    assert parser.pending == 50
    assert parser.feed(second[50:]) == [Frame(FrameType.AUDIO, AUDIO_20MS[::-1])]


# ----------------------------------------------------------------- unknown type
def test_unknown_type_is_returned_with_kind_none() -> None:
    (frame,) = FrameParser().feed(b"\x42\x00\x03abc")
    assert frame.type == 0x42
    assert frame.kind is None
    assert frame.payload == b"abc"


def test_unknown_type_does_not_desync_the_stream() -> None:
    stream = (
        encode_frame(FrameType.AUDIO, AUDIO_20MS)
        + encode_frame(0x7E, b"future extension")
        + encode_frame(FrameType.AUDIO, AUDIO_20MS)
        + encode_frame(FrameType.HANGUP)
    )
    frames = FrameParser().feed(stream)
    assert [f.kind for f in frames] == [FrameType.AUDIO, None, FrameType.AUDIO, FrameType.HANGUP]


# ----------------------------------------------------------------------- hangup
def test_hangup_frame() -> None:
    (frame,) = FrameParser().feed(b"\x00\x00\x00")
    assert frame.kind is FrameType.HANGUP
    assert frame.payload == b""


def test_encode_hangup() -> None:
    assert encode_frame(FrameType.HANGUP) == b"\x00\x00\x00"


def test_hangup_after_audio_and_trailing_bytes_stay_buffered() -> None:
    parser = FrameParser()
    frames = parser.feed(encode_frame(FrameType.AUDIO, AUDIO_20MS) + b"\x00\x00\x00" + b"\x10\x01")
    assert [f.kind for f in frames] == [FrameType.AUDIO, FrameType.HANGUP]
    assert parser.pending == 2
