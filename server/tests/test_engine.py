import io
import math
import struct
import wave
from dataclasses import dataclass
from typing import Any

import av
import numpy as np
import pytest
from svd_server.engine import (
    SAMPLE_RATE,
    AudioDecodeError,
    AudioTooLongError,
    FasterWhisperEngine,
    Transcript,
    decode_audio_bytes,
)
from svd_server.settings import Settings

KEY = "k" * 40


def make_wav(seconds: float = 1.0, rate: int = SAMPLE_RATE) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        frames = b"".join(
            struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / rate)))
            for i in range(int(seconds * rate))
        )
        wav.writeframes(frames)
    return buffer.getvalue()


def test_decode_wav_at_16k() -> None:
    audio = decode_audio_bytes(make_wav(1.0))
    assert audio.dtype == np.float32
    assert abs(len(audio) - SAMPLE_RATE) < 200


def test_declared_duration_above_limit_raises_too_long() -> None:
    with pytest.raises(AudioTooLongError):
        decode_audio_bytes(make_wav(3.0), max_seconds=2)


def test_declared_duration_below_limit_passes() -> None:
    assert decode_audio_bytes(make_wav(1.0), max_seconds=2).size > 0


def test_decode_resamples_to_16k() -> None:
    audio = decode_audio_bytes(make_wav(1.0, rate=44_100))
    assert abs(len(audio) - SAMPLE_RATE) < 200


def make_flac(seconds: float, rate: int = SAMPLE_RATE) -> bytes:
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="flac") as container:
        stream = container.add_stream("flac", rate=rate, layout="mono")
        t = np.arange(int(seconds * rate))
        samples = (8000 * np.sin(2 * np.pi * 440 * t / rate)).astype(np.int16).reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
        frame.sample_rate = rate
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return buffer.getvalue()


def hide_flac_duration(data: bytes) -> bytes:
    """Zero the 36-bit `total samples` field of the FLAC STREAMINFO block.

    Layout: "fLaC" marker (4 bytes), metadata block header (4 bytes), then the
    34-byte STREAMINFO: min/max block size (2+2 bytes), min/max frame size (3+3),
    sample rate (20 bits), channels-1 (3 bits), bits per sample-1 (5 bits),
    total samples (36 bits), MD5 (16 bytes). The total samples field therefore
    starts at the low nibble of STREAMINFO byte 13 and runs through byte 17.
    Zero means "unknown", so the container no longer declares any duration.
    """
    start = data.find(b"fLaC")
    assert start >= 0 and data[start + 4] & 0x7F == 0  # first metadata block is STREAMINFO
    info = start + 8
    patched = bytearray(data)
    total = (patched[info + 13] & 0x0F) << 32 | int.from_bytes(patched[info + 14 : info + 18])
    assert total > 0
    patched[info + 13] &= 0xF0
    patched[info + 14 : info + 18] = b"\x00\x00\x00\x00"
    with av.open(io.BytesIO(bytes(patched)), mode="r") as container:
        assert container.duration is None  # the declared-duration fast path is blind now
    return bytes(patched)


def test_hidden_duration_is_still_capped_while_decoding() -> None:
    with pytest.raises(AudioTooLongError):
        decode_audio_bytes(hide_flac_duration(make_flac(5.0)), max_seconds=2)


def test_hidden_duration_short_audio_still_decodes() -> None:
    audio = decode_audio_bytes(hide_flac_duration(make_flac(1.0)), max_seconds=2)
    assert audio.dtype == np.float32
    assert abs(len(audio) - SAMPLE_RATE) < 200
    assert float(np.max(np.abs(audio))) <= 1.0


def test_too_long_is_not_a_decode_error() -> None:
    assert not issubclass(AudioTooLongError, AudioDecodeError)


@pytest.mark.parametrize("data", [b"", b"not audio at all", b"\x00\xff" * 500])
def test_decode_garbage_raises_audio_decode_error(data: bytes) -> None:
    with pytest.raises(AudioDecodeError):
        decode_audio_bytes(data)


@dataclass
class FakeSegment:
    text: str


@dataclass
class FakeInfo:
    language: str


class FakeModel:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.init_args = args
        self.init_kwargs = kwargs
        self.calls: list[dict[str, Any]] = []

    def transcribe(self, audio: np.ndarray, **kwargs: Any) -> tuple[Any, FakeInfo]:
        self.calls.append(kwargs)
        segments = iter([FakeSegment(" Bonjour"), FakeSegment(" le monde. ")])
        return segments, FakeInfo(language="fr")


def make_engine(**overrides: Any) -> tuple[FasterWhisperEngine, FakeModel]:
    created: list[FakeModel] = []

    def factory(*args: Any, **kwargs: Any) -> FakeModel:
        model = FakeModel(*args, **kwargs)
        created.append(model)
        return model

    engine = FasterWhisperEngine(Settings(api_keys=(KEY,), **overrides), model_factory=factory)
    return engine, created[0]


def test_model_is_created_from_settings() -> None:
    _, model = make_engine(whisper_model="medium", whisper_compute_type="int8", cpu_threads=3)
    assert model.init_args == ("medium",)
    assert model.init_kwargs == {"device": "cpu", "compute_type": "int8", "cpu_threads": 3}


def test_cpu_threads_zero_means_all_cores(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("svd_server.engine.os.cpu_count", lambda: 12)
    _, model = make_engine(cpu_threads=0)
    assert model.init_kwargs["cpu_threads"] == 12


def test_transcribe_joins_segments_and_forwards_options() -> None:
    engine, model = make_engine(whisper_beam_size=1)
    audio = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)
    result = engine.transcribe(audio, language="fr", prompt="Polyspace, UCAD")
    assert result == Transcript(text="Bonjour le monde.", language="fr", duration=2.0)
    assert model.calls[-1] == {
        "language": "fr",
        "initial_prompt": "Polyspace, UCAD",
        "beam_size": 1,
        "vad_filter": True,
    }


def test_warmup_runs_the_model_once() -> None:
    engine, model = make_engine()
    engine.warmup()
    assert len(model.calls) == 1
    assert model.calls[0]["vad_filter"] is False


@pytest.mark.slow
def test_real_tiny_model_transcribes_silence() -> None:
    engine = FasterWhisperEngine(Settings(api_keys=(KEY,), whisper_model="tiny"))
    engine.warmup()
    result = engine.transcribe(decode_audio_bytes(make_wav(1.0)), language="fr", prompt=None)
    assert result.language == "fr"
    assert result.duration == pytest.approx(1.0, abs=0.05)
