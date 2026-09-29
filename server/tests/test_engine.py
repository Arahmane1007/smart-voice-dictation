import io
import math
import struct
import wave
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from svd_server.engine import (
    SAMPLE_RATE,
    AudioDecodeError,
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


def test_decode_resamples_to_16k() -> None:
    audio = decode_audio_bytes(make_wav(1.0, rate=44_100))
    assert abs(len(audio) - SAMPLE_RATE) < 200


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
