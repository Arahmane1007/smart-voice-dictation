"""Transcription engine: audio decoding and faster-whisper."""

import io
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import av
import numpy as np
from faster_whisper import WhisperModel, decode_audio

from svd_server.settings import Settings

SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str
    duration: float


class AudioDecodeError(Exception):
    """The uploaded bytes are not decodable audio."""


class AudioTooLongError(Exception):
    """The audio container declares a duration above the allowed maximum."""


class TranscriptionEngine(Protocol):
    def warmup(self) -> None: ...

    def decode(self, data: bytes) -> np.ndarray: ...

    def transcribe(
        self, audio: np.ndarray, language: str | None, prompt: str | None
    ) -> Transcript: ...


def _declared_seconds(data: bytes) -> float | None:
    """Duration announced by the container, without decoding any sample."""
    try:
        with av.open(io.BytesIO(data), mode="r") as container:
            if container.duration is None:
                return None
            return float(container.duration) / av.time_base
    except Exception:  # undecodable input is reported by the real decode below
        return None


def decode_audio_bytes(data: bytes, max_seconds: float | None = None) -> np.ndarray:
    """Decode any audio container to mono float32 at 16 kHz, entirely in memory.

    With `max_seconds`, a container declaring a longer duration is rejected before decoding.
    """
    if not data:
        raise AudioDecodeError("empty audio")
    if max_seconds is not None:
        declared = _declared_seconds(data)
        if declared is not None and declared > max_seconds:
            raise AudioTooLongError(f"audio longer than {max_seconds} seconds")
    try:
        audio = decode_audio(io.BytesIO(data), sampling_rate=SAMPLE_RATE)
    except Exception as exc:  # PyAV raises many different error types
        raise AudioDecodeError("could not decode audio") from exc
    if not isinstance(audio, np.ndarray) or audio.size == 0:
        raise AudioDecodeError("no audio samples")
    return audio.astype(np.float32, copy=False)


class FasterWhisperEngine:
    def __init__(
        self,
        settings: Settings,
        model_factory: Callable[..., Any] = WhisperModel,
    ) -> None:
        self._beam_size = settings.whisper_beam_size
        self._max_audio_seconds = settings.max_audio_seconds
        self._model = model_factory(
            settings.whisper_model,
            device="cpu",
            compute_type=settings.whisper_compute_type,
            cpu_threads=settings.cpu_threads or os.cpu_count() or 1,
        )

    def warmup(self) -> None:
        """Run one short inference so the first real dictation is not slowed down."""
        silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
        segments, _ = self._model.transcribe(silence, language="en", beam_size=1, vad_filter=False)
        list(segments)

    def decode(self, data: bytes) -> np.ndarray:
        return decode_audio_bytes(data, max_seconds=self._max_audio_seconds)

    def transcribe(self, audio: np.ndarray, language: str | None, prompt: str | None) -> Transcript:
        segments, info = self._model.transcribe(
            audio,
            language=language,
            initial_prompt=prompt,
            beam_size=self._beam_size,
            vad_filter=True,
        )
        text = "".join(segment.text for segment in segments).strip()
        return Transcript(text=text, language=info.language, duration=len(audio) / SAMPLE_RATE)
