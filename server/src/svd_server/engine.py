"""Transcription engine: audio decoding and faster-whisper."""

import gc
import io
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import av
import av.error
import numpy as np
from av.audio.resampler import AudioResampler
from faster_whisper import WhisperModel

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


def _stream_decode(data: bytes, max_samples: int | None) -> np.ndarray:
    """Decode the first audio stream to mono s16 at 16 kHz, frame by frame.

    Samples are counted as they are produced, so a container that hides or lies
    about its duration is stopped as soon as it exceeds `max_samples`, instead of
    being decoded (and buffered) in full.
    """
    resampler = AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
    chunks: list[np.ndarray] = []
    count = 0

    def keep(frames: list[av.AudioFrame]) -> None:
        nonlocal count
        for frame in frames:
            chunk = frame.to_ndarray().reshape(-1)
            count += chunk.size
            if max_samples is not None and count > max_samples:
                raise AudioTooLongError("audio longer than the allowed maximum")
            chunks.append(chunk)

    try:
        with av.open(io.BytesIO(data), mode="r", metadata_errors="ignore") as container:
            if not container.streams.audio:
                raise AudioDecodeError("no audio stream")
            stream = container.streams.audio[0]
            for packet in container.demux(stream):
                try:
                    frames = packet.decode()
                except av.error.InvalidDataError:
                    continue  # skip a corrupt packet, like faster-whisper does
                for frame in frames:
                    keep(resampler.resample(frame))
            keep(resampler.resample(None))  # flush the samples buffered by the resampler
    finally:
        del resampler
        gc.collect()  # PyAV resampler objects are otherwise not freed promptly

    if not chunks:
        raise AudioDecodeError("no audio samples")
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def decode_audio_bytes(data: bytes, max_seconds: float | None = None) -> np.ndarray:
    """Decode any audio container to mono float32 at 16 kHz, entirely in memory.

    With `max_seconds`, a container declaring a longer duration is rejected before
    decoding, and decoding itself stops as soon as the decoded length exceeds it
    (the declared duration is attacker-controlled and may be missing or false).
    """
    if not data:
        raise AudioDecodeError("empty audio")
    max_samples: int | None = None
    if max_seconds is not None:
        declared = _declared_seconds(data)
        if declared is not None and declared > max_seconds:
            raise AudioTooLongError(f"audio longer than {max_seconds} seconds")
        max_samples = int(max_seconds * SAMPLE_RATE)
    try:
        audio = _stream_decode(data, max_samples)
    except (AudioTooLongError, AudioDecodeError):
        raise
    except Exception as exc:  # PyAV raises many different error types
        raise AudioDecodeError("could not decode audio") from exc
    if audio.size == 0:
        raise AudioDecodeError("no audio samples")
    return audio


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
