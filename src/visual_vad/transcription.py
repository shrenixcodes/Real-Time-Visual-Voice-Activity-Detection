"""Optional, local speech-to-text gated by visual VAD events.

The core V-VAD module never opens a microphone. This adapter is only used by
the webcam demo when ``--stt`` is selected and requires a local Vosk model.
"""

from __future__ import annotations

import json
import queue
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class TranscriptSnapshot:
    finalized: tuple[str, ...]
    partial: str
    listening: bool


class VisualGateTranscriber(Protocol):
    """Shared contract for optional audio transcription backends."""

    def start_utterance(self) -> None: ...

    def stop_utterance(self) -> None: ...

    def snapshot(self) -> TranscriptSnapshot: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class _WhisperJob:
    session_id: int
    audio: bytes
    final: bool


class VoskVisualGateTranscriber:
    """Records microphone audio only between visual start/end events.

    Recognition runs on a background thread so webcam inference remains
    responsive. Audio never leaves the machine; Vosk uses the local model
    directory supplied by the caller.
    """

    def __init__(self, model_path: str | Path, sample_rate: int = 16_000) -> None:
        try:
            import sounddevice as sd
            from vosk import KaldiRecognizer, Model
        except ImportError as exc:
            raise RuntimeError(
                "STT needs the optional dependencies. Install with: "
                "python -m pip install -e '.[stt]'"
            ) from exc

        model_dir = Path(model_path)
        if not model_dir.is_dir():
            raise FileNotFoundError(
                f"Vosk model directory was not found: {model_dir}. "
                "Download a local Vosk model and pass it with --vosk-model."
            )

        self._sample_rate = sample_rate
        self._recognizer_type = KaldiRecognizer
        self._model = Model(str(model_dir))
        self._queue: queue.Queue[tuple[int, bytes | None]] = queue.Queue()
        self._lock = threading.Lock()
        self._session_id = 0
        self._recording = False
        self._recognizers: dict[int, object] = {}
        self._finalized: list[str] = []
        self._partial = ""
        self._closed = False
        self._worker = threading.Thread(target=self._consume_audio, name="vosk-stt", daemon=True)
        self._worker.start()
        self._stream = sd.RawInputStream(
            samplerate=sample_rate,
            blocksize=4_000,
            dtype="int16",
            channels=1,
            callback=self._on_audio,
        )
        self._stream.start()

    def start_utterance(self) -> None:
        with self._lock:
            if self._closed or self._recording:
                return
            self._session_id += 1
            self._recognizers[self._session_id] = self._recognizer_type(self._model, self._sample_rate)
            self._recording = True
            self._partial = ""

    def stop_utterance(self) -> None:
        with self._lock:
            if not self._recording:
                return
            self._recording = False
            self._queue.put((self._session_id, None))

    def snapshot(self) -> TranscriptSnapshot:
        with self._lock:
            return TranscriptSnapshot(tuple(self._finalized), self._partial, self._recording)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._recording = False
        self._stream.stop()
        self._stream.close()
        self._queue.put((-1, None))
        self._worker.join(timeout=2.0)

    def _on_audio(self, indata: bytes, frames: int, time_info: object, status: object) -> None:
        del frames, time_info, status
        with self._lock:
            if self._recording:
                self._queue.put((self._session_id, bytes(indata)))

    def _consume_audio(self) -> None:
        while True:
            session_id, data = self._queue.get()
            if session_id == -1:
                return
            with self._lock:
                recognizer = self._recognizers.get(session_id)
            if recognizer is None:
                continue
            if data is None:
                self._append_final(recognizer.FinalResult())
                with self._lock:
                    self._recognizers.pop(session_id, None)
                continue
            if recognizer.AcceptWaveform(data):
                self._append_final(recognizer.Result())
            else:
                partial = json.loads(recognizer.PartialResult()).get("partial", "").strip()
                with self._lock:
                    if session_id == self._session_id:
                        self._partial = partial

    def _append_final(self, payload: str) -> None:
        text = json.loads(payload).get("text", "").strip()
        with self._lock:
            if text:
                self._finalized.append(text)
            self._partial = ""


class WhisperVisualGateTranscriber:
    """Higher-accuracy, multilingual offline STT for noisy kiosk deployments.

    A lightweight streaming recognizer emits provisional words as microphone
    samples arrive. Whisper then performs the authoritative pass after the
    visual utterance ends. This split keeps the UI responsive on CPU while
    retaining Whisper's resilience to accents and noisy kiosk environments.
    """

    def __init__(
        self,
        model_size: str = "small",
        language: str | None = None,
        input_device: str | int | None = None,
        sample_rate: int = 16_000,
        device: str = "cpu",
        compute_type: str = "int8",
        initial_prompt: str | None = None,
        model_cache_dir: str | Path = "models/whisper",
        live_model_size: str | None = None,
        live_vosk_model_path: str | Path = "models/vosk-model-small-en-us-0.15",
        live_update_seconds: float = 0.25,
        live_window_seconds: float = 2.0,
        minimum_live_seconds: float = 0.20,
        final_beam_size: int = 5,
    ) -> None:
        try:
            import sounddevice as sd
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "Whisper STT needs the optional dependencies. Install with: "
                "python -m pip install -e '.[stt]'"
            ) from exc

        self._sample_rate = sample_rate
        self._language = language
        self._initial_prompt = initial_prompt
        if min(live_update_seconds, live_window_seconds, minimum_live_seconds) <= 0:
            raise ValueError("Live STT timings must be positive.")
        self._final_beam_size = final_beam_size
        self._live_update_seconds = live_update_seconds
        self._live_window_bytes = int(sample_rate * live_window_seconds * 2)
        self._minimum_live_bytes = int(sample_rate * minimum_live_seconds * 2)
        cache_dir = Path(model_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._final_model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root=str(cache_dir),
        )
        # Vosk is genuinely streaming and is used for the instant, editable
        # draft. Fall back to a small Whisper rolling window only if its local
        # model is unavailable.
        self._draft_recognizer_type = None
        self._draft_model = None
        try:
            from vosk import KaldiRecognizer, Model

            draft_model_dir = Path(live_vosk_model_path)
            if draft_model_dir.is_dir():
                self._draft_recognizer_type = KaldiRecognizer
                self._draft_model = Model(str(draft_model_dir))
        except ImportError:
            pass
        resolved_live_model = live_model_size or ("tiny.en" if language == "en" else "tiny")
        self._live_model = None
        if self._draft_model is None:
            self._live_model = (
                self._final_model
                if resolved_live_model == model_size
                else WhisperModel(
                    resolved_live_model,
                    device=device,
                    compute_type=compute_type,
                    download_root=str(cache_dir),
                )
            )
        self._jobs: deque[_WhisperJob] = deque()
        self._lock = threading.Lock()
        self._job_ready = threading.Condition(self._lock)
        self._session_id = 0
        self._active_session: int | None = None
        self._recording = False
        self._audio_buffers: dict[int, bytearray] = {}
        # Preserve two seconds before visual VAD fires. The streaming draft
        # receives only its recent 400 ms; the Whisper fallback can use all of
        # it when the streaming model is unavailable.
        self._preroll: deque[bytes] = deque(maxlen=40)
        self._last_live_update = 0.0
        self._finalized: list[str] = []
        self._partial = ""
        self._closed = False
        self._worker = threading.Thread(target=self._transcribe_utterances, name="whisper-stt", daemon=True)
        self._worker.start()
        self._draft_queue: queue.Queue[tuple[int, bytes | None]] = queue.Queue()
        self._draft_recognizers: dict[int, object] = {}
        self._draft_worker: threading.Thread | None = None
        if self._draft_model is not None:
            self._draft_worker = threading.Thread(
                target=self._consume_streaming_drafts, name="streaming-draft-stt", daemon=True
            )
            self._draft_worker.start()
        self._stream = sd.RawInputStream(
            samplerate=sample_rate,
            # 50 ms blocks let the live recognizer publish the first words as
            # they form instead of waiting for Whisper's old 250 ms chunks.
            blocksize=800,
            dtype="int16",
            channels=1,
            device=input_device,
            callback=self._on_audio,
        )
        self._stream.start()

    def start_utterance(self) -> None:
        with self._lock:
            if self._closed or self._recording:
                return
            self._session_id += 1
            buffer = bytearray()
            for chunk in self._preroll:
                buffer.extend(chunk)
            self._audio_buffers[self._session_id] = buffer
            if self._draft_recognizer_type is not None and self._draft_model is not None:
                self._draft_recognizers[self._session_id] = self._draft_recognizer_type(
                    self._draft_model, self._sample_rate
                )
                # Visual VAD takes a moment to declare speech. Feed only the
                # most recent 400 ms to recover the opening word, without
                # allowing old environmental audio to dominate the draft.
                for chunk in list(self._preroll)[-8:]:
                    self._draft_queue.put((self._session_id, chunk))
            self._active_session = self._session_id
            self._recording = True
            self._last_live_update = 0.0
            self._partial = "Listening..."

    def stop_utterance(self) -> None:
        with self._lock:
            if not self._recording:
                return
            session_id = self._session_id
            self._recording = False
            self._partial = "Finalizing transcript..."
        # This must occur after releasing the lock: a new speech start can now
        # create another session, but it cannot cancel this queued job.
        self._queue_final(session_id)

    def snapshot(self) -> TranscriptSnapshot:
        with self._lock:
            return TranscriptSnapshot(tuple(self._finalized), self._partial, self._recording)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._recording = False
            self._audio_buffers.clear()
            self._jobs.clear()
        self._stream.stop()
        self._stream.close()
        if self._draft_worker is not None:
            self._draft_queue.put((-1, None))
        with self._job_ready:
            self._jobs.append(_WhisperJob(-1, b"", False))
            self._job_ready.notify_all()
        self._worker.join(timeout=10.0)
        if self._draft_worker is not None:
            self._draft_worker.join(timeout=2.0)

    def _on_audio(self, indata: bytes, frames: int, time_info: object, status: object) -> None:
        del frames, time_info, status
        chunk = bytes(indata)
        with self._lock:
            self._preroll.append(chunk)
            if self._recording:
                self._audio_buffers[self._session_id].extend(chunk)
                if self._draft_model is not None:
                    self._draft_queue.put((self._session_id, chunk))
                    self._partial = "Transcribing live..."
                else:
                    now = time.monotonic()
                    if (
                        len(self._audio_buffers[self._session_id]) >= self._minimum_live_bytes
                        and now - self._last_live_update >= self._live_update_seconds
                    ):
                        audio = self._audio_buffers[self._session_id]
                        self._enqueue_live_revision(self._session_id, bytes(audio[-self._live_window_bytes :]))
                        self._last_live_update = now
                        self._partial = "Transcribing live..."

    def _enqueue_live_revision(self, session_id: int, audio: bytes) -> None:
        # Keep only the newest revision for this utterance when CPU inference
        # cannot keep up. Showing stale partials is worse than skipping one.
        self._jobs = deque(job for job in self._jobs if job.final or job.session_id != session_id)
        self._jobs.append(_WhisperJob(session_id, audio, False))
        self._job_ready.notify()

    def _queue_final(self, session_id: int) -> None:
        with self._job_ready:
            if self._closed or self._active_session != session_id:
                return
            audio = bytes(self._audio_buffers.pop(session_id, b""))
            self._active_session = None
            self._partial = "Finalizing transcript..."
            # Final decoding must come after the newest live revision so that
            # the UI remains responsive until the authoritative result arrives.
            self._jobs.append(_WhisperJob(session_id, audio, True))
            self._job_ready.notify()

    def _consume_streaming_drafts(self) -> None:
        """Publish Vosk's mutable partial result without blocking audio I/O."""
        while True:
            session_id, chunk = self._draft_queue.get()
            if session_id == -1:
                return
            with self._lock:
                recognizer = self._draft_recognizers.get(session_id)
            if recognizer is None or chunk is None:
                continue
            try:
                if recognizer.AcceptWaveform(chunk):
                    text = json.loads(recognizer.Result()).get("text", "").strip()
                else:
                    text = json.loads(recognizer.PartialResult()).get("partial", "").strip()
                with self._lock:
                    if session_id == self._active_session and self._recording:
                        self._partial = text or "Transcribing live..."
            except Exception:
                # The final Whisper pass remains available if a provisional
                # draft cannot be parsed on a particular audio device.
                continue

    def _transcribe_utterances(self) -> None:
        while True:
            with self._job_ready:
                while not self._jobs:
                    self._job_ready.wait()
                item = self._jobs.popleft()
            if item.session_id == -1:
                return
            try:
                transcript = self._decode(item.audio, self._final_beam_size if item.final else 1, item.final) if item.audio else ""
                with self._lock:
                    if item.final:
                        if transcript:
                            self._finalized.append(transcript)
                        elif item.session_id == self._session_id:
                            self._partial = "No confident speech recognized."
                        if item.session_id == self._session_id and transcript:
                            self._partial = ""
                    elif item.session_id == self._active_session and self._recording:
                        # This is a live, replaceable draft, not an appended
                        # immutable segment. It visibly corrects itself as the
                        # next rolling audio window supplies more context.
                        self._partial = transcript or "Transcribing live..."
            except Exception as exc:
                with self._lock:
                    if item.session_id == self._session_id:
                        self._partial = f"STT error: {exc}"

    def _decode(self, payload: bytes, beam_size: int, final: bool) -> str:
        audio = np.frombuffer(payload, dtype=np.int16).astype(np.float32) / 32768.0
        model = self._final_model if final else self._live_model
        if model is None:
            return ""
        segments, _ = model.transcribe(
            audio,
            language=self._language,
            beam_size=beam_size,
            # Visual VAD already selected the active user. Avoid a second VAD
            # suppressing short live drafts; retain it for the final pass where
            # it helps reject environmental noise.
            vad_filter=final,
            vad_parameters={"min_silence_duration_ms": 300},
            condition_on_previous_text=False,
            initial_prompt=self._initial_prompt,
        )
        accepted = []
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue
            if final and (segment.avg_logprob < -1.0 or segment.no_speech_prob > 0.60):
                continue
            accepted.append(text)
        return " ".join(accepted)


def merge_transcript_text(existing: str, incoming: str) -> str:
    """Append a chunk while removing the longest repeated token overlap."""
    old_words = existing.split()
    new_words = incoming.split()
    if not old_words:
        return " ".join(new_words)
    if not new_words:
        return " ".join(old_words)

    def canonical(word: str) -> str:
        return re.sub(r"[^\w']", "", word).casefold()

    old_key = [canonical(word) for word in old_words]
    new_key = [canonical(word) for word in new_words]
    maximum = min(len(old_words), len(new_words), 12)
    overlap = 0
    for size in range(maximum, 0, -1):
        if old_key[-size:] == new_key[:size]:
            overlap = size
            break
    return " ".join(old_words + new_words[overlap:])
