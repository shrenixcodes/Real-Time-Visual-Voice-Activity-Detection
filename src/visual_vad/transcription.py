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

    A rolling window is decoded while the user is speaking. The resulting
    draft is intentionally revisable: Whisper can correct a word as more
    context arrives. After a short visual-VAD gap, a final high-confidence pass
    over the complete utterance is committed to the transcript.
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
        live_update_seconds: float = 0.85,
        live_window_seconds: float = 5.0,
        minimum_live_seconds: float = 0.80,
        speech_gap_seconds: float = 1.10,
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
        if min(live_update_seconds, live_window_seconds, minimum_live_seconds, speech_gap_seconds) <= 0:
            raise ValueError("Live STT timings must be positive.")
        self._final_beam_size = final_beam_size
        self._live_update_seconds = live_update_seconds
        self._live_window_bytes = int(sample_rate * live_window_seconds * 2)
        self._minimum_live_bytes = int(sample_rate * minimum_live_seconds * 2)
        cache_dir = Path(model_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root=str(cache_dir),
        )
        self._jobs: deque[_WhisperJob] = deque()
        self._lock = threading.Lock()
        self._job_ready = threading.Condition(self._lock)
        self._session_id = 0
        self._active_session: int | None = None
        self._recording = False
        self._audio_buffers: dict[int, bytearray] = {}
        self._preroll: deque[bytes] = deque(maxlen=2)  # 500 ms at the 4,000-sample block size.
        self._last_live_update = 0.0
        self._finish_timer: threading.Timer | None = None
        self._finalized: list[str] = []
        self._partial = ""
        self._closed = False
        self._worker = threading.Thread(target=self._transcribe_utterances, name="whisper-stt", daemon=True)
        self._worker.start()
        self._stream = sd.RawInputStream(
            samplerate=sample_rate,
            blocksize=4_000,
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
            if self._finish_timer is not None and self._active_session is not None:
                # Visual VAD briefly toggled quiet, then detected speech again.
                # Keep audio and context in one continuous utterance.
                self._finish_timer.cancel()
                self._finish_timer = None
                self._recording = True
                self._partial = self._partial or "Listening..."
                return
            self._session_id += 1
            buffer = bytearray()
            for chunk in self._preroll:
                buffer.extend(chunk)
            self._audio_buffers[self._session_id] = buffer
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
            self._partial = "Listening for a final phrase..."
            if self._finish_timer is not None:
                self._finish_timer.cancel()
            self._finish_timer = threading.Timer(self._speech_gap_seconds, self._queue_final, args=(session_id,))
            self._finish_timer.daemon = True
            self._finish_timer.start()

    def snapshot(self) -> TranscriptSnapshot:
        with self._lock:
            return TranscriptSnapshot(tuple(self._finalized), self._partial, self._recording)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._recording = False
            if self._finish_timer is not None:
                self._finish_timer.cancel()
                self._finish_timer = None
            self._audio_buffers.clear()
            self._jobs.clear()
        self._stream.stop()
        self._stream.close()
        with self._job_ready:
            self._jobs.append(_WhisperJob(-1, b"", False))
            self._job_ready.notify_all()
        self._worker.join(timeout=10.0)

    def _on_audio(self, indata: bytes, frames: int, time_info: object, status: object) -> None:
        del frames, time_info, status
        chunk = bytes(indata)
        with self._lock:
            self._preroll.append(chunk)
            if self._recording:
                self._audio_buffers[self._session_id].extend(chunk)
                now = time.monotonic()
                if (
                    len(self._audio_buffers[self._session_id]) >= self._minimum_live_bytes
                    and now - self._last_live_update >= self._live_update_seconds
                ):
                    audio = self._audio_buffers[self._session_id]
                    self._enqueue_live_revision(self._session_id, bytes(audio[-self._live_window_bytes :]))
                    self._last_live_update = now

    def _enqueue_live_revision(self, session_id: int, audio: bytes) -> None:
        # Keep only the newest revision for this utterance when CPU inference
        # cannot keep up. Showing stale partials is worse than skipping one.
        self._jobs = deque(job for job in self._jobs if job.final or job.session_id != session_id)
        self._jobs.append(_WhisperJob(session_id, audio, False))
        self._job_ready.notify()

    def _queue_final(self, session_id: int) -> None:
        with self._job_ready:
            if self._closed or self._recording or self._active_session != session_id:
                return
            audio = bytes(self._audio_buffers.pop(session_id, b""))
            self._active_session = None
            self._finish_timer = None
            self._partial = "Finalizing transcript..."
            # Final decoding must come after the newest live revision so that
            # the UI remains responsive until the authoritative result arrives.
            self._jobs.append(_WhisperJob(session_id, audio, True))
            self._job_ready.notify()

    def _transcribe_utterances(self) -> None:
        while True:
            with self._job_ready:
                while not self._jobs:
                    self._job_ready.wait()
                item = self._jobs.popleft()
            if item.session_id == -1:
                return
            try:
                transcript = self._decode(item.audio, self._final_beam_size if item.final else 1) if item.audio else ""
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
                        self._partial = transcript or "Listening..."
            except Exception as exc:
                with self._lock:
                    if item.session_id == self._session_id:
                        self._partial = f"STT error: {exc}"

    def _decode(self, payload: bytes, beam_size: int) -> str:
        audio = np.frombuffer(payload, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = self._model.transcribe(
            audio,
            language=self._language,
            beam_size=beam_size,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
            condition_on_previous_text=False,
            initial_prompt=self._initial_prompt,
        )
        accepted = [
            segment.text.strip()
            for segment in segments
            if segment.text.strip()
            and segment.avg_logprob >= -1.0
            and segment.no_speech_prob <= 0.60
        ]
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
