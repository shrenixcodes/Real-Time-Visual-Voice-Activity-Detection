"""Optional, local speech-to-text gated by visual VAD events.

The core V-VAD module never opens a microphone. This adapter is only used by
the webcam demo when ``--stt`` is selected and requires a local Vosk model.
"""

from __future__ import annotations

import json
import queue
import re
import threading
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

    Short audio chunks are decoded while the primary user is still speaking,
    which keeps the dashboard responsive. The final trailing chunk completes
    the utterance after speech end. Overlap-aware text merging prevents the
    context overlap from repeating words in the transcript.
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
        partial_chunk_seconds: float = 2.2,
        overlap_seconds: float = 0.35,
        beam_size: int = 5,
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
        self._beam_size = beam_size
        if partial_chunk_seconds <= 0 or overlap_seconds < 0:
            raise ValueError("STT chunk durations must be non-negative and the chunk duration must be positive.")
        cache_dir = Path(model_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root=str(cache_dir),
        )
        self._queue: queue.Queue[tuple[int, bytes, bool] | None] = queue.Queue()
        self._lock = threading.Lock()
        self._session_id = 0
        self._recording = False
        self._audio_buffers: dict[int, bytearray] = {}
        self._queued_to: dict[int, int] = {}
        self._session_text: dict[int, str] = {}
        self._preroll: deque[bytes] = deque(maxlen=2)  # 500 ms at the 4,000-sample block size.
        self._chunk_bytes = int(sample_rate * partial_chunk_seconds * 2)
        self._overlap_bytes = int(sample_rate * overlap_seconds * 2)
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
            self._session_id += 1
            buffer = bytearray()
            for chunk in self._preroll:
                buffer.extend(chunk)
            self._audio_buffers[self._session_id] = buffer
            self._queued_to[self._session_id] = 0
            self._session_text[self._session_id] = ""
            self._recording = True
            self._partial = "Listening..."

    def stop_utterance(self) -> None:
        with self._lock:
            if not self._recording:
                return
            session_id = self._session_id
            self._recording = False
            self._partial = "Transcribing..."
            self._enqueue_chunk(session_id, final=True)

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
            self._queued_to.clear()
        self._stream.stop()
        self._stream.close()
        self._queue.put(None)
        self._worker.join(timeout=10.0)

    def _on_audio(self, indata: bytes, frames: int, time_info: object, status: object) -> None:
        del frames, time_info, status
        chunk = bytes(indata)
        with self._lock:
            self._preroll.append(chunk)
            if self._recording:
                self._audio_buffers[self._session_id].extend(chunk)
                self._enqueue_chunk(self._session_id, final=False)

    def _enqueue_chunk(self, session_id: int, final: bool) -> None:
        """Queue a bounded chunk while holding ``_lock``.

        Each live chunk has a small look-back window so words crossing a chunk
        boundary retain context. Work is only performed in the decoder thread.
        """
        buffer = self._audio_buffers.get(session_id)
        if buffer is None:
            return
        end = len(buffer)
        queued_to = self._queued_to.get(session_id, 0)
        if not final and end - queued_to < self._chunk_bytes:
            return
        if end > queued_to:
            start = max(0, queued_to - self._overlap_bytes)
            self._queue.put((session_id, bytes(buffer[start:end]), final))
            self._queued_to[session_id] = end
        elif final:
            # A previous live chunk already covers the complete utterance. A
            # marker preserves FIFO ordering and finalizes after that chunk.
            self._queue.put((session_id, b"", True))
        if final:
            self._audio_buffers.pop(session_id, None)
            self._queued_to.pop(session_id, None)

    def _transcribe_utterances(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            session_id, payload, final = item
            try:
                transcript = self._decode(payload) if payload else ""
                with self._lock:
                    if transcript:
                        self._session_text[session_id] = merge_transcript_text(
                            self._session_text.get(session_id, ""), transcript
                        )
                    merged = self._session_text.get(session_id, "")
                    if final:
                        if merged:
                            self._finalized.append(merged)
                            if session_id == self._session_id:
                                self._partial = ""
                        elif session_id == self._session_id:
                            self._partial = "No confident speech recognized."
                        self._session_text.pop(session_id, None)
                    elif session_id == self._session_id:
                        self._partial = merged or "Listening..."
            except Exception as exc:
                with self._lock:
                    if session_id == self._session_id:
                        self._partial = f"STT error: {exc}"

    def _decode(self, payload: bytes) -> str:
        audio = np.frombuffer(payload, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = self._model.transcribe(
            audio,
            language=self._language,
            beam_size=self._beam_size,
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
