"""Optional, local speech-to-text gated by visual VAD events.

The core V-VAD module never opens a microphone. This adapter is only used by
the webcam demo when ``--stt`` is selected and requires a local Vosk model.
"""

from __future__ import annotations

import json
import queue
import threading
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TranscriptSnapshot:
    finalized: tuple[str, ...]
    partial: str
    listening: bool


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
