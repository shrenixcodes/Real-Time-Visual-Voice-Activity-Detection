"""Local web dashboard that joins camera, visual VAD, events, and STT."""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import cv2
import numpy as np

from .mediapipe_backend import MediaPipeFaceMeshDetector
from .models import FrameResult, VVADEvent
from .pipeline import VisualVAD
from .transcription import VisualGateTranscriber, VoskVisualGateTranscriber, WhisperVisualGateTranscriber


@dataclass(frozen=True)
class DashboardConfig:
    camera: int = 0
    width: int = 1280
    height: int = 720
    max_faces: int = 5
    enable_stt: bool = True
    stt_backend: str = "whisper"
    whisper_model: str = "small"
    whisper_cache: Path = Path("models/whisper")
    language: Optional[str] = "en"
    audio_device: str | int | None = None
    initial_prompt: Optional[str] = None
    live_update_seconds: float = 0.85
    live_window_seconds: float = 5.0


class DashboardState:
    """Thread-safe state shared by the camera worker and dashboard requests."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._jpeg: Optional[bytes] = None
        self._frame_version = 0
        self._status = "Starting camera..."
        self._error: Optional[str] = None
        self._fps = 0.0
        self._processing_ms = 0.0
        self._signal = 0.0
        self._speaking = False
        self._primary_track: Optional[int] = None
        self._events: deque[dict[str, object]] = deque(maxlen=50)
        self._transcript: deque[dict[str, str]] = deque(maxlen=40)
        self._partial = ""
        self._seen_finalized = 0

    def set_error(self, message: str) -> None:
        with self._condition:
            self._error = message
            self._status = "Camera unavailable"
            self._condition.notify_all()

    def update_frame(self, frame: np.ndarray, result: FrameResult, fps: float) -> None:
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 84])
        if not ok:
            return
        with self._condition:
            self._jpeg = encoded.tobytes()
            self._frame_version += 1
            self._fps = fps
            self._processing_ms = result.processing_ms
            self._signal = result.signal
            self._speaking = result.speaking
            self._primary_track = result.primary.track_id if result.primary else None
            self._status = "Speaking" if result.speaking else "Monitoring"
            self._error = None
            self._condition.notify_all()

    def add_event(self, event: VVADEvent) -> None:
        record = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "type": event.event_type,
            "track": event.primary_track_id,
            "latency": round(event.latency_ms),
        }
        with self._condition:
            self._events.appendleft(record)
            self._condition.notify_all()

    def update_transcript(self, finalized: tuple[str, ...], partial: str) -> None:
        with self._condition:
            if len(finalized) < self._seen_finalized:
                self._seen_finalized = 0
            for text in finalized[self._seen_finalized :]:
                self._transcript.append({"time": datetime.now().strftime("%H:%M:%S"), "text": text})
            self._seen_finalized = len(finalized)
            self._partial = partial

    def snapshot(self) -> dict[str, object]:
        with self._condition:
            return {
                "status": self._status,
                "error": self._error,
                "fps": round(self._fps, 1),
                "processing_ms": round(self._processing_ms, 1),
                "signal": round(self._signal, 2),
                "speaking": self._speaking,
                "primary_track": self._primary_track,
                "events": list(self._events),
                "transcript": list(self._transcript),
                "partial": self._partial,
            }

    def wait_for_frame(self, version: int, timeout: float = 2.0) -> tuple[Optional[bytes], int]:
        with self._condition:
            if self._frame_version == version:
                self._condition.wait(timeout)
            return self._jpeg, self._frame_version


class DashboardRuntime:
    """Background webcam/STT worker backing the local dashboard."""

    def __init__(self, config: DashboardConfig, state: DashboardState) -> None:
        self.config = config
        self.state = state
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="visual-vad-camera", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        capture: Optional[cv2.VideoCapture] = None
        detector: Optional[MediaPipeFaceMeshDetector] = None
        transcriber: Optional[VisualGateTranscriber] = None
        try:
            capture = cv2.VideoCapture(self.config.camera)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            if not capture.isOpened():
                raise RuntimeError(f"Could not open camera {self.config.camera}.")

            detector = MediaPipeFaceMeshDetector(max_faces=self.config.max_faces)
            if self.config.enable_stt:
                if self.config.stt_backend == "whisper":
                    transcriber = WhisperVisualGateTranscriber(
                        model_size=self.config.whisper_model,
                        language=self.config.language,
                        input_device=self.config.audio_device,
                        model_cache_dir=self.config.whisper_cache,
                        initial_prompt=self.config.initial_prompt,
                        live_update_seconds=self.config.live_update_seconds,
                        live_window_seconds=self.config.live_window_seconds,
                    )
                else:
                    transcriber = VoskVisualGateTranscriber(Path("models/vosk-model-small-en-us-0.15"))

            def on_event(event: VVADEvent) -> None:
                self.state.add_event(event)
                if transcriber is None:
                    return
                if event.event_type == "speech_start":
                    transcriber.start_utterance()
                else:
                    transcriber.stop_utterance()

            vad = VisualVAD(detector=detector, on_event=on_event)
            frames = 0
            window_started = time.monotonic()
            while not self._stop.is_set():
                ok, frame = capture.read()
                if not ok:
                    self.state.set_error("Camera frame could not be read.")
                    break
                result = vad.process_frame(frame)
                frames += 1
                elapsed = max(time.monotonic() - window_started, 1e-6)
                fps = frames / elapsed
                self._draw_vision_overlay(frame, result, fps)
                self.state.update_frame(frame, result, fps)
                if transcriber is not None:
                    snapshot = transcriber.snapshot()
                    self.state.update_transcript(snapshot.finalized, snapshot.partial)
        except Exception as exc:
            self.state.set_error(str(exc))
        finally:
            if transcriber is not None:
                transcriber.close()
            if detector is not None:
                detector.close()
            if capture is not None:
                capture.release()

    @staticmethod
    def _draw_vision_overlay(frame: np.ndarray, result: FrameResult, fps: float) -> None:
        primary_index = result.primary.face.detector_index if result.primary else None
        for face in result.faces:
            box = face.bbox
            is_primary = face.detector_index == primary_index
            color = (82, 213, 255) if is_primary else (150, 150, 150)
            cv2.rectangle(frame, (int(box.x), int(box.y)), (int(box.x + box.width), int(box.y + box.height)), color, 2)
            label = "PRIMARY" if is_primary else "background"
            cv2.putText(frame, label, (int(box.x), max(20, int(box.y) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        status = "SPEAKING" if result.speaking else "MONITORING"
        cv2.rectangle(frame, (0, 0), (min(frame.shape[1], 620), 37), (15, 24, 26), -1)
        text = f"{status}  |  {fps:.1f} FPS  |  signal {result.signal:.2f}  |  {result.processing_ms:.1f} ms"
        cv2.putText(frame, text, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (243, 244, 236), 2)


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], state: DashboardState, runtime: DashboardRuntime) -> None:
        super().__init__(address, DashboardRequestHandler)
        self.state = state
        self.runtime = runtime


class DashboardRequestHandler(BaseHTTPRequestHandler):
    server: DashboardServer

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/":
            html = Path(__file__).with_name("dashboard.html").read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
            return
        if path == "/api/state":
            payload = json.dumps(self.server.state.snapshot()).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if path == "/stream.mjpg":
            self._stream_camera()
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def _stream_camera(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        version = -1
        try:
            while not self.server.runtime._stop.is_set():
                jpeg, version = self.server.state.wait_for_frame(version)
                if jpeg is None:
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            return

    def log_message(self, format: str, *args: object) -> None:
        return


def create_dashboard_server(host: str, port: int, config: DashboardConfig) -> DashboardServer:
    state = DashboardState()
    runtime = DashboardRuntime(config, state)
    runtime.start()
    return DashboardServer((host, port), state, runtime)
