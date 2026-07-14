from __future__ import annotations

import time
from collections.abc import Callable
from typing import Optional, Protocol

import numpy as np

from .activity import VisualActivityDetector
from .config import VisualVADConfig
from .models import FaceObservation, FrameResult, VVADEvent
from .selection import PrimaryUserSelector


class FaceDetector(Protocol):
    def detect(self, frame: np.ndarray) -> list[FaceObservation]: ...


EventCallback = Callable[[VVADEvent], None]


class VisualVAD:
    """Reusable V-VAD module with a compact callback integration contract.

    Use :meth:`process_frame` with a detector backend, or
    :meth:`process_observations` when another component already supplies faces
    and mouth landmarks. The callback is invoked synchronously for each
    debounced event, preserving event order for downstream kiosk logic.
    """

    def __init__(
        self,
        detector: Optional[FaceDetector] = None,
        on_event: Optional[EventCallback] = None,
        config: Optional[VisualVADConfig] = None,
    ) -> None:
        self.detector = detector
        self.on_event = on_event
        self.config = config or VisualVADConfig()
        self._selector = PrimaryUserSelector(self.config.primary)
        self._activity = VisualActivityDetector(self.config.vad)
        self._last_selected_track_id: Optional[int] = None

    def process_frame(self, frame: np.ndarray, timestamp: Optional[float] = None) -> FrameResult:
        """Process one BGR webcam frame using the configured detector backend."""
        if self.detector is None:
            raise RuntimeError("No face detector was configured. Use process_observations instead.")
        started = time.perf_counter()
        faces = self.detector.detect(frame)
        result = self.process_observations(faces, (frame.shape[1], frame.shape[0]), timestamp)
        return FrameResult(
            timestamp=result.timestamp,
            faces=result.faces,
            primary=result.primary,
            speaking=result.speaking,
            raw_activity=result.raw_activity,
            signal=result.signal,
            events=result.events,
            processing_ms=(time.perf_counter() - started) * 1000,
        )

    def process_observations(
        self,
        faces: list[FaceObservation],
        frame_size: tuple[int, int],
        timestamp: Optional[float] = None,
    ) -> FrameResult:
        """Process detector-neutral face observations; useful for tests and integration."""
        now = time.monotonic() if timestamp is None else timestamp
        primary = self._selector.update(faces, frame_size, now)
        if primary is not None and primary.track_id != self._last_selected_track_id:
            self._activity.reset()
            self._last_selected_track_id = primary.track_id

        update = self._activity.update(primary.face.mouth_aspect_ratio if primary else None, now)
        events: list[VVADEvent] = []
        if update.event_type is not None:
            track_id = primary.track_id if primary else self._selector.current_track_id
            if track_id is not None:
                event = VVADEvent(
                    event_type=update.event_type,  # type: ignore[arg-type]
                    timestamp=now,
                    primary_track_id=track_id,
                    latency_ms=update.event_latency_ms,
                    signal=update.signal,
                )
                events.append(event)
                if self.on_event is not None:
                    self.on_event(event)

        return FrameResult(
            timestamp=now,
            faces=faces,
            primary=primary,
            speaking=update.speaking,
            raw_activity=update.raw_activity,
            signal=update.signal,
            events=events,
        )

