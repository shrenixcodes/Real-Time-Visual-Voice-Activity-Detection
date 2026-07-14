from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from statistics import pstdev
from typing import Optional

from .config import VADConfig


@dataclass(frozen=True)
class ActivityUpdate:
    speaking: bool
    raw_activity: bool
    signal: float
    event_type: Optional[str] = None
    event_latency_ms: float = 0.0


class VisualActivityDetector:
    """Mouth-motion VAD with time-based smoothing and hysteresis.

    The signal combines exponentially-smoothed per-frame mouth aspect ratio
    movement with short-window MAR variability. Decisions are duration based,
    not frame-count based, so behavior remains comparable across camera FPS.
    """

    def __init__(self, config: VADConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.speaking = False
        self._previous_mar: Optional[float] = None
        self._smoothed_motion = 0.0
        self._history: deque[tuple[float, float]] = deque()
        self._active_since: Optional[float] = None
        self._quiet_since: Optional[float] = None
        self._missing_since: Optional[float] = None

    def update(self, mar: Optional[float], timestamp: float) -> ActivityUpdate:
        if mar is None:
            return self._handle_missing_face(timestamp)

        self._missing_since = None
        if self._previous_mar is None:
            self._previous_mar = mar
            self._history.append((timestamp, mar))
            return ActivityUpdate(self.speaking, False, 0.0)

        movement = abs(mar - self._previous_mar)
        self._previous_mar = mar
        self._smoothed_motion = (
            self.config.ema_alpha * movement
            + (1 - self.config.ema_alpha) * self._smoothed_motion
        )
        self._history.append((timestamp, mar))
        self._discard_old_history(timestamp)
        variability = pstdev(value for _, value in self._history) if len(self._history) > 1 else 0.0
        motion_ratio = self._smoothed_motion / self.config.motion_threshold
        variability_ratio = variability / self.config.variance_threshold
        motion_active = self._smoothed_motion >= self.config.motion_threshold
        # A static closed mouth can still produce landmark jitter and a high
        # short-window variance. Variance is useful supporting evidence, but it
        # is never allowed to start speech without meaningful current movement.
        variability_active = (
            variability >= self.config.variance_threshold
            and self._smoothed_motion >= self.config.variance_motion_floor
        )
        signal = max(motion_ratio, variability_ratio if variability_active else 0.0)
        raw_activity = motion_active or variability_active
        return self._apply_debounce(raw_activity, signal, timestamp, self.config.end_hold_seconds)

    def _handle_missing_face(self, timestamp: float) -> ActivityUpdate:
        self._previous_mar = None
        self._history.clear()
        self._smoothed_motion = 0.0
        self._active_since = None
        if self._missing_since is None:
            self._missing_since = timestamp
        if self.speaking and timestamp - self._missing_since >= self.config.face_missing_end_seconds:
            self.speaking = False
            self._quiet_since = None
            return ActivityUpdate(False, False, 0.0, "speech_end", (timestamp - self._missing_since) * 1000)
        return ActivityUpdate(self.speaking, False, 0.0)

    def _apply_debounce(
        self, raw_activity: bool, signal: float, timestamp: float, end_hold: float
    ) -> ActivityUpdate:
        if raw_activity:
            self._quiet_since = None
            if self._active_since is None:
                self._active_since = timestamp
            if not self.speaking and timestamp - self._active_since >= self.config.start_hold_seconds:
                self.speaking = True
                return ActivityUpdate(True, True, signal, "speech_start", (timestamp - self._active_since) * 1000)
            return ActivityUpdate(self.speaking, True, signal)

        self._active_since = None
        if self._quiet_since is None:
            self._quiet_since = timestamp
        if self.speaking and timestamp - self._quiet_since >= end_hold:
            self.speaking = False
            return ActivityUpdate(False, False, signal, "speech_end", (timestamp - self._quiet_since) * 1000)
        return ActivityUpdate(self.speaking, False, signal)

    def _discard_old_history(self, timestamp: float) -> None:
        cutoff = timestamp - self.config.history_seconds
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()
