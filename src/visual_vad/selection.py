from __future__ import annotations

import math
from typing import Optional

from .config import PrimarySelectorConfig
from .models import FaceObservation, SelectedFace


class PrimaryUserSelector:
    """Stable closest/most-central primary-user selection.

    A current user always wins if a detected face overlaps their previous box.
    This deliberately resists switching to a moving or talking bystander.
    """

    def __init__(self, config: PrimarySelectorConfig) -> None:
        self.config = config
        self._current: Optional[SelectedFace] = None
        self._missing_since: Optional[float] = None
        self._next_track_id = 1

    @property
    def current_track_id(self) -> Optional[int]:
        return self._current.track_id if self._current else None

    def update(
        self,
        faces: list[FaceObservation],
        frame_size: tuple[int, int],
        timestamp: float,
    ) -> Optional[SelectedFace]:
        """Return the selected primary face, or ``None`` during a hand-off gap."""
        if self._current is not None:
            best_match = self._best_overlap(faces, self._current.face)
            if best_match is not None and self._current.face.bbox.iou(best_match.bbox) >= self.config.min_iou_match:
                self._missing_since = None
                self._current = SelectedFace(
                    face=best_match,
                    track_id=self._current.track_id,
                    score=self._score(best_match, frame_size),
                )
                return self._current

            if self._missing_since is None:
                self._missing_since = timestamp
            if timestamp - self._missing_since < self.config.missing_timeout_seconds:
                return None
            self._current = None
            self._missing_since = None

        if not faces:
            return None

        primary = max(faces, key=lambda face: self._score(face, frame_size))
        self._current = SelectedFace(
            face=primary,
            track_id=self._next_track_id,
            score=self._score(primary, frame_size),
        )
        self._next_track_id += 1
        return self._current

    def _best_overlap(
        self, faces: list[FaceObservation], previous: FaceObservation
    ) -> Optional[FaceObservation]:
        return max(faces, key=lambda face: previous.bbox.iou(face.bbox), default=None)

    def _score(self, face: FaceObservation, frame_size: tuple[int, int]) -> float:
        frame_width, frame_height = frame_size
        frame_area = max(1.0, float(frame_width * frame_height))
        area_score = min(1.0, face.bbox.area / (frame_area * 0.28))

        cx, cy = face.bbox.center
        dx = (cx - frame_width / 2) / max(1.0, frame_width / 2)
        dy = (cy - frame_height / 2) / max(1.0, frame_height / 2)
        centrality = max(0.0, 1.0 - min(1.0, math.hypot(dx, dy) / math.sqrt(2)))
        return self.config.area_weight * area_score + self.config.centrality_weight * centrality

