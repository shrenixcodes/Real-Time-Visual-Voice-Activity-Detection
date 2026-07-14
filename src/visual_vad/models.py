from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal, Optional


@dataclass(frozen=True)
class BoundingBox:
    """Pixel coordinate face box represented by top-left, width, and height."""

    x: float
    y: float
    width: float
    height: float

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)

    def iou(self, other: "BoundingBox") -> float:
        left = max(self.x, other.x)
        top = max(self.y, other.y)
        right = min(self.x + self.width, other.x + other.width)
        bottom = min(self.y + self.height, other.y + other.height)
        overlap = max(0.0, right - left) * max(0.0, bottom - top)
        union = self.area + other.area - overlap
        return overlap / union if union > 0 else 0.0


@dataclass(frozen=True)
class FaceObservation:
    """The small set of per-face measurements needed by the core pipeline."""

    bbox: BoundingBox
    mouth_aspect_ratio: float
    detector_index: int = 0


@dataclass(frozen=True)
class SelectedFace:
    face: FaceObservation
    track_id: int
    score: float


@dataclass(frozen=True)
class VVADEvent:
    event_type: Literal["speech_start", "speech_end"]
    timestamp: float
    primary_track_id: int
    latency_ms: float
    signal: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FrameResult:
    timestamp: float
    faces: list[FaceObservation]
    primary: Optional[SelectedFace]
    speaking: bool
    raw_activity: bool
    signal: float
    events: list[VVADEvent] = field(default_factory=list)
    processing_ms: float = 0.0

