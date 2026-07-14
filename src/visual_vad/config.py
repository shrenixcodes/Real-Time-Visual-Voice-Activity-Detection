from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PrimarySelectorConfig:
    """Controls which face is treated as the kiosk user.

    Area is a lightweight distance proxy. Centrality is the inverse normalized
    distance from the image centre. Once chosen, a user is kept by IoU matching
    until they have been absent for ``missing_timeout_seconds``.
    """

    area_weight: float = 0.60
    centrality_weight: float = 0.40
    min_iou_match: float = 0.18
    missing_timeout_seconds: float = 0.75

    def __post_init__(self) -> None:
        if self.area_weight < 0 or self.centrality_weight < 0:
            raise ValueError("Primary-user weights must be non-negative.")
        if self.area_weight + self.centrality_weight == 0:
            raise ValueError("At least one primary-user weight must be positive.")
        if not 0 <= self.min_iou_match <= 1:
            raise ValueError("min_iou_match must be between 0 and 1.")
        if self.missing_timeout_seconds <= 0:
            raise ValueError("missing_timeout_seconds must be positive.")


@dataclass(frozen=True)
class VADConfig:
    """Tunable thresholds for the visual speech signal and its debounce."""

    ema_alpha: float = 0.35
    history_seconds: float = 0.65
    motion_threshold: float = 0.010
    variance_threshold: float = 0.0035
    start_hold_seconds: float = 0.16
    end_hold_seconds: float = 0.55
    face_missing_end_seconds: float = 0.55

    def __post_init__(self) -> None:
        if not 0 < self.ema_alpha <= 1:
            raise ValueError("ema_alpha must be in (0, 1].")
        for name in (
            "history_seconds",
            "motion_threshold",
            "variance_threshold",
            "start_hold_seconds",
            "end_hold_seconds",
            "face_missing_end_seconds",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")


@dataclass(frozen=True)
class VisualVADConfig:
    primary: PrimarySelectorConfig = field(default_factory=PrimarySelectorConfig)
    vad: VADConfig = field(default_factory=VADConfig)

