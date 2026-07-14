from __future__ import annotations

import math

import numpy as np

from .models import BoundingBox, FaceObservation


class MediaPipeFaceMeshDetector:
    """FaceMesh adapter that returns all visible faces and each mouth MAR."""

    # MediaPipe's stable Face Mesh landmark indices for outer mouth corners and lips.
    LEFT_MOUTH = 61
    RIGHT_MOUTH = 291
    UPPER_LIP = 13
    LOWER_LIP = 14

    def __init__(self, max_faces: int = 5) -> None:
        try:
            import cv2
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "The webcam backend needs opencv-python and mediapipe. "
                "Install requirements.txt first."
            ) from exc
        self._cv2 = cv2
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=False,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.55,
        )

    def detect(self, frame: np.ndarray) -> list[FaceObservation]:
        height, width = frame.shape[:2]
        rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        result = self._mesh.process(rgb)
        observations: list[FaceObservation] = []
        for index, face in enumerate(result.multi_face_landmarks or []):
            points = face.landmark
            xs = [point.x * width for point in points]
            ys = [point.y * height for point in points]
            left, right = max(0.0, min(xs)), min(float(width), max(xs))
            top, bottom = max(0.0, min(ys)), min(float(height), max(ys))
            mouth_width = self._distance(points[self.LEFT_MOUTH], points[self.RIGHT_MOUTH], width, height)
            mouth_open = self._distance(points[self.UPPER_LIP], points[self.LOWER_LIP], width, height)
            if mouth_width <= 1e-6:
                continue
            observations.append(
                FaceObservation(
                    bbox=BoundingBox(left, top, right - left, bottom - top),
                    mouth_aspect_ratio=mouth_open / mouth_width,
                    detector_index=index,
                )
            )
        return observations

    def close(self) -> None:
        self._mesh.close()

    @staticmethod
    def _distance(a: object, b: object, width: int, height: int) -> float:
        return math.hypot((a.x - b.x) * width, (a.y - b.y) * height)  # type: ignore[attr-defined]
