from __future__ import annotations

import math

import numpy as np

try:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except ImportError as exc:
    raise RuntimeError(
        "The webcam backend needs opencv-python and mediapipe. "
        "Install requirements.txt first."
    ) from exc

from .models import BoundingBox, FaceObservation


class MediaPipeFaceMeshDetector:
    """FaceMesh adapter that returns all visible faces and each mouth MAR."""

    # MediaPipe's stable Face Mesh landmark indices for outer mouth corners and lips.
    LEFT_MOUTH = 61
    RIGHT_MOUTH = 291
    UPPER_LIP = 13
    LOWER_LIP = 14

    def __init__(self, max_faces: int = 5) -> None:
        self._cv2 = cv2
        # Create FaceLandmarker options with IMAGE mode
        base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=max_faces,
            min_face_detection_confidence=0.55,
            min_face_presence_confidence=0.55,
            min_tracking_confidence=0.55,
        )
        self._detector = vision.FaceLandmarker.create_from_options(options)

    def detect(self, frame: np.ndarray) -> list[FaceObservation]:
        height, width = frame.shape[:2]
        rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detection_result = self._detector.detect(mp_image)
        observations: list[FaceObservation] = []
        for index, face_landmarks in enumerate(detection_result.face_landmarks or []):
            xs = [landmark.x * width for landmark in face_landmarks]
            ys = [landmark.y * height for landmark in face_landmarks]
            left, right = max(0.0, min(xs)), min(float(width), max(xs))
            top, bottom = max(0.0, min(ys)), min(float(height), max(ys))
            mouth_width = self._distance(face_landmarks[self.LEFT_MOUTH], face_landmarks[self.RIGHT_MOUTH], width, height)
            mouth_open = self._distance(face_landmarks[self.UPPER_LIP], face_landmarks[self.LOWER_LIP], width, height)
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
        self._detector.close()

    @staticmethod
    def _distance(a: object, b: object, width: int, height: int) -> float:
        return math.hypot((a.x - b.x) * width, (a.y - b.y) * height)  # type: ignore[attr-defined]
