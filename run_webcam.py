"""Interactive webcam demo for the visual VAD module.

Run: python run_webcam.py --camera 0
Press q or Escape to close the preview.
"""

from __future__ import annotations

import argparse
import json
import time

import cv2

from visual_vad import VVADEvent, VisualVAD
from visual_vad.mediapipe_backend import MediaPipeFaceMeshDetector


def print_event(event: VVADEvent) -> None:
    print(json.dumps(event.as_dict()), flush=True)


def draw_overlay(frame, result) -> None:
    primary_index = result.primary.face.detector_index if result.primary else None
    for face in result.faces:
        box = face.bbox
        is_primary = face.detector_index == primary_index
        color = (0, 220, 0) if is_primary else (110, 110, 110)
        cv2.rectangle(frame, (int(box.x), int(box.y)), (int(box.x + box.width), int(box.y + box.height)), color, 2)
        label = "PRIMARY" if is_primary else "ignored"
        cv2.putText(frame, label, (int(box.x), max(20, int(box.y) - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    state = "SPEAKING" if result.speaking else "quiet"
    track = result.primary.track_id if result.primary else "-"
    text = f"{state} | primary={track} | signal={result.signal:.2f} | {result.processing_ms:.1f} ms"
    cv2.rectangle(frame, (0, 0), (min(frame.shape[1], 620), 34), (0, 0, 0), -1)
    cv2.putText(frame, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time visual voice activity detection demo")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--max-faces", type=int, default=5)
    args = parser.parse_args()

    capture = cv2.VideoCapture(args.camera)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}.")

    detector = MediaPipeFaceMeshDetector(max_faces=args.max_faces)
    vad = VisualVAD(detector=detector, on_event=print_event)
    started = time.monotonic()
    frames = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result = vad.process_frame(frame)
            frames += 1
            draw_overlay(frame, result)
            elapsed = max(time.monotonic() - started, 1e-6)
            cv2.putText(frame, f"FPS: {frames / elapsed:.1f}", (10, frame.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.imshow("Visual VAD - q to quit", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        detector.close()
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

