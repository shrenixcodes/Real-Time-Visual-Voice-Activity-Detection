"""Interactive webcam demo for the visual VAD module.

Run: python run_webcam.py --camera 0
Press q or Escape to close the preview.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from visual_vad import VVADEvent, VisualVAD
from visual_vad.mediapipe_backend import MediaPipeFaceMeshDetector
from visual_vad.transcription import (
    TranscriptSnapshot,
    VisualGateTranscriber,
    VoskVisualGateTranscriber,
    WhisperVisualGateTranscriber,
)


def parse_audio_device(value: str) -> str | int:
    """Accept either a numeric sounddevice ID or a descriptive device name."""
    return int(value) if value.isdigit() else value


def print_event(event: VVADEvent) -> None:
    print(json.dumps(event.as_dict()), flush=True)


def handle_event(event: VVADEvent, transcriber: VisualGateTranscriber | None) -> None:
    print_event(event)
    if transcriber is None:
        return
    if event.event_type == "speech_start":
        transcriber.start_utterance()
    else:
        transcriber.stop_utterance()


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


def draw_transcript_window(snapshot: TranscriptSnapshot, backend: str) -> None:
    canvas = np.full((250, 820, 3), 18, dtype=np.uint8)
    status = "LISTENING" if snapshot.listening else "waiting for visual speech_start"
    color = (60, 220, 80) if snapshot.listening else (185, 185, 185)
    cv2.putText(canvas, f"{backend.upper()} STT: {status}", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2)
    y = 62
    for text in snapshot.finalized[-3:]:
        for line in wrap_text(text):
            cv2.putText(canvas, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240, 240, 240), 1)
            y += 27
    if snapshot.partial:
        cv2.putText(canvas, f"> {snapshot.partial[:88]}", (16, 224), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (110, 190, 255), 1)
    cv2.imshow("Visual VAD Transcript (STT)", canvas)


def wrap_text(text: str, width: int = 88) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    return lines + ([current] if current else [])


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time visual voice activity detection demo")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--max-faces", type=int, default=5)
    parser.add_argument("--stt", action="store_true", help="Enable local, Vosk-based microphone transcription")
    parser.add_argument(
        "--stt-backend",
        choices=("whisper", "vosk"),
        default="whisper",
        help="Whisper is the higher-accuracy default; Vosk is a lightweight fallback",
    )
    parser.add_argument(
        "--vosk-model",
        type=Path,
        default=Path("models/vosk-model-small-en-us-0.15"),
        help="Path to an unpacked local Vosk model directory",
    )
    parser.add_argument("--whisper-model", default="small", help="Whisper model name or local model path")
    parser.add_argument(
        "--whisper-cache",
        type=Path,
        default=Path("models/whisper"),
        help="Git-ignored local cache for a downloaded Whisper model",
    )
    parser.add_argument("--language", default=None, help="Optional spoken-language code, such as en or hi")
    parser.add_argument(
        "--audio-device",
        type=parse_audio_device,
        default=None,
        help="Microphone name or index; use a directional kiosk mic",
    )
    parser.add_argument(
        "--initial-prompt",
        default=None,
        help="Optional domain terms to improve recognition, for example ticket names or station names",
    )
    parser.add_argument("--list-audio-devices", action="store_true", help="Print available microphone devices and exit")
    args = parser.parse_args()

    if args.list_audio_devices:
        import sounddevice as sd

        print(sd.query_devices())
        return

    capture = cv2.VideoCapture(args.camera)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}.")

    detector = MediaPipeFaceMeshDetector(max_faces=args.max_faces)
    transcriber: VisualGateTranscriber | None = None
    started = time.monotonic()
    frames = 0
    try:
        if args.stt:
            if args.stt_backend == "whisper":
                transcriber = WhisperVisualGateTranscriber(
                    model_size=args.whisper_model,
                    language=args.language,
                    input_device=args.audio_device,
                    model_cache_dir=args.whisper_cache,
                    initial_prompt=args.initial_prompt,
                )
            else:
                transcriber = VoskVisualGateTranscriber(args.vosk_model)
        vad = VisualVAD(detector=detector, on_event=lambda event: handle_event(event, transcriber))
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
            if transcriber is not None:
                draw_transcript_window(transcriber.snapshot(), args.stt_backend)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        if transcriber is not None:
            transcriber.close()
        detector.close()
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
