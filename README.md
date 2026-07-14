# Real-Time Visual Voice Activity Detection

A standalone webcam module for a ticket-kiosk AI agent. It detects all visible faces, locks onto one primary kiosk user, measures only that person's lip motion, and emits debounced `speech_start` / `speech_end` events. It deliberately has no microphone or audio dependency.

## What is implemented

- **Primary-user selection:** a weighted score of face area (distance proxy, 60%) and centrality (40%). After selection, the same person is retained by bounding-box IoU tracking. A larger or more animated bystander cannot steal the channel. A new track is chosen only after the old primary is absent for 750 ms.
- **Visual VAD:** MediaPipe Face Mesh supplies lip landmarks. Mouth aspect ratio (MAR) movement is exponentially smoothed and combined with short-window MAR variability. Variance alone is insufficient: current mouth movement must also clear a configurable floor, which suppresses closed-mouth landmark jitter.
- **Stable events:** time-based start and end holds (160 ms / 550 ms by default) prevent chatter and make behavior less dependent on webcam frame rate.
- **Resilience:** no-face periods produce a clean `speech_end` after the configured grace period; lost faces, face edges, and low-confidence detection are therefore treated as uncertainty rather than as a crash or immediate switch.
- **Observability:** the demo shows every detected face, the selected primary, speaking state, visual signal, FPS, and processing time. JSON events go to stdout.

## Setup

Python 3.10+ is required. Create and activate a virtual environment, then install the module:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the unit tests:

```powershell
python -m pytest
```

Start the webcam demo:

```powershell
python run_webcam.py --camera 0
```

Press `q` or `Esc` to stop. Use a different `--camera` index if needed.

## Optional local STT transcript window

The V-VAD core is strictly visual and never opens a microphone. The demo can optionally use its debounced visual events to gate a **local**, multilingual Whisper speech-to-text stream and show a second transcript window. Audio is processed on the machine and is captured only while the visual VAD is in the speaking state. Whisper's own audio VAD filters non-speech before decoding, and low-confidence segments are withheld instead of being shown as a transcript.

Install the optional package, then run the default `small` Whisper model. Its model files are downloaded and cached under the git-ignored `models/whisper/` folder on first use. Leave `--language` unset for automatic language detection, or set it when the kiosk has a known primary language:

```powershell
python -m pip install -e ".[stt]"
python run_webcam.py --camera 0 --stt --whisper-model small --language en
```

`--stt-backend vosk` remains available as a lightweight fallback and requires `python -m pip install -e ".[stt-vosk]"` plus a local model directory.

### Crowded-kiosk deployment

Software alone cannot identify which person supplied a sound when several people reach the same omnidirectional microphone. The visual gate reduces when audio is transcribed, but it is not an audio source separator. For a production kiosk, use a directional microphone array co-located and calibrated with the camera, enable its beamforming/noise suppression, and select it through `--audio-device`. Measure word-error rate by accent, language, microphone placement, noise level, and customer distance before deployment. Never treat an STT transcript as an authoritative transaction instruction without confirmation.

Use this mode only for the interactive demo; it is not part of the assessment's visual-only VAD contract.

## Integration contract

The kiosk service supplies a callback when it constructs `VisualVAD`. The callback is invoked synchronously and in-order only for debounced events:

```python
from visual_vad import VisualVAD, VVADEvent
from visual_vad.mediapipe_backend import MediaPipeFaceMeshDetector

def send_to_kiosk(event: VVADEvent) -> None:
    # event.as_dict() is JSON-ready.
    print(event.as_dict())

vad = VisualVAD(
    detector=MediaPipeFaceMeshDetector(),
    on_event=send_to_kiosk,
)

# Per BGR OpenCV frame:
result = vad.process_frame(frame)
```

Each callback payload has this shape:

```json
{
  "event_type": "speech_start",
  "timestamp": 1234.567,
  "primary_track_id": 7,
  "latency_ms": 161.4,
  "signal": 1.83
}
```

`timestamp` is monotonic-clock seconds, appropriate for ordering and latency calculation inside the running process. `latency_ms` is the observed mouth-motion-to-event delay caused by the start/end debounce. The kiosk integration can attach a wall-clock timestamp at its boundary if it needs cross-service time correlation.

If the kiosk already has its own vision detector, call `process_observations(faces, frame_size, timestamp)` instead. The only required measurement per detected face is a bounding box and mouth aspect ratio; see `FaceObservation`.

## Tuning and limitations

All behavioral constants are exposed as `VisualVADConfig`, `PrimarySelectorConfig`, and `VADConfig`; no threshold is buried in the pipeline. `variance_motion_floor` is the key false-positive control for a still, closed mouth. Calibrate it together with `motion_threshold`, `variance_threshold`, and the debounce holds against representative kiosk camera footage before production rollout.

This is visual speech activity detection, not speech recognition and not identity verification. Occluded mouths, masks, extreme profiles, very small faces, poor lighting, or silent mouth movement can reduce accuracy. MediaPipe's face tracker can lose a face at an image edge; the explicit no-face timeout handles that case conservatively. For a production rollout, measure precision/recall against labelled kiosk video and record CPU, FPS, and event latency on the target hardware.

## Assessment demo checklist

Record a 2-4 minute screen-and-webcam capture that demonstrates:

1. A single primary user speaking and going quiet, with JSON and overlay events visible.
2. A background person talking or waving while the quiet primary remains selected, with no false primary event.
3. The primary leaving and a new customer becoming primary after the hand-off timeout.

During the recording, show the overlay's FPS and per-frame processing time. The default target is at least 10-15 FPS on a CPU; actual sustained FPS and latency must be reported from the machine used for the recording.
