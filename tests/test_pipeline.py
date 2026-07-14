from visual_vad.config import PrimarySelectorConfig, VADConfig, VisualVADConfig
from visual_vad.models import BoundingBox, FaceObservation
from visual_vad.pipeline import VisualVAD


def face(x: float, y: float, size: float, mar: float, index: int) -> FaceObservation:
    return FaceObservation(BoundingBox(x, y, size, size), mar, index)


def test_background_mouth_motion_does_not_emit_primary_event() -> None:
    config = VisualVADConfig(
        primary=PrimarySelectorConfig(missing_timeout_seconds=0.5),
        vad=VADConfig(
            ema_alpha=1.0,
            history_seconds=0.5,
            motion_threshold=0.005,
            variance_threshold=0.10,
            start_hold_seconds=0.1,
            end_hold_seconds=0.25,
            face_missing_end_seconds=0.25,
        ),
    )
    callback_events = []
    vad = VisualVAD(on_event=callback_events.append, config=config)
    frame_size = (640, 480)
    primary = lambda mar: face(250, 120, 190, mar, 0)
    background = lambda mar: face(20, 100, 100, mar, 1)

    vad.process_observations([primary(0.30), background(0.30)], frame_size, 0.0)
    vad.process_observations([primary(0.31), background(0.80)], frame_size, 0.05)
    started = vad.process_observations([primary(0.32), background(0.10)], frame_size, 0.16)
    assert [event.event_type for event in started.events] == ["speech_start"]

    vad.process_observations([primary(0.32), background(0.90)], frame_size, 0.20)
    ended = vad.process_observations([primary(0.32), background(0.10)], frame_size, 0.46)
    assert [event.event_type for event in ended.events] == ["speech_end"]
    assert [event.event_type for event in callback_events] == ["speech_start", "speech_end"]
