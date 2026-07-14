from visual_vad.activity import VisualActivityDetector
from visual_vad.config import VADConfig


def test_debounce_emits_one_start_and_one_end() -> None:
    detector = VisualActivityDetector(
        VADConfig(
            ema_alpha=1.0,
            history_seconds=0.5,
            motion_threshold=0.005,
            variance_threshold=0.10,
            start_hold_seconds=0.15,
            end_hold_seconds=0.40,
            face_missing_end_seconds=0.40,
        )
    )
    assert detector.update(0.30, 0.00).event_type is None
    assert detector.update(0.31, 0.05).event_type is None
    assert detector.update(0.32, 0.11).event_type is None
    started = detector.update(0.33, 0.21)
    assert started.event_type == "speech_start"
    assert started.speaking

    assert detector.update(0.33, 0.25).event_type is None
    ended = detector.update(0.33, 0.66)
    assert ended.event_type == "speech_end"
    assert not ended.speaking


def test_missing_face_ends_an_active_utterance() -> None:
    detector = VisualActivityDetector(
        VADConfig(
            ema_alpha=1.0,
            history_seconds=0.5,
            motion_threshold=0.005,
            variance_threshold=0.10,
            start_hold_seconds=0.1,
            end_hold_seconds=0.4,
            face_missing_end_seconds=0.3,
        )
    )
    detector.update(0.30, 0.0)
    detector.update(0.31, 0.05)
    detector.update(0.32, 0.16)
    assert detector.speaking
    assert detector.update(None, 0.20).event_type is None
    assert detector.update(None, 0.51).event_type == "speech_end"


def test_landmark_jitter_does_not_start_speech() -> None:
    detector = VisualActivityDetector(
        VADConfig(
            ema_alpha=1.0,
            history_seconds=0.5,
            motion_threshold=0.015,
            variance_threshold=0.002,
            variance_motion_floor=0.006,
            start_hold_seconds=0.1,
            end_hold_seconds=0.4,
            face_missing_end_seconds=0.4,
        )
    )
    for timestamp, mar in ((0.0, 0.300), (0.05, 0.304), (0.11, 0.300), (0.18, 0.304), (0.25, 0.300)):
        update = detector.update(mar, timestamp)
        assert not update.raw_activity
        assert not update.speaking
        assert update.event_type is None
