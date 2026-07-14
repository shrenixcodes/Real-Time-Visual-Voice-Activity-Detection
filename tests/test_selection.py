from visual_vad.config import PrimarySelectorConfig
from visual_vad.models import BoundingBox, FaceObservation
from visual_vad.selection import PrimaryUserSelector


def face(x: float, y: float, width: float, height: float, mar: float = 0.3) -> FaceObservation:
    return FaceObservation(BoundingBox(x, y, width, height), mar)


def test_primary_stays_locked_when_bystander_becomes_larger() -> None:
    selector = PrimaryUserSelector(PrimarySelectorConfig(missing_timeout_seconds=0.5))
    user = face(250, 120, 190, 190)
    bystander = face(20, 100, 120, 120)
    first = selector.update([user, bystander], (640, 480), 0.0)
    assert first is not None
    assert first.track_id == 1
    assert first.face == user

    same_user = face(258, 122, 192, 192)
    moving_bystander = face(120, 70, 300, 300)
    second = selector.update([same_user, moving_bystander], (640, 480), 0.1)
    assert second is not None
    assert second.track_id == 1
    assert second.face == same_user


def test_handoff_occurs_only_after_primary_missing_timeout() -> None:
    selector = PrimaryUserSelector(PrimarySelectorConfig(missing_timeout_seconds=0.5))
    old_user = face(250, 120, 190, 190)
    new_user = face(40, 120, 210, 210)
    assert selector.update([old_user], (640, 480), 0.0) is not None
    assert selector.update([new_user], (640, 480), 0.2) is None
    handoff = selector.update([new_user], (640, 480), 0.71)
    assert handoff is not None
    assert handoff.track_id == 2
