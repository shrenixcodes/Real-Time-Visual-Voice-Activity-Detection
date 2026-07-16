from collections import deque
import threading

from visual_vad.transcription import _WhisperJob, WhisperVisualGateTranscriber


def bare_transcriber() -> WhisperVisualGateTranscriber:
    transcriber = object.__new__(WhisperVisualGateTranscriber)
    transcriber._lock = threading.Lock()
    transcriber._job_ready = threading.Condition(transcriber._lock)
    transcriber._jobs = deque()
    transcriber._closed = False
    transcriber._recording = False
    transcriber._active_session = 1
    transcriber._audio_buffers = {1: bytearray(b"first request")}
    transcriber._partial = ""
    return transcriber


def test_new_live_audio_does_not_evict_a_previous_final_job() -> None:
    transcriber = bare_transcriber()
    transcriber._queue_final(1)
    transcriber._active_session = 2
    with transcriber._lock:
        transcriber._enqueue_live_revision(2, b"second request")

    assert list(transcriber._jobs) == [
        _WhisperJob(1, b"first request", True),
        _WhisperJob(2, b"second request", False),
    ]


def test_speech_end_queues_an_immutable_final_job() -> None:
    transcriber = bare_transcriber()
    transcriber._queue_final(1)

    assert transcriber._active_session is None
    assert 1 not in transcriber._audio_buffers
    assert list(transcriber._jobs) == [_WhisperJob(1, b"first request", True)]
