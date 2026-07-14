from run_webcam import parse_audio_device


def test_audio_device_numeric_index_is_not_passed_as_a_name() -> None:
    assert parse_audio_device("1") == 1
    assert parse_audio_device("Microphone Array") == "Microphone Array"
