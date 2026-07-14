from visual_vad.transcription import merge_transcript_text


def test_merges_repeated_overlap_from_live_chunks() -> None:
    merged = merge_transcript_text("book a ticket to central", "to central station tomorrow")
    assert merged == "book a ticket to central station tomorrow"


def test_merges_without_overlap() -> None:
    merged = merge_transcript_text("book a ticket", "for two passengers")
    assert merged == "book a ticket for two passengers"
