"""
Tests for the VTT transcript parser.
"""

import pytest
from pathlib import Path
from callanalyser.utils.transcript import (
    parse_timestamp,
    parse_vtt,
    get_text_at_timestamp,
    TranscriptSegment
)

def test_parse_timestamp():
    """Test timestamp parsing."""
    assert parse_timestamp("00:00:00.000") == 0.0
    assert parse_timestamp("00:00:01.500") == 1.5
    assert parse_timestamp("00:01:00.000") == 60.0
    assert parse_timestamp("01:00:00.000") == 3600.0
    assert parse_timestamp("01:23:45.678") == 5025.678

def test_parse_vtt_file(transcript_path):
    """Test parsing a real VTT file."""
    segments = list(parse_vtt(transcript_path))
    
    # Basic checks
    assert len(segments) > 0
    
    # Check segment structure
    for segment in segments:
        assert isinstance(segment, TranscriptSegment)
        assert segment.start_time >= 0
        assert segment.end_time > segment.start_time
        assert isinstance(segment.text, str)
        assert len(segment.text) > 0
        
        # If speaker is present, it should be a non-empty string
        if segment.speaker is not None:
            assert isinstance(segment.speaker, str)
            assert len(segment.speaker) > 0

def test_get_text_at_timestamp(transcript_path):
    """Test retrieving text at specific timestamps."""
    # Get all segments first
    segments = list(parse_vtt(transcript_path))
    
    # Test with the timestamp from the middle of the first segment
    first_segment = segments[0]
    mid_time = (first_segment.start_time + first_segment.end_time) / 2
    
    result = get_text_at_timestamp(transcript_path, mid_time)
    assert result is not None
    assert result.text == first_segment.text
    assert result.start_time == first_segment.start_time
    assert result.end_time == first_segment.end_time
    
    # Test with a timestamp between segments (should return None)
    if len(segments) > 1:
        gap_time = (segments[0].end_time + segments[1].start_time) / 2
        result = get_text_at_timestamp(transcript_path, gap_time)
        assert result is None
    
    # Test with a timestamp way before the first segment
    result = get_text_at_timestamp(transcript_path, -1.0)
    assert result is None
    
    # Test with a timestamp way after the last segment
    last_segment = segments[-1]
    result = get_text_at_timestamp(transcript_path, last_segment.end_time + 10.0)
    assert result is None

def test_transcript_continuity(transcript_path):
    """Test that transcript segments are continuous and non-overlapping."""
    segments = list(parse_vtt(transcript_path))
    
    for i in range(len(segments) - 1):
        current = segments[i]
        next_segment = segments[i + 1]
        
        # Segments should not overlap
        assert current.end_time <= next_segment.start_time
        
        # Gap between segments should be small (if any)
        gap = next_segment.start_time - current.end_time
        assert gap >= 0
        assert gap < 1.0  # Assuming gaps shouldn't be more than 1 second

def test_speaker_extraction(transcript_path):
    """Test that speaker information is correctly extracted."""
    segments = list(parse_vtt(transcript_path))
    
    # Count segments with speaker information
    segments_with_speaker = [s for s in segments if s.speaker is not None]
    
    # Log some statistics
    total_segments = len(segments)
    segments_with_speaker_count = len(segments_with_speaker)
    
    print(f"\nTranscript statistics:")
    print(f"Total segments: {total_segments}")
    print(f"Segments with speaker: {segments_with_speaker_count}")
    print(f"Speaker detection rate: {segments_with_speaker_count/total_segments:.1%}")
    
    # If we have segments with speakers, check their format
    if segments_with_speaker:
        for segment in segments_with_speaker:
            assert isinstance(segment.speaker, str)
            assert len(segment.speaker) > 0
            assert ': ' not in segment.text  # Speaker should be removed from text 