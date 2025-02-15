"""
Tests using real video call recordings.
"""

import pytest
import cv2
import numpy as np
from pathlib import Path
from callanalyser.video.processor import VideoProcessor

def test_load_real_video(real_video_path):
    """Test loading a real video file and checking its properties."""
    with VideoProcessor(str(real_video_path)) as processor:
        # Basic sanity checks
        assert processor.frame_width == 1920
        assert processor.frame_height == 1080
        assert processor.fps > 0
        assert processor.duration > 0
        assert processor.frame_count > 0

def test_extract_frames_from_real_video(real_video_path):
    """Test frame extraction from a real video."""
    with VideoProcessor(str(real_video_path)) as processor:
        # Extract first 3 frames at 1-second intervals
        frames = list(processor.extract_frames(interval=1.0, end_time=3.0))
        
        assert len(frames) == 3
        
        # Check frame properties
        for timestamp, frame in frames:
            assert isinstance(frame, np.ndarray)
            assert frame.shape == (1080, 1920, 3)
            assert 0 <= timestamp <= 3.0

def test_multiple_streams(real_video_streams):
    """Test loading and processing multiple video streams."""
    stream_info = []
    
    # Collect information about each stream
    for stream_path in real_video_streams:
        with VideoProcessor(str(stream_path)) as processor:
            stream_info.append({
                'path': stream_path.name,
                'duration': processor.duration,
                'frame_count': processor.frame_count,
                'fps': processor.fps
            })
    
    # All streams should have similar properties
    base_duration = stream_info[0]['duration']
    base_fps = stream_info[0]['fps']
    
    for info in stream_info:
        # Durations should be within 1 second of each other
        assert abs(info['duration'] - base_duration) < 1.0
        # FPS should match exactly
        assert info['fps'] == base_fps

def test_frame_extraction_consistency(real_video_streams):
    """Test that frame extraction is consistent across streams."""
    timestamp = 5.0  # Extract frame at 5 seconds
    frames = {}
    
    # Extract the same frame from each stream
    for stream_path in real_video_streams:
        with VideoProcessor(str(stream_path)) as processor:
            frame = processor.get_frame_at_timestamp(timestamp)
            assert frame is not None
            frames[stream_path.name] = frame
    
    # All frames should have the same resolution
    resolutions = {name: frame.shape for name, frame in frames.items()}
    assert all(shape == (1080, 1920, 3) for shape in resolutions.values())

def test_video_seeking_stability(real_video_path):
    """Test that seeking to different positions in the video is stable."""
    with VideoProcessor(str(real_video_path)) as processor:
        # Test seeking to various positions
        test_positions = [0.0, 10.0, 30.0, 60.0]  # in seconds
        
        for pos in test_positions:
            frame1 = processor.get_frame_at_timestamp(pos)
            # Seek to the same position again
            frame2 = processor.get_frame_at_timestamp(pos)
            
            assert frame1 is not None
            assert frame2 is not None
            # Frames should be identical when seeking to the same position
            assert np.array_equal(frame1, frame2)

def test_long_interval_extraction(real_video_path):
    """Test extracting frames over a longer interval."""
    with VideoProcessor(str(real_video_path)) as processor:
        # Extract frames every 30 seconds for 2 minutes
        frames = list(processor.extract_frames(
            interval=30.0,
            start_time=0.0,
            end_time=120.0
        ))
        
        # Should get 4 frames (at 0, 30, 60, 90 seconds)
        assert len(frames) == 4
        
        # Check timestamps
        timestamps = [t for t, _ in frames]
        expected_timestamps = [0.0, 30.0, 60.0, 90.0]
        assert timestamps == expected_timestamps 