"""
Pytest configuration and fixtures for video processing tests.
"""

import pytest
import tempfile
import cv2
import numpy as np
from pathlib import Path
import shutil

@pytest.fixture(scope="session")
def test_data_dir():
    """Return path to the calls directory containing real test data."""
    return Path("calls")

@pytest.fixture(scope="session")
def synthetic_video_path():
    """Create a synthetic test video with known properties."""
    temp_dir = tempfile.mkdtemp()
    video_path = Path(temp_dir) / "synthetic_test.mp4"
    
    # Create a simple test video (black and white frames)
    width, height = 640, 480
    fps = 30
    duration = 2  # seconds
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    
    try:
        # Generate frames
        for i in range(duration * fps):
            # Alternate between black and white frames
            color = 255 if i % 2 == 0 else 0
            frame = np.full((height, width, 3), color, dtype=np.uint8)
            out.write(frame)
    finally:
        out.release()
    
    yield video_path
    
    # Cleanup
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="session")
def real_video_path(test_data_dir):
    """Return path to a real test video."""
    # Use the main recording file
    video_path = test_data_dir / "GMT20241125-094826_Recording_1920x1080.mp4"
    if not video_path.exists():
        pytest.skip("Real test video not found")
    return video_path

@pytest.fixture(scope="session")
def real_video_streams(test_data_dir):
    """Return paths to all video streams from the recording."""
    pattern = "GMT20241125-094826_Recording_*_1920x1080.mp4"
    streams = list(test_data_dir.glob(pattern))
    if not streams:
        pytest.skip("No video streams found")
    return streams

@pytest.fixture(scope="session")
def transcript_path(test_data_dir):
    """Return path to the VTT transcript file."""
    vtt_path = test_data_dir / "GMT20241125-094826_Recording.transcript.vtt"
    if not vtt_path.exists():
        pytest.skip("Transcript file not found")
    return vtt_path 