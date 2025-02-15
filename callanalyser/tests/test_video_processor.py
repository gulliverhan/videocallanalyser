"""
Tests for the video processing module.
"""

import unittest
import tempfile
import cv2
import numpy as np
from pathlib import Path
from callanalyser.video.processor import VideoProcessor

class TestVideoProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Create a temporary test video file."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.video_path = Path(cls.temp_dir) / "test_video.mp4"
        
        # Create a simple test video (black and white frames)
        width, height = 640, 480
        fps = 30
        duration = 2  # seconds
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(cls.video_path), fourcc, fps, (width, height))
        
        # Generate frames
        for i in range(duration * fps):
            # Alternate between black and white frames
            color = 255 if i % 2 == 0 else 0
            frame = np.full((height, width, 3), color, dtype=np.uint8)
            out.write(frame)
        
        out.release()

    def setUp(self):
        """Initialize the video processor before each test."""
        self.processor = VideoProcessor(str(self.video_path))

    def tearDown(self):
        """Clean up after each test."""
        self.processor.cap.release()

    def test_video_properties(self):
        """Test if video properties are correctly loaded."""
        self.assertEqual(self.processor.frame_width, 640)
        self.assertEqual(self.processor.frame_height, 480)
        self.assertEqual(self.processor.fps, 30)
        self.assertEqual(self.processor.duration, 2.0)
        self.assertEqual(self.processor.frame_count, 60)

    def test_get_frame_at_timestamp(self):
        """Test frame extraction at specific timestamps."""
        # Get frame at 0.5 seconds
        frame = self.processor.get_frame_at_timestamp(0.5)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (480, 640, 3))

        # Test invalid timestamp
        frame = self.processor.get_frame_at_timestamp(-1)
        self.assertIsNone(frame)
        frame = self.processor.get_frame_at_timestamp(3.0)
        self.assertIsNone(frame)

    def test_extract_frames(self):
        """Test frame extraction at regular intervals."""
        frames = list(self.processor.extract_frames(interval=0.5))
        
        # Should get 4 frames (at 0, 0.5, 1.0, 1.5 seconds)
        self.assertEqual(len(frames), 4)
        
        # Check frame timestamps
        timestamps = [t for t, _ in frames]
        self.assertEqual(timestamps, [0.0, 0.5, 1.0, 1.5])

    def test_resize_frame(self):
        """Test frame resizing functionality."""
        frame = self.processor.get_frame_at_timestamp(0)
        resized = self.processor.resize_frame(frame, 320, 240)
        self.assertEqual(resized.shape, (240, 320, 3))

if __name__ == '__main__':
    unittest.main() 