"""
Video processing module for extracting and managing video frames.

This module provides functionality for:
- Loading video files
- Extracting frames at specified intervals or timestamps
- Basic frame manipulation and caching
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Generator
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self, video_path: str):
        """
        Initialize the video processor with a video file path.
        
        Args:
            video_path (str): Path to the video file
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")
        
        # Get video properties
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Loaded video: {video_path}")
        logger.info(f"Duration: {self.duration:.2f}s, FPS: {self.fps}, "
                   f"Resolution: {self.frame_width}x{self.frame_height}")

    def get_frame_at_timestamp(self, timestamp: float) -> Optional[np.ndarray]:
        """
        Extract a frame at a specific timestamp.
        
        Args:
            timestamp (float): Time in seconds
            
        Returns:
            Optional[np.ndarray]: Frame image if successful, None otherwise
        """
        if timestamp < 0 or timestamp > self.duration:
            logger.warning(f"Timestamp {timestamp} is out of range [0, {self.duration}]")
            return None
        
        frame_number = int(timestamp * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        
        if not ret:
            logger.warning(f"Failed to read frame at timestamp {timestamp}")
            return None
            
        return frame

    def extract_frames(self, 
                      interval: float = 1.0,
                      start_time: float = 0.0,
                      end_time: Optional[float] = None) -> Generator[Tuple[float, np.ndarray], None, None]:
        """
        Extract frames from the video at regular intervals.
        
        Args:
            interval (float): Time interval between frames in seconds
            start_time (float): Start time in seconds
            end_time (Optional[float]): End time in seconds, or None for video end
            
        Yields:
            Tuple[float, np.ndarray]: Timestamp and frame pairs
        """
        if end_time is None:
            end_time = self.duration
            
        if start_time < 0 or start_time >= self.duration:
            logger.error(f"Invalid start time: {start_time}")
            return
            
        if end_time <= start_time or end_time > self.duration:
            logger.error(f"Invalid end time: {end_time}")
            return
            
        current_time = start_time
        while current_time < end_time:
            frame = self.get_frame_at_timestamp(current_time)
            if frame is not None:
                yield current_time, frame
            current_time += interval

    def resize_frame(self, frame: np.ndarray, width: int, height: int) -> np.ndarray:
        """
        Resize a frame to the specified dimensions.
        
        Args:
            frame (np.ndarray): Input frame
            width (int): Target width
            height (int): Target height
            
        Returns:
            np.ndarray: Resized frame
        """
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cap.release() 