"""
Module for detecting significant changes in video streams.

This module provides functionality to:
- Detect scene changes using frame differences
- Work at multiple resolutions for efficiency
- Generate frame pairs for each detected change
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Generator, NamedTuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class SceneChange:
    """Represents a detected scene change."""
    timestamp: float
    confidence: float
    frame_before: np.ndarray
    frame_after: np.ndarray

class ChangeDetector:
    def __init__(self, 
                 min_change_threshold: float = 30.0,  # Minimum difference to consider a change
                 window_size: int = 5,               # Number of frames to look at around potential change
                 low_res_width: int = 320,          # Width for initial low-res scanning
                 low_res_height: int = 180,         # Height for initial low-res scanning
                 sample_interval: float = 10.0):     # Sample interval in seconds
        """
        Initialize the change detector.
        
        Args:
            min_change_threshold: Minimum difference to consider a change
            window_size: Number of frames to look at around potential change
            low_res_width: Width for initial low-res scanning
            low_res_height: Height for initial low-res scanning
            sample_interval: Sample interval in seconds
        """
        self.min_change_threshold = min_change_threshold
        self.window_size = window_size
        self.low_res_width = low_res_width
        self.low_res_height = low_res_height
        self.sample_interval = sample_interval
        
    def _compute_frame_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Compute the difference between two frames.
        
        Args:
            frame1: First frame
            frame2: Second frame
            
        Returns:
            float: Difference score between 0 and 100
        """
        # Convert to grayscale if needed
        if len(frame1.shape) == 3:
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        if len(frame2.shape) == 3:
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
        # Compute absolute difference
        diff = cv2.absdiff(frame1, frame2)
        
        # Normalize difference score to 0-100 range
        score = np.mean(diff) * (100.0 / 255.0)
        
        return score
    
    def _is_significant_change(self, 
                             differences: List[float], 
                             current_idx: int) -> Tuple[bool, float]:
        """
        Determine if a point represents a significant change by looking at surrounding frames.
        
        Args:
            differences: List of frame differences
            current_idx: Index to check
            
        Returns:
            Tuple[bool, float]: (is_change, confidence)
        """
        if current_idx < self.window_size or current_idx >= len(differences) - self.window_size:
            return False, 0.0
            
        # Get the current difference
        current_diff = differences[current_idx]
        
        # If below threshold, not a change
        if current_diff < self.min_change_threshold:
            return False, 0.0
            
        # Look at surrounding window
        window_before = differences[current_idx - self.window_size:current_idx]
        window_after = differences[current_idx + 1:current_idx + self.window_size + 1]
        
        # Current difference should be significantly larger than surrounding windows
        avg_before = np.mean(window_before)
        avg_after = np.mean(window_after)
        
        # Compute how many times larger the current difference is
        ratio_before = current_diff / max(avg_before, 1.0)
        ratio_after = current_diff / max(avg_after, 1.0)
        
        # Both ratios should be significant
        is_change = ratio_before > 2.0 and ratio_after > 2.0
        
        # Confidence based on how much larger the difference is
        confidence = min(ratio_before, ratio_after) * (current_diff / 100.0)
        
        return is_change, confidence

    def detect_changes(self, video_path: str, min_confidence: float = 0.5) -> Generator[SceneChange, None, None]:
        """
        Detect scene changes in a video file using the specified interval sampling.
        
        Args:
            video_path: Path to video file
            min_confidence: Minimum confidence to report a change
            
        Yields:
            SceneChange: Detected scene changes
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        
        logger.info(f"Processing video with {frame_count} frames at {fps} FPS")
        logger.info(f"Video duration: {duration:.1f} seconds")
        
        # Use sample_interval from initialization
        base_interval = int(fps * self.sample_interval)  # Convert seconds to frames
        logger.info(f"Sampling interval: {base_interval} frames ({self.sample_interval:.1f} seconds)")
        
        current_frame_idx = 0
        
        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            return
            
        # Log initial progress
        logger.info(f"Starting analysis at frame 0/{frame_count} (0.0%)")
            
        # Resize for efficiency
        prev_frame_small = cv2.resize(prev_frame, (self.low_res_width, self.low_res_height))
        
        while current_frame_idx < frame_count:
            # Log progress more frequently (every 250 frames)
            if current_frame_idx % 250 == 0:
                logger.info(f"Processed {current_frame_idx}/{frame_count} frames ({current_frame_idx/frame_count*100:.1f}%)")
            
            # Skip frames according to interval
            for _ in range(base_interval - 1):
                ret = cap.grab()  # Just grab frame without decoding
                if not ret:
                    break
                current_frame_idx += 1
            
            # Read the next frame to analyze
            ret, current_frame = cap.read()
            if not ret:
                break
                
            current_frame_idx += 1
            
            # Compute difference on low-res version
            current_frame_small = cv2.resize(current_frame, (self.low_res_width, self.low_res_height))
            diff = self._compute_frame_difference(prev_frame_small, current_frame_small)
            
            # If significant difference detected, do a brief burst of more frequent sampling
            if diff > self.min_change_threshold:
                # Store the frames where we detected the change
                frame_before = prev_frame
                frame_after = current_frame
                timestamp = current_frame_idx / fps
                
                # Sample a few more frames at 1-second intervals to catch any immediate follow-up changes
                burst_interval = int(fps)  # 1 second
                for _ in range(3):  # Check next 3 seconds
                    # Skip to next sample point
                    for _ in range(burst_interval - 1):
                        ret = cap.grab()
                        if not ret:
                            break
                        current_frame_idx += 1
                    
                    ret, burst_frame = cap.read()
                    if not ret:
                        break
                    
                    current_frame_idx += 1
                    burst_frame_small = cv2.resize(burst_frame, (self.low_res_width, self.low_res_height))
                    burst_diff = self._compute_frame_difference(current_frame_small, burst_frame_small)
                    
                    # If we detect another significant change, yield it too
                    if burst_diff > self.min_change_threshold:
                        yield SceneChange(
                            timestamp=current_frame_idx / fps,
                            confidence=burst_diff / 100.0,
                            frame_before=current_frame,
                            frame_after=burst_frame
                        )
                    
                    current_frame = burst_frame
                    current_frame_small = burst_frame_small
                
                # Yield the original change
                yield SceneChange(
                    timestamp=timestamp,
                    confidence=diff / 100.0,
                    frame_before=frame_before,
                    frame_after=frame_after
                )
            
            prev_frame = current_frame
            prev_frame_small = current_frame_small
        
        cap.release()
        logger.info("Completed change detection") 