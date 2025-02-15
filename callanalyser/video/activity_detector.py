"""
Module for detecting and classifying different types of Zoom call activities.

This module extends the change detector to:
- Identify common Zoom activities (screen sharing, speaker view, etc.)
- Classify changes based on visual characteristics
- Provide activity-specific analysis
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Generator, Optional, Dict
from dataclasses import dataclass
import logging
from .change_detector import ChangeDetector, SceneChange
from collections import deque

logger = logging.getLogger(__name__)

class ContentType:
    """Enumeration of content types in a Zoom call."""
    SPEAKER_VIEW = "speaker_view"
    SCREEN_SHARE = "screen_share"
    SLIDES = "slides"
    WHITEBOARD = "whiteboard"
    UNKNOWN = "unknown"

@dataclass
class ActivityChange(SceneChange):
    """Extends SceneChange with activity classification."""
    activity_type: str
    content_type: str  # One of ContentType values
    details: dict  # Additional activity-specific details

class ActivityDetector(ChangeDetector):
    # Activity detection thresholds - adjusted to be more sensitive
    SCREEN_SHARE_THRESHOLD = 0.7  # Lowered to detect more screen sharing transitions
    SPEAKER_CHANGE_THRESHOLD = 15.0  # More sensitive to speaker changes
    
    # Content type detection parameters - refined based on reference
    SLIDE_EDGE_DENSITY_MIN = 0.03  # Lower minimum for slides
    SLIDE_EDGE_DENSITY_MAX = 0.12  # Lower maximum for slides
    WHITEBOARD_BRIGHTNESS_MIN = 170  # Adjusted for better whiteboard detection
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Adjust default parameters for more sensitive detection
        self.min_change_threshold = 10.0  # More sensitive to changes
        self.window_size = 2  # Smaller window for quicker detection
        
        # Track content type state
        self.current_content_type = ContentType.SPEAKER_VIEW
        self.content_type_confidence = 0.0
        self.last_stable_content_type = ContentType.SPEAKER_VIEW
        
        # Motion tracking
        self.prev_frame = None
        self.motion_history = deque(maxlen=10)  # Track last 10 frames of motion
        self.content_stability = deque(maxlen=30)  # Track content stability over time
    
    def _compute_motion_metrics(self, frame: np.ndarray) -> dict:
        """
        Compute motion-related metrics between consecutive frames.
        
        Args:
            frame: Current frame
            
        Returns:
            dict: Motion metrics
        """
        if self.prev_frame is None:
            self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return {
                'motion_magnitude': 0,
                'motion_area': 0,
                'motion_smoothness': 0,
                'content_stability': 1.0  # Initialize as fully stable
            }
        
        # Convert current frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_frame, gray, None, 
            pyr_scale=0.5, levels=3, winsize=15, 
            iterations=3, poly_n=5, poly_sigma=1.2, 
            flags=0
        )
        
        # Calculate motion magnitude and direction
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Calculate motion metrics
        motion_magnitude = np.mean(magnitude)
        motion_area = np.mean(magnitude > 0.5)  # Areas with significant motion
        
        # Calculate motion smoothness (variation in motion magnitude)
        motion_smoothness = np.std(magnitude)
        
        # Update motion history
        self.motion_history.append(motion_magnitude)
        
        # Calculate content stability
        if len(self.motion_history) > 1:
            stability = 1.0 - min(1.0, sum(self.motion_history) / len(self.motion_history))
        else:
            stability = 1.0  # Initialize as fully stable
        
        self.content_stability.append(stability)
        
        # Update previous frame
        self.prev_frame = gray
        
        return {
            'motion_magnitude': motion_magnitude,
            'motion_area': motion_area,
            'motion_smoothness': motion_smoothness,
            'content_stability': np.mean(self.content_stability) if self.content_stability else 1.0
        }
    
    def _analyze_frame_content(self, frame: np.ndarray) -> dict:
        """
        Analyze frame content for specific characteristics.
        
        Args:
            frame: Input frame
            
        Returns:
            dict: Frame characteristics
        """
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        # Compute various metrics
        avg_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        # Check for transition frames (very dark/black frames)
        is_transition = avg_brightness < 5 and std_brightness < 2
        
        # Detect edges for layout analysis
        edges = cv2.Canny(gray, 50, 150)  # More sensitive edge detection
        edge_density = np.mean(edges > 0)
        
        # Analyze different regions of the frame
        h, w = gray.shape
        top_bar = gray[0:h//10, :]  # Top 10% of frame
        bottom_bar = gray[9*h//10:, :]  # Bottom 10% of frame
        center_region = gray[h//4:3*h//4, w//4:3*w//4]  # Center 50% of frame
        
        # Compute region-specific metrics
        center_brightness = np.mean(center_region)
        center_std = np.std(center_region)
        
        # Detect text-like content (horizontal lines in edges)
        horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        text_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_structure)
        text_line_density = np.mean(text_lines > 0)
        
        # Detect UI elements (vertical lines for menus/panels)
        vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
        ui_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_structure)
        ui_line_density = np.mean(ui_lines > 0)
        
        # Analyze top bar for Zoom UI elements (participant list, chat, etc.)
        top_edges = cv2.Canny(top_bar, 50, 150)
        top_ui_density = np.mean(top_edges > 0)
        
        # Analyze color distribution for whiteboard detection
        if len(frame.shape) == 3:
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # Calculate color variance in each channel
            color_variance = np.std(hsv, axis=(0,1))
            # Calculate white pixel ratio (high V, low S in HSV)
            white_mask = (hsv[:,:,1] < 30) & (hsv[:,:,2] > 200)
            white_ratio = np.mean(white_mask)
        else:
            color_variance = np.array([0, 0, 0])
            white_ratio = 0
        
        # Get motion metrics
        motion_stats = self._compute_motion_metrics(frame)
        
        # Combine all metrics
        stats = {
            'brightness': avg_brightness,
            'brightness_variation': std_brightness,
            'is_transition': is_transition,
            'edge_density': edge_density,
            'top_bar_brightness': np.mean(top_bar),
            'bottom_bar_brightness': np.mean(bottom_bar),
            'center_brightness': center_brightness,
            'center_variation': center_std,
            'text_line_density': text_line_density,
            'ui_line_density': ui_line_density,
            'top_ui_density': top_ui_density,
            'color_variance': color_variance,
            'white_ratio': white_ratio
        }
        stats.update(motion_stats)
        
        return stats
    
    def _detect_content_type(self, frame: np.ndarray, stats: dict) -> str:
        """
        Detect the type of content being shown.
        
        Args:
            frame: Input frame
            stats: Frame statistics from _analyze_frame_content
            
        Returns:
            str: Content type (one of ContentType values)
        """
        # Initialize confidence for this frame
        confidence = 0.0
        detected_type = ContentType.UNKNOWN
        
        # Check for speaker view first - it has distinct characteristics
        if ((stats['brightness_variation'] > 15 or  # Video content variation
             stats['center_variation'] > 20) and  # High variation in center (where video is)
            stats['ui_line_density'] > 0.008 and  # UI elements present
            stats['top_ui_density'] > 0.08 and  # Zoom controls in top bar
            (stats['motion_magnitude'] > 0.15 or  # Some motion from video
             stats['motion_area'] > 0.05) and  # Motion in a significant area
            stats['edge_density'] < 0.15):  # Not too complex (like screen sharing)
            detected_type = ContentType.SPEAKER_VIEW
            confidence = 0.85 + min(0.15, stats['motion_magnitude'])
        
        # Check for whiteboard - almost pure white screen with minimal UI
        elif (stats['white_ratio'] > 0.85 and  # Almost entirely white
              stats['ui_line_density'] < 0.005 and  # Almost no UI elements
              stats['top_ui_density'] < 0.05 and  # No Zoom controls at top
              stats['edge_density'] < 0.03):  # Very few edges (except for drawings)
            detected_type = ContentType.WHITEBOARD
            confidence = 0.9
        
        # Check for slides - static content with structured layout and high stability
        elif ((self.SLIDE_EDGE_DENSITY_MIN < stats['edge_density'] < self.SLIDE_EDGE_DENSITY_MAX or
               stats['text_line_density'] > 0.015) and  # Text-like content or moderate edges
              stats['content_stability'] > 0.8 and  # Very stable content
              stats['motion_magnitude'] < 0.5 and  # Limited motion
              stats['motion_area'] < 0.15 and  # Small areas of motion
              stats['top_ui_density'] > 0.05):  # Some UI controls visible
            detected_type = ContentType.SLIDES
            confidence = 0.8 + min(0.2, stats['content_stability'] - 0.8)
        
        # Check for screen sharing - more dynamic content with continuous motion
        elif (stats['top_ui_density'] > 0.08 and  # Zoom sharing controls visible
              (stats['motion_magnitude'] > 0.3 or  # Significant motion
               stats['motion_area'] > 0.1 or  # Larger areas of motion
               stats['motion_smoothness'] > 0.4 or  # Variable motion (scrolling, mouse movement)
               stats['content_stability'] < 0.75 or  # Less stable content
               stats['edge_density'] > 0.08)):  # Complex content
            detected_type = ContentType.SCREEN_SHARE
            confidence = 0.8
        
        # Additional checks for speaker view (fallback)
        elif ((stats['brightness_variation'] > 12 or  # Lower threshold for variation
               stats['center_variation'] > 15) and  # Lower threshold for center variation
              stats['ui_line_density'] > 0.005 and  # Minimal UI presence
              stats['top_ui_density'] > 0.05 and  # Some controls visible
              stats['motion_magnitude'] > 0.1):  # Any noticeable motion
            detected_type = ContentType.SPEAKER_VIEW
            confidence = 0.7
        
        # If we have high stability and UI, it might be slides
        elif (stats['content_stability'] > 0.75 and 
              stats['ui_line_density'] > 0.01 and
              stats['top_ui_density'] > 0.05 and
              stats['motion_magnitude'] < 0.3):  # Not too much motion
            detected_type = ContentType.SLIDES
            confidence = 0.7
            
        # If we have continuous motion and UI elements, default to screen share
        elif ((stats['motion_magnitude'] > 0.2 or 
               stats['motion_area'] > 0.08 or
               stats['content_stability'] < 0.8) and
              stats['ui_line_density'] > 0.01 and
              stats['top_ui_density'] > 0.05):
            detected_type = ContentType.SCREEN_SHARE
            confidence = 0.7
        
        # If we have UI elements but don't match other patterns
        elif stats['top_ui_density'] > 0.08:
            # Bias towards speaker view if we see video-like characteristics
            if (stats['brightness_variation'] > 10 or
                stats['center_variation'] > 12 or
                stats['motion_magnitude'] > 0.08):
                detected_type = ContentType.SPEAKER_VIEW
                confidence = 0.6
            else:
                detected_type = ContentType.SCREEN_SHARE
                confidence = 0.6
        
        # During transitions or uncertain periods, bias towards the last stable content type
        if confidence < 0.6 and self.content_type_confidence > 0.7:
            detected_type = self.last_stable_content_type
            confidence = 0.5
        
        # Update state
        self.content_type_confidence = confidence
        if confidence > 0.7:
            self.last_stable_content_type = detected_type
        
        return detected_type
    
    def _classify_activity(self, 
                          frame_before: np.ndarray, 
                          frame_after: np.ndarray, 
                          difference: float) -> Tuple[str, str, dict]:
        """
        Classify the type of activity and content based on frame analysis.
        
        Args:
            frame_before: Frame before change
            frame_after: Frame after change
            difference: Difference score between frames
            
        Returns:
            Tuple[str, str, dict]: (activity_type, content_type, details)
        """
        before_stats = self._analyze_frame_content(frame_before)
        after_stats = self._analyze_frame_content(frame_after)
        
        # Track transition frames
        had_transition = before_stats['is_transition'] or after_stats['is_transition']
        
        # Detect content types
        content_before = self._detect_content_type(frame_before, before_stats)
        content_after = self._detect_content_type(frame_after, after_stats)
        
        # Compute changes in metrics
        brightness_change = abs(after_stats['brightness'] - before_stats['brightness'])
        edge_change = abs(after_stats['edge_density'] - before_stats['edge_density'])
        ui_change = abs(after_stats['ui_line_density'] - before_stats['ui_line_density'])
        
        # Determine activity type with increased confidence if we saw a transition
        confidence_boost = 1.5 if had_transition else 1.0
        
        if content_before != content_after:
            # Content type changed
            activity_type = "content_change"
            details = {
                'previous_content': content_before,
                'new_content': content_after,
                'brightness_change': brightness_change,
                'edge_change': edge_change,
                'had_transition': had_transition,
                'confidence_boost': confidence_boost,
                'content_confidence': self.content_type_confidence
            }
        elif content_after in [ContentType.SLIDES, ContentType.WHITEBOARD]:
            # Slide transition or whiteboard update
            activity_type = "content_update"
            details = {
                'content_type': content_after,
                'brightness_change': brightness_change,
                'edge_change': edge_change,
                'had_transition': had_transition,
                'content_confidence': self.content_type_confidence
            }
        elif content_after == ContentType.SPEAKER_VIEW and brightness_change > self.SPEAKER_CHANGE_THRESHOLD:
            # Speaker change in speaker view
            activity_type = "speaker_change"
            details = {
                'brightness_change': brightness_change,
                'edge_change': edge_change,
                'ui_change': ui_change,
                'had_transition': had_transition,
                'content_confidence': self.content_type_confidence
            }
        else:
            # Minor update within the same content type
            activity_type = "content_update"
            details = {
                'content_type': content_after,
                'brightness_change': brightness_change,
                'edge_change': edge_change,
                'had_transition': had_transition,
                'content_confidence': self.content_type_confidence
            }
        
        # Update current content type
        self.current_content_type = content_after
        
        return activity_type, content_after, details
    
    def detect_activities(self, 
                         video_path: str, 
                         min_confidence: float = 0.5) -> Generator[ActivityChange, None, None]:
        """
        Detect and classify activities in a video file.
        
        Args:
            video_path: Path to video file
            min_confidence: Minimum confidence to report a change
            
        Yields:
            ActivityChange: Detected and classified activities
        """
        self.current_content_type = ContentType.SPEAKER_VIEW  # Reset state
        
        for change in super().detect_changes(video_path, min_confidence):
            # Classify the activity
            activity_type, content_type, details = self._classify_activity(
                change.frame_before,
                change.frame_after,
                self._compute_frame_difference(change.frame_before, change.frame_after)
            )
            
            # Adjust confidence based on transition frames
            adjusted_confidence = change.confidence
            if details.get('had_transition', False):
                adjusted_confidence *= details.get('confidence_boost', 1.5)
            
            if adjusted_confidence >= min_confidence:
                yield ActivityChange(
                    timestamp=change.timestamp,
                    confidence=adjusted_confidence,
                    frame_before=change.frame_before,
                    frame_after=change.frame_after,
                    activity_type=activity_type,
                    content_type=content_type,
                    details=details
                ) 