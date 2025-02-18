"""
Module for detecting and classifying different types of Zoom call activities.

This module extends the change detector to:
- Identify common Zoom activities (screen sharing, speaker view, etc.)
- Classify changes based on visual characteristics
- Provide activity-specific analysis
- Use OCR to verify slide content
- Use face detection to identify speaker view
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Generator, Optional, Dict, Iterator
from dataclasses import dataclass
import logging
from .change_detector import ChangeDetector, SceneChange
from collections import deque
import pytesseract
from PIL import Image
import io
from collections import Counter
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class ContentSegment:
    """Represents a segment of content with consistent type."""
    start_time: float
    end_time: float
    content_type: str
    confidence: float
    is_verified: bool = False
    face_count: int = 0  # Track number of faces detected

@dataclass
class ActivityEvent:
    """Represents a segment of activity with consistent content type."""
    start_time: float
    end_time: Optional[float]
    content_type: str
    confidence: float
    metrics: Dict[str, float]
    slide_number: Optional[int] = None  # Track slide number within a presentation

class ContentType:
    """Content types in a Zoom call."""
    SPEAKER = "speaker"          # When faces are detected
    SLIDES = "slides"           # When structured text is detected (default)
    INTERACTIVE = "interactive"  # Dynamic content (browsing, etc)

@dataclass
class ActivityChange(SceneChange):
    """Extends SceneChange with activity classification."""
    activity_type: str
    content_type: str  # One of ContentType values
    details: dict  # Additional activity-specific details

class ActivityDetector(ChangeDetector):
    # Detection thresholds
    SCREEN_SHARE_THRESHOLD = 0.4  # Increased threshold for screen sharing
    SPEAKER_CHANGE_THRESHOLD = 5.0
    SLIDE_EDGE_DENSITY_MIN = 0.008
    SLIDE_EDGE_DENSITY_MAX = 0.045
    SLIDE_TEXT_DENSITY_MIN = 0.002
    SPEAKER_FACE_RATIO_MIN = 0.04
    SPEAKER_FACE_RATIO_STRONG = 0.08
    MOTION_MAGNITUDE_MIN = 3.0  # Lowered to better detect screen sharing
    SLIDE_MIN_DURATION = 5.0  # Increased minimum duration
    MIN_SEGMENT_DURATION = 5.0  # Increased to reduce fragmentation
    SLIDE_CHANGE_THRESHOLD = 15.0
    SLIDE_MAX_DURATION = 300.0
    
    # Confidence thresholds
    MIN_CONFIDENCE = 0.3
    STRONG_CONFIDENCE = 0.7  # Increased for more confident detections
    MEDIUM_CONFIDENCE = 0.5  # Increased medium confidence threshold
    
    def __init__(self, 
                 min_change_threshold: float = 30.0,
                 window_size: int = 5,
                 low_res_width: int = 320,
                 low_res_height: int = 180,
                 use_llm: bool = False,
                 llm_provider: Optional[str] = None,
                 sample_interval: float = 10.0):
        """
        Initialize the activity detector.
        
        Args:
            min_change_threshold: Minimum difference to consider a change
            window_size: Number of frames to look at around potential change
            low_res_width: Width for initial low-res scanning
            low_res_height: Height for initial low-res scanning
            use_llm: Whether to use LLM for content verification
            llm_provider: LLM provider to use if use_llm is True
            sample_interval: Interval between frame samples in seconds
        """
        super().__init__(
            min_change_threshold=min_change_threshold,
            window_size=window_size,
            low_res_width=low_res_width,
            low_res_height=low_res_height,
            sample_interval=sample_interval
        )
        
        self.use_llm = use_llm
        self.llm_provider = llm_provider
        
        # Initialize face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Initialize slide tracking
        self.slide_hashes = {}  # Changed from list to dict for proper tracking
        self.current_slide_number = 0
        self.last_slide_hash = None
        self.slide_change_history = deque(maxlen=10)
        
        if use_llm and not llm_provider:
            logger.warning("LLM verification enabled but no provider specified. Using default (anthropic).")
            self.llm_provider = "anthropic"
        
        # Track content type state
        self.current_content_type = ContentType.SLIDES
        self.content_type_confidence = 0.0
        
        # Motion tracking
        self.prev_frame = None
        self.motion_history = deque(maxlen=10)
        self.content_stability = deque(maxlen=30)
        
        # Add temporal pattern tracking
        self.change_timestamps = deque(maxlen=20)  # Track recent content changes
        self.segment_durations = deque(maxlen=10)  # Track durations between changes
        self.last_change_time = 0.0
        
        # Add temporal smoothing
        self.content_type_history = deque(maxlen=15)
        self.confidence_history = deque(maxlen=15)
        self.temporal_smoothing_window = 10
        
        # Face detection history
        self.face_history = deque(maxlen=5)  # Track recent face detections
        
        # Add LLM batching queue
        self.llm_verification_queue = []
        self.max_batch_size = 4  # Process up to 4 frames at once
        self.min_confidence_for_llm = 0.7  # Only use LLM when confidence is below this
        
        # Add slide tracking
        self.last_slide_frame = None
        self.last_content_type = None
    
    def _compute_motion_metrics(self, frame) -> Dict[str, float]:
        """
        Compute motion-related metrics for a frame.
        
        Args:
            frame: Current frame
            
        Returns:
            Dict with motion metrics
        """
        # Resize frame to low resolution for faster processing
        frame = cv2.resize(frame, (self.low_res_width, self.low_res_height))
        
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Initialize metrics
        metrics = {
            'motion_magnitude': 0.0,
            'motion_std': 0.0,
            'motion_max': 0.0
        }
        
        # If this is the first frame, store it and return default metrics
        if not hasattr(self, '_prev_gray'):
            self._prev_gray = gray
            return metrics
        
        # Ensure frames are the same size
        if gray.shape != self._prev_gray.shape:
            self._prev_gray = cv2.resize(self._prev_gray, (gray.shape[1], gray.shape[0]))
        
        # Calculate optical flow
        try:
            flow = cv2.calcOpticalFlowFarneback(
                self._prev_gray, gray,
                None,  # No initial flow
                0.5,   # Pyramid scale
                3,     # Pyramid levels
                15,    # Window size
                3,     # Iterations
                5,     # Poly neighborhood
                1.2,   # Poly sigma
                0      # Flags
            )
            
            # Calculate motion magnitude
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            
            # Update metrics
            metrics['motion_magnitude'] = float(np.mean(magnitude))
            metrics['motion_std'] = float(np.std(magnitude))
            metrics['motion_max'] = float(np.max(magnitude))
        except cv2.error:
            # If optical flow fails, return default metrics
            pass
        
        # Store current frame for next iteration
        self._prev_gray = gray
        
        return metrics
    
    def _perform_ocr(self, frame: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """
        Temporarily disabled OCR processing.
        Returns default values for metrics.
        """
        return "", 0.0, {
            'avg_text_size': 0.0,
            'text_size_std': 0.0,
            'max_text_size': 0.0,
            'min_text_size': 0.0,
            'text_confidence': 0.0
        }
    
    def _calculate_content_scores(self, metrics: Dict[str, float]) -> Dict[ContentType, float]:
        """Calculate scores for each content type based on frame metrics."""
        scores = {}
        
        # Speaker view score
        speaker_score = 0.0
        if metrics['face_ratio'] > 0.03:  # If significant face presence
            speaker_score = min(1.0, metrics['face_ratio'] * 10)
            # Reduce score if high text density or edge density
            if metrics['text_line_density'] > 0.005 or metrics['edge_density'] > 0.02:
                speaker_score *= 0.5
        scores[ContentType.SPEAKER] = speaker_score
        
        # Slides score
        slides_score = 0.0
        if metrics['text_line_density'] > 0.005:  # If significant text presence
            slides_score = min(1.0, metrics['text_line_density'] * 100)
            # Boost score if stable content and high OCR confidence
            if metrics['content_stability'] > 0.5 and metrics['ocr_confidence'] > 0.3:
                slides_score *= 1.5
            # Reduce score if high motion or face presence
            if metrics['motion_magnitude'] > 2.0 or metrics['face_ratio'] > 0.03:
                slides_score *= 0.5
        scores[ContentType.SLIDES] = min(1.0, slides_score)
        
        # Interactive content score
        interactive_score = 0.0
        # Use multiple indicators for interactive content
        indicators = 0
        score_sum = 0.0
        
        # Edge density (now more lenient)
        if metrics['edge_density'] > 0.005:  # Lowered from 0.02
            indicators += 1
            score_sum += min(1.0, metrics['edge_density'] * 20)  # Increased multiplier
        
        # UI elements
        if metrics['ui_line_density'] > 0.001:
            indicators += 1
            score_sum += min(1.0, metrics['ui_line_density'] * 200)
        
        # Motion
        if metrics['motion_magnitude'] > 1.0:
            indicators += 1
            score_sum += min(1.0, metrics['motion_magnitude'] / 10.0)
        
        # White ratio
        if metrics['white_ratio'] > 0.2:
            indicators += 1
            score_sum += min(1.0, metrics['white_ratio'])
        
        # Calculate average score if we have any indicators
        if indicators > 0:
            interactive_score = score_sum / indicators
            # Boost score if we have multiple indicators
            if indicators >= 2:
                interactive_score *= 1.2
            # Reduce score if very high face ratio
            if metrics['face_ratio'] > 0.05:
                interactive_score *= 0.5
        
        scores[ContentType.INTERACTIVE] = min(1.0, interactive_score)
        
        return scores
        
    def _determine_content_type(self, scores: Dict[ContentType, float]) -> Tuple[ContentType, float]:
        """Determine content type from scores."""
        # Get highest scoring content type
        max_score = 0.0
        content_type = ContentType.SLIDES  # Default to slides
        
        for ctype, score in scores.items():
            if score > max_score:
                max_score = score
                content_type = ctype
        
        # Only switch from default if score exceeds minimum threshold
        min_threshold = 0.15  # Lowered threshold for better sensitivity
        if max_score < min_threshold:
            return ContentType.SLIDES, max_score
            
        return content_type, max_score
    
    def _process_llm_batch(self) -> List[Tuple[str, float]]:
        """
        Process a batch of frames with LLM verification.
        Returns list of (content_type, confidence) tuples.
        """
        if not self.llm_verification_queue:
            return []
            
        from ..utils.llm_api import analyze_frames
        import cv2
        from PIL import Image
        import io
        
        # Prepare batch of frames
        batch_frames = []
        for frames, current_type, proposed_type in self.llm_verification_queue[:self.max_batch_size]:
            # Use middle frame from each transition
            middle_frame = frames[len(frames)//2]
            # Resize for efficiency
            small_frame = cv2.resize(middle_frame, (640, 360))
            
            # Convert BGR to RGB for PIL
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Save to bytes buffer
            img_buffer = io.BytesIO()
            pil_image.save(img_buffer, format='JPEG', quality=85)
            img_data = img_buffer.getvalue()
            
            batch_frames.append((
                img_data,
                current_type,
                proposed_type
            ))
        
        try:
            # Use analyze_frames from our new LLM API
            results = analyze_frames(
                frames=batch_frames,
                # Provider and model will be taken from environment variables
                temperature=0.3  # Lower temperature for more consistent analysis
            )
            
            # Clear processed items from queue
            self.llm_verification_queue = self.llm_verification_queue[len(results):]
            
            # Convert string content types to ContentType enum
            converted_results = []
            for content_type, confidence in results:
                if content_type == "speaker":
                    converted_type = ContentType.SPEAKER
                elif content_type == "slides":
                    converted_type = ContentType.SLIDES
                elif content_type == "interactive":
                    converted_type = ContentType.INTERACTIVE
                else:
                    converted_type = ContentType.SLIDES  # Default to slides
                converted_results.append((converted_type, confidence))
            
            return converted_results
            
        except Exception as e:
            logger.error(f"Error during LLM verification: {str(e)}")
            # Return conservative results on error
            return [(ContentType.SLIDES, 0.5) for _ in batch_frames]
    
    def _analyze_temporal_patterns(self, current_time: float) -> dict:
        """
        Analyze temporal patterns to help distinguish between slides and interactive content.
        
        Args:
            current_time: Current timestamp in seconds
            
        Returns:
            dict: Pattern analysis results
        """
        # Update change history
        if self.change_timestamps and current_time - self.last_change_time > 0.5:  # Debounce changes
            self.change_timestamps.append(current_time)
            duration = current_time - self.last_change_time
            self.segment_durations.append(duration)
            self.last_change_time = current_time
        
        # Analyze patterns
        if len(self.segment_durations) < 2:
            return {
                'likely_slides': False,
                'likely_interactive': False,
                'pattern_confidence': 0.0
            }
        
        # Calculate statistics
        avg_duration = np.mean(self.segment_durations)
        duration_std = np.std(self.segment_durations)
        change_frequency = len(self.change_timestamps) / (current_time - self.change_timestamps[0])
        
        # Determine content patterns
        likely_slides = (
            avg_duration >= self.SLIDE_MIN_DURATION and 
            avg_duration <= self.SLIDE_MAX_DURATION and
            duration_std < avg_duration * 0.5  # Relatively consistent durations
        )
        
        likely_interactive = (
            change_frequency > 1.0 / self.INTERACTIVE_CHANGE_THRESHOLD or
            (avg_duration < self.SLIDE_MIN_DURATION and len(self.segment_durations) > 3)
        )
        
        # Calculate confidence based on sample size and consistency
        pattern_confidence = min(1.0, len(self.segment_durations) / 5.0)  # Increases with more samples
        if duration_std > avg_duration:
            pattern_confidence *= 0.5  # Reduce confidence if durations are very inconsistent
        
        return {
            'likely_slides': likely_slides,
            'likely_interactive': likely_interactive,
            'pattern_confidence': pattern_confidence,
            'avg_duration': avg_duration,
            'change_frequency': change_frequency
        }
    
    def _classify_activity(self, 
                           frame_before: np.ndarray, 
                           frame_after: np.ndarray, 
                           current_time: float,
                           metrics: dict) -> Tuple[ContentType, float]:
        """
        Classify the activity type based on frame analysis and temporal patterns.
        
        Args:
            frame_before: Frame before the change
            frame_after: Frame after the change
            current_time: Current timestamp in seconds
            metrics: Dictionary of computed metrics
            
        Returns:
            Tuple of (ContentType, confidence)
        """
        # Get temporal pattern analysis
        temporal_patterns = self._analyze_temporal_patterns(current_time)
        
        # Extract metrics
        edge_density = metrics.get('edge_density', 0.0)
        brightness = metrics.get('brightness', 0.0)
        face_count = metrics.get('face_count', 0)
        motion_level = metrics.get('motion_level', 0.0)
        text_line_density = metrics.get('text_line_density', 0.0)
        
        # Initialize confidence scores
        scores = {
            ContentType.SLIDES: 0.25,  # Lower base confidence for slides
            ContentType.INTERACTIVE: 0.0,
            ContentType.SPEAKER: 0.0
        }
        
        # Speaker view detection (when we're very confident)
        if face_count > 0:
            face_score = 0.4  # Lower base score
            if face_count > 1:
                face_score += 0.2
            scores[ContentType.SPEAKER] = face_score
            
            # Less aggressive reduction of other scores
            if face_score > 0.5:
                scores[ContentType.SLIDES] *= 0.7
                scores[ContentType.INTERACTIVE] *= 0.7
        
        # More lenient screen sharing detection
        if (motion_level > self.MOTION_MAGNITUDE_MIN or 
            edge_density > self.SLIDE_EDGE_DENSITY_MAX or
            (edge_density > 0.015 and text_line_density > 0.003)):
            scores[ContentType.INTERACTIVE] += 0.4
            
            if temporal_patterns['likely_interactive']:
                scores[ContentType.INTERACTIVE] += 0.2
        
        # Slides detection (now more lenient)
        if text_line_density >= self.SLIDE_TEXT_DENSITY_MIN:
            scores[ContentType.SLIDES] += 0.3
            
            if temporal_patterns['likely_slides']:
                scores[ContentType.SLIDES] += 0.2
        
        # Get the highest scoring content type
        max_score = max(scores.values())
        if max_score < 0.3:  # Lower threshold
            return ContentType.SLIDES, 0.25  # Default to slides with low confidence
        
        content_type = max(scores.items(), key=lambda x: x[1])[0]
        return content_type, max_score

    def _compute_slide_hash(self, frame: np.ndarray) -> str:
        """
        Compute a perceptual hash of a slide frame.
        Uses a combination of downscaling and edge detection to create a robust hash.
        
        Args:
            frame: Input frame
            
        Returns:
            str: Hash string representing the slide content
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        # Resize to a small size (32x32) to remove high-frequency details
        small = cv2.resize(gray, (32, 32))
        
        # Compute edges to focus on structural content
        edges = cv2.Canny(small, 50, 150)
        
        # Compute the hash using mean values
        avg = np.mean(edges)
        hash_bits = (edges > avg).flatten()
        
        # Convert boolean array to hash string
        hash_str = ''.join(['1' if b else '0' for b in hash_bits])
        return hash_str
        
    def _is_similar_hash(self, hash1: str, hash2: str, threshold: int = 50) -> bool:
        """
        Compare two hashes to determine if they represent similar slides.
        
        Args:
            hash1: First hash string
            hash2: Second hash string
            threshold: Maximum number of different bits to consider similar
            
        Returns:
            bool: True if hashes are similar
        """
        if not hash1 or not hash2:
            return False
        
        # Count differing bits
        diff_bits = sum(b1 != b2 for b1, b2 in zip(hash1, hash2))
        return diff_bits <= threshold

    def _is_new_slide(self, frame: np.ndarray, metrics: Dict[str, float]) -> bool:
        """
        Determine if a frame represents a new slide using perceptual hashing.
        """
        # Compute hash for current frame
        current_hash = self._compute_slide_hash(frame)
        
        # If this is the first slide, store its hash and return True
        if not self.last_slide_hash:
            self.last_slide_hash = current_hash
            self.slide_hashes[current_hash] = 1  # First slide is number 1
            self.current_slide_number = 1
            return True
        
        # Check if enough time has passed since last slide change
        if hasattr(self, 'last_slide_time'):
            time_since_last = metrics.get('timestamp', 0) - self.last_slide_time
            if time_since_last < self.SLIDE_MIN_DURATION:
                return False
        
        # Check if frame is a transition effect
        if metrics.get('is_transition', False):
            return False
        
        # Check if the frame has strong slide-like properties
        edge_density = metrics.get('edge_density', 0)
        text_density = metrics.get('text_line_density', 0)
        if (edge_density < self.SLIDE_EDGE_DENSITY_MIN * 1.5 or
            text_density < self.SLIDE_TEXT_DENSITY_MIN * 1.5):
            return False
        
        # Compare with previous slide hash
        if not self._is_similar_hash(current_hash, self.last_slide_hash):
            # Check if we've seen this slide before
            for existing_hash, slide_num in self.slide_hashes.items():
                if self._is_similar_hash(current_hash, existing_hash):
                    # We've seen this slide before, use its number
                    self.current_slide_number = slide_num
                    self.last_slide_hash = current_hash
                    self.last_slide_time = metrics.get('timestamp', 0)
                    metrics['slide_number'] = slide_num  # Add slide number to metrics
                    return True
            
            # This is a new unique slide
            self.current_slide_number = len(self.slide_hashes) + 1
            self.slide_hashes[current_hash] = self.current_slide_number
            self.last_slide_hash = current_hash
            self.last_slide_time = metrics.get('timestamp', 0)
            metrics['slide_number'] = self.current_slide_number  # Add slide number to metrics
            return True
        
        # Not a new slide, but still update metrics with current slide number
        metrics['slide_number'] = self.current_slide_number
        return False

    def detect_activities(self, video_path: str, sample_interval: float = 1.0) -> List[ActivityEvent]:
        """
        Detect activities in a video file.
        
        Args:
            video_path: Path to the video file
            sample_interval: Time interval between frame samples in seconds
            
        Returns:
            List of ActivityEvent objects
        """
        activities = []
        current_type = None
        current_start = 0
        current_metrics = defaultdict(float)
        current_count = 0
        current_confidence = 0.0
        
        # Process video frames
        for timestamp, metrics in self._process_video(video_path, sample_interval):
            # Classify content type for current frame
            content_type, confidence = self._classify_content_type(metrics)
            
            # Initialize or update current segment
            if current_type is None:
                current_type = content_type
                current_start = timestamp
                current_metrics = defaultdict(float)
                current_count = 0
                current_confidence = confidence
            
            # Update running metrics
            for key, value in metrics.items():
                if key == 'frame':  # Skip frame data
                    continue
                if value is not None:  # Only add non-None values
                    if isinstance(value, (int, float)):
                        current_metrics[key] += float(value)
                    else:
                        current_metrics[key] = value  # For non-numeric values, just use the latest
            current_count += 1
            
            # Check if content type has changed
            if content_type != current_type:
                # Only create a new segment if:
                # 1. Current segment is long enough (>= 5 seconds)
                # 2. We have enough confidence in the new type
                segment_duration = timestamp - current_start
                if segment_duration >= 5.0 and confidence >= 0.5:
                    # Create activity event for previous segment
                    if current_count > 0:
                        avg_metrics = {}
                        for key, value in current_metrics.items():
                            if isinstance(value, (int, float)):
                                avg_metrics[key] = value / current_count
                            else:
                                avg_metrics[key] = value
                        
                        # Handle slide number specially
                        slide_num = None
                        if current_type == ContentType.SLIDES:
                            if 'slide_number' in avg_metrics:
                                slide_num = int(avg_metrics['slide_number'])
                            elif hasattr(self, 'current_slide_number'):
                                slide_num = self.current_slide_number
                        
                        activities.append(ActivityEvent(
                            start_time=current_start,
                            end_time=timestamp,
                            content_type=current_type,
                            confidence=current_confidence,
                            metrics=avg_metrics,
                            slide_number=slide_num
                        ))
                    
                    # Start new segment
                    current_type = content_type
                    current_start = timestamp
                    current_metrics = defaultdict(float)
                    current_count = 0
                    current_confidence = confidence
            
            # Update confidence for current segment
            current_confidence = (current_confidence + confidence) / 2  # Running average
        
        # Add final segment if it's long enough
        if current_count > 0 and (timestamp - current_start) >= 5.0:
            avg_metrics = {}
            for key, value in current_metrics.items():
                if isinstance(value, (int, float)):
                    avg_metrics[key] = value / current_count
                else:
                    avg_metrics[key] = value
            
            # Handle slide number specially
            slide_num = None
            if current_type == ContentType.SLIDES:
                if 'slide_number' in avg_metrics:
                    slide_num = int(avg_metrics['slide_number'])
                elif hasattr(self, 'current_slide_number'):
                    slide_num = self.current_slide_number
            
            activities.append(ActivityEvent(
                start_time=current_start,
                end_time=timestamp,
                content_type=current_type,
                confidence=current_confidence,
                metrics=avg_metrics,
                slide_number=slide_num
            ))
        
        return activities

    def _analyze_frame_content(self, frame: np.ndarray) -> dict:
        """
        Analyze a single frame for content-specific metrics.
        
        Args:
            frame: Input frame
            
        Returns:
            Dictionary of computed metrics
        """
        # Store the frame in metrics for slide detection
        stats = {'frame': frame}
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100),
            maxSize=(400, 400)
        )
        face_area = sum(w * h for (x, y, w, h) in faces)
        frame_area = frame.shape[0] * frame.shape[1]
        face_ratio = face_area / frame_area
        
        # Perform OCR and get text metrics
        text, ocr_score, text_metrics = self._perform_ocr(frame)
        
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
        
        # Get motion metrics
        motion_stats = self._compute_motion_metrics(frame)
        
        # Combine all metrics
        stats.update({
            'brightness': avg_brightness,
            'brightness_variation': std_brightness,
            'is_transition': is_transition,
            'edge_density': edge_density,
            'center_brightness': center_brightness,
            'center_variation': center_std,
            'text_line_density': text_line_density,
            'face_ratio': face_ratio,
            'face_count': len(faces),
            'ocr_score': ocr_score
        })
        
        # Add text metrics
        stats.update(text_metrics)
        
        # Add motion metrics
        stats.update(motion_stats)
        
        return stats

    def _classify_content_type(self, metrics: Dict[str, float]) -> Tuple[ContentType, float]:
        """Classify the content type based on the metrics."""
        # First check for speaker view - must have faces and low text density
        if metrics['face_count'] > 0 and metrics['face_ratio'] >= self.SPEAKER_FACE_RATIO_MIN:
            if metrics['text_line_density'] < self.SLIDE_TEXT_DENSITY_MIN * 2:
                return ContentType.SPEAKER, 0.8
        
        # Check for screen sharing first - look for UI patterns and motion
        if (metrics['motion_magnitude'] > self.MOTION_MAGNITUDE_MIN or 
            metrics['edge_density'] < self.SLIDE_EDGE_DENSITY_MIN or
            metrics['brightness_variation'] > 70):
            
            confidence = 0.5
            if metrics['motion_magnitude'] > self.MOTION_MAGNITUDE_MIN * 1.5:
                confidence += 0.2
            if metrics['edge_density'] < self.SLIDE_EDGE_DENSITY_MIN * 0.8:
                confidence += 0.2
            if metrics['face_count'] == 0:
                confidence += 0.1
                
            # Clear slide number for non-slide content
            metrics['slide_number'] = None
            return ContentType.INTERACTIVE, min(0.9, confidence)
        
        # Check for slides - must have structured text and stable content
        if metrics['text_line_density'] >= self.SLIDE_TEXT_DENSITY_MIN:
            if metrics['motion_magnitude'] <= self.MOTION_MAGNITUDE_MIN:
                if metrics['edge_density'] >= self.SLIDE_EDGE_DENSITY_MIN:
                    # Update slide number if this is a new slide
                    self._is_new_slide(metrics.get('frame'), metrics)
                    confidence = min(1.0, metrics['text_line_density'] / self.SLIDE_TEXT_DENSITY_MIN)
                    return ContentType.SLIDES, max(0.6, confidence)
        
        # Clear slide number for non-slide content
        metrics['slide_number'] = None
        return ContentType.INTERACTIVE, 0.5

    def _process_video(self, video_path: str, sample_interval: float = 1.0) -> Iterator[Tuple[float, Dict[str, float]]]:
        """
        Process video frames at given sample interval.
        
        Args:
            video_path: Path to video file
            sample_interval: Time interval between frame samples in seconds
            
        Yields:
            Tuple of (timestamp, metrics)
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        
        # Calculate frame interval from sample interval
        frame_interval = int(fps * sample_interval)
        
        # Reset slide tracking state for new video
        self.slide_hashes = {}
        self.current_slide_number = 0
        self.last_slide_hash = None
        self.slide_change_history.clear()
        
        frame_idx = 0
        while frame_idx < frame_count:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
                
            timestamp = frame_idx / fps
            metrics = self._analyze_frame_content(frame)
            
            # Check for new slide if content type might be slides
            if metrics['text_line_density'] >= self.SLIDE_TEXT_DENSITY_MIN:
                is_new_slide = self._is_new_slide(frame, metrics)
                if is_new_slide:
                    metrics['slide_number'] = self.current_slide_number
            
            yield timestamp, metrics
            frame_idx += frame_interval
        
        cap.release() 