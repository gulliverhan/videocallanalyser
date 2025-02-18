"""
Module for analyzing video content metrics and generating metric reports.

This module provides functionality to:
- Collect frame-level metrics from video segments
- Analyze metrics by content type
- Generate metric reports and statistics
"""

import cv2
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Tuple
import json
from .activity_detector import ActivityDetector, ContentType

logger = logging.getLogger(__name__)

def parse_timestamp(time_str: str) -> float:
    """Convert MM:SS format to seconds."""
    parts = time_str.split(':')
    return int(parts[0]) * 60 + int(parts[1])

def parse_ground_truth(actions_file: str) -> List[Tuple[float, str]]:
    """Parse ground truth timestamps and content types."""
    segments = []
    with open(actions_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            time_str, action = line.split(' ', 1)
            timestamp = parse_timestamp(time_str)
            
            # Map actions to content types
            content_type = ContentType.SPEAKER
            if 'screen sharing' in action or 'screenshare' in action:
                content_type = ContentType.INTERACTIVE
            elif 'slides' in action:
                content_type = ContentType.SLIDES
            elif 'whiteboard' in action:
                content_type = ContentType.INTERACTIVE
            
            segments.append((timestamp, content_type))
    return segments

def collect_segment_metrics(video_path: str, start_time: float, end_time: float, detector: ActivityDetector) -> Dict:
    """
    Collect metrics for a specific segment of the video.
    
    Args:
        video_path: Path to the video file
        start_time: Start time in seconds
        end_time: End time in seconds
        detector: ActivityDetector instance for metric computation
        
    Returns:
        Dict: Aggregated metrics for the segment
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Sample frames every 10 seconds
    interval = 10.0  # Using 10-second intervals
    sample_interval = int(fps * interval)
    metrics_list = []
    
    # Set initial position
    start_frame = int(start_time * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    current_time = start_time
    while current_time < end_time:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Skip frames according to sample interval
        for _ in range(sample_interval - 1):
            cap.grab()
        
        # Collect metrics for this frame
        stats = detector._analyze_frame_content(frame)
        
        # Add face detection metrics
        faces = detector.face_cascade.detectMultiScale(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100),
            maxSize=(400, 400)
        )
        face_area = sum(w * h for (x, y, w, h) in faces)
        frame_area = frame.shape[0] * frame.shape[1]
        stats['face_count'] = len(faces)
        stats['face_ratio'] = face_area / frame_area
        
        # Remove OCR analysis as it's very slow
        stats['ocr_confidence'] = 0.0  # Set a default value
        
        metrics_list.append(stats)
        current_time += interval
    
    cap.release()
    
    # Calculate aggregate statistics
    metrics_summary = {}
    for key in metrics_list[0].keys():
        values = [m[key] for m in metrics_list]
        metrics_summary[key] = {
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'mean': float(np.mean(values)),
            'std': float(np.std(values))
        }
    
    return metrics_summary

def analyze_content_metrics(video_path: str, ground_truth_segments: List[Tuple[float, str]], detector: ActivityDetector) -> Dict:
    """
    Analyze metrics for each content type based on ground truth segments.
    
    Args:
        video_path: Path to the video file
        ground_truth_segments: List of (timestamp, content_type) tuples
        detector: ActivityDetector instance for metric computation
        
    Returns:
        Dict: Metrics analysis by content type
    """
    metrics_by_type = {}
    
    # Analyze each segment
    for i in range(len(ground_truth_segments) - 1):
        start_time, content_type = ground_truth_segments[i]
        end_time = ground_truth_segments[i + 1][0]
        
        logger.info(f"\nAnalyzing {content_type} segment from {start_time:.1f}s to {end_time:.1f}s")
        
        # Collect metrics for this segment
        metrics = collect_segment_metrics(video_path, start_time, end_time, detector)
        
        # Print detailed metrics for this segment
        logger.info("Segment metrics:")
        key_metrics = [
            'face_ratio',
            'text_line_density',
            'edge_density',
            'motion_magnitude',
            'content_stability',
            'ocr_confidence',
            'ui_line_density',
            'white_ratio'
        ]
        for metric in key_metrics:
            if metric in metrics:
                m = metrics[metric]
                logger.info(f"  {metric}:")
                logger.info(f"    range: {m['min']:.3f} - {m['max']:.3f}")
                logger.info(f"    mean ± std: {m['mean']:.3f} ± {m['std']:.3f}")
        
        # Store metrics by content type
        if content_type not in metrics_by_type:
            metrics_by_type[content_type] = []
        metrics_by_type[content_type].append(metrics)
    
    # Calculate aggregate metrics for each content type
    content_type_analysis = {}
    for content_type, segment_metrics_list in metrics_by_type.items():
        # Combine metrics from all segments of this type
        combined_metrics = {}
        metric_keys = segment_metrics_list[0].keys()
        
        for key in metric_keys:
            all_values = []
            for segment_metrics in segment_metrics_list:
                stats = segment_metrics[key]
                all_values.extend([
                    stats['min'],
                    stats['max'],
                    stats['mean']
                ])
            
            combined_metrics[key] = {
                'min': float(np.min(all_values)),
                'max': float(np.max(all_values)),
                'mean': float(np.mean(all_values)),
                'std': float(np.std(all_values))
            }
        
        content_type_analysis[content_type] = combined_metrics
    
    return content_type_analysis

def generate_metric_report(metrics: Dict, output_file: Path) -> None:
    """
    Generate a detailed metric report.
    
    Args:
        metrics: Metrics dictionary by content type
        output_file: Path to save the report
    """
    # Save full metrics as JSON
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Print key findings to console
    logger.info("\nKey metric ranges by content type:")
    for content_type, content_metrics in metrics.items():
        logger.info(f"\n{content_type}:")
        # Print most discriminative metrics
        key_metrics = [
            'face_ratio',
            'text_line_density',
            'edge_density',
            'motion_magnitude',
            'content_stability',
            'ocr_confidence'
        ]
        for metric in key_metrics:
            if metric in content_metrics:
                m = content_metrics[metric]
                logger.info(f"  {metric}:")
                logger.info(f"    range: {m['min']:.3f} - {m['max']:.3f}")
                logger.info(f"    mean ± std: {m['mean']:.3f} ± {m['std']:.3f}")

def analyze_video_metrics(video_path: str, actions_file: str, output_dir: Path) -> Dict:
    """
    Analyze video metrics using ground truth actions file.
    
    Args:
        video_path: Path to the video file
        actions_file: Path to the ground truth actions file
        output_dir: Directory to save output files
        
    Returns:
        Dict: Content metrics analysis
    """
    # Initialize detector
    detector = ActivityDetector(
        min_change_threshold=10.0,
        window_size=2,
        low_res_width=320,
        low_res_height=180
    )
    
    # Parse ground truth
    ground_truth_segments = parse_ground_truth(actions_file)
    
    # Analyze metrics
    content_metrics = analyze_content_metrics(video_path, ground_truth_segments, detector)
    
    # Generate report
    video_name = Path(video_path).stem
    metrics_file = output_dir / f"content_metrics_{video_name}.json"
    generate_metric_report(content_metrics, metrics_file)
    
    return content_metrics

def collect_interval_metrics(video_path: str, detector: ActivityDetector) -> List[Dict]:
    """
    Collect metrics at 10-second intervals for the entire video.
    
    Args:
        video_path: Path to the video file
        detector: ActivityDetector instance for metric computation
        
    Returns:
        List[Dict]: List of metrics for each interval
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    
    interval = 10.0  # Using 10-second intervals
    metrics_list = []
    current_time = 0.0
    
    logger.info(f"Collecting metrics at {interval}s intervals for {duration:.1f}s video")
    
    while current_time < duration:
        # Set position to current time
        frame_number = int(current_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Collect metrics for this frame
        metrics = detector._analyze_frame_content(frame)
        metrics['timestamp'] = current_time
        metrics_list.append(metrics)
        
        # Log progress at each interval
        logger.info(f"Processed {current_time:.1f}s / {duration:.1f}s ({current_time/duration*100:.1f}%)")
        
        current_time += interval
    
    cap.release()
    return metrics_list

def compare_with_ground_truth(metrics_list: List[Dict], ground_truth_segments: List[Tuple[float, str]]) -> Dict:
    """
    Compare collected metrics with ground truth segments.
    
    Args:
        metrics_list: List of metrics at intervals
        ground_truth_segments: List of (timestamp, content_type) tuples
        
    Returns:
        Dict: Analysis of metrics by content type
    """
    # Group metrics by ground truth content type
    metrics_by_type = {}
    
    for metrics in metrics_list:
        timestamp = metrics['timestamp']
        # Find corresponding ground truth segment
        content_type = None
        for i in range(len(ground_truth_segments) - 1):
            start_time = ground_truth_segments[i][0]
            end_time = ground_truth_segments[i + 1][0]
            if start_time <= timestamp < end_time:
                content_type = ground_truth_segments[i][1]
                break
        
        if content_type:
            if content_type not in metrics_by_type:
                metrics_by_type[content_type] = []
            metrics_by_type[content_type].append(metrics)
    
    # Calculate statistics for each content type
    analysis = {}
    key_metrics = [
        'edge_density',
        'brightness',
        'face_count',
        'text_structure',
        'motion_level',
        'content_stability',
        'face_ratio',
        'text_line_density',
        'ui_line_density',
        'white_ratio'
    ]
    
    for content_type, type_metrics in metrics_by_type.items():
        analysis[content_type] = {}
        for metric in key_metrics:
            if metric in type_metrics[0]:
                values = [m[metric] for m in type_metrics]
                analysis[content_type][metric] = {
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'values': values  # Keep raw values for distribution analysis
                }
    
    return analysis

def analyze_training_videos(video_paths: List[str], truth_paths: List[str], output_dir: Path):
    """Analyze training videos and save metrics."""
    from .activity_detector import ActivityDetector
    import cv2
    import json
    
    detector = ActivityDetector()
    results = {}
    
    for video_path, truth_path in zip(video_paths, truth_paths):
        logger.info(f"\nAnalyzing video: {video_path}")
        
        # Read ground truth
        with open(truth_path, 'r') as f:
            ground_truth = []
            for line in f:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                # Split on first space, rest is content type description
                time_str, content_type = line.split(' ', 1)
                # Convert MM:SS to seconds
                minutes, seconds = map(int, time_str.split(':'))
                time_seconds = minutes * 60 + seconds
                
                ground_truth.append({
                    'time': time_seconds,
                    'content_type': content_type
                })
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        interval = 10.0  # Using 10-second intervals
        logger.info(f"Collecting metrics at {interval}s intervals for {duration:.1f}s video")
        
        # Collect metrics every 10 seconds
        metrics = []
        current_time = 0.0
        while current_time < duration:
            # Log progress at each interval
            logger.info(f"Processed {current_time:.1f}s / {duration:.1f}s ({100 * current_time / duration:.1f}%)")
            
            # Seek to current time
            frame_num = int(current_time * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
                
            # Analyze frame
            frame_metrics = detector._analyze_frame_content(frame)
            
            # Convert numpy types to Python types for JSON serialization
            serializable_metrics = {}
            for key, value in frame_metrics.items():
                if isinstance(value, (np.integer, np.floating)):
                    serializable_metrics[key] = float(value)
                elif isinstance(value, (bool, int, float, str)):
                    serializable_metrics[key] = value
                elif isinstance(value, np.ndarray):
                    serializable_metrics[key] = value.tolist()
                else:
                    serializable_metrics[key] = str(value)
            
            metrics.append({
                'time': current_time,
                'metrics': serializable_metrics
            })
            
            current_time += interval  # Use the defined interval
        
        cap.release()
        
        # Find ground truth content type for each metric
        for metric in metrics:
            time = metric['time']
            # Find last ground truth entry before this time
            content_type = None
            for entry in ground_truth:
                if entry['time'] <= time:
                    content_type = entry['content_type']
                else:
                    break
            metric['ground_truth'] = content_type
        
        results[video_path] = {
            'metrics': metrics,
            'ground_truth': ground_truth
        }
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(results, f, indent=2) 