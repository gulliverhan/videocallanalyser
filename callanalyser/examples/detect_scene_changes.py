"""
Script to detect and visualize scene changes in a video.
"""

import cv2
import numpy as np
from pathlib import Path
import logging
from callanalyser.video.change_detector import ChangeDetector
from callanalyser.utils.transcript import get_text_at_timestamp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_change_visualization(output_dir: Path, scene_change, index: int):
    """Save before/after frames and create a side-by-side visualization."""
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Save individual frames
    cv2.imwrite(str(output_dir / f"change_{index:04d}_before.jpg"), scene_change.frame_before)
    cv2.imwrite(str(output_dir / f"change_{index:04d}_after.jpg"), scene_change.frame_after)
    
    # Create side-by-side visualization
    h, w = scene_change.frame_before.shape[:2]
    vis_width = 1920  # Fixed width for visualization
    scale = vis_width / (w * 2)  # Scale to fit two frames side by side
    vis_height = int(h * scale)
    
    # Resize frames
    frame_before_resized = cv2.resize(scene_change.frame_before, (vis_width//2, vis_height))
    frame_after_resized = cv2.resize(scene_change.frame_after, (vis_width//2, vis_height))
    
    # Create side-by-side visualization
    visualization = np.hstack([frame_before_resized, frame_after_resized])
    
    # Add text overlay
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    color = (255, 255, 255)
    thickness = 2
    
    # Add timestamp and confidence
    cv2.putText(visualization, 
                f"Time: {scene_change.timestamp:.1f}s", 
                (10, 30), font, font_scale, color, thickness)
    cv2.putText(visualization, 
                f"Confidence: {scene_change.confidence:.2f}", 
                (10, 70), font, font_scale, color, thickness)
    
    # Save visualization
    cv2.imwrite(str(output_dir / f"change_{index:04d}_visualization.jpg"), visualization)

def main():
    # Initialize detector
    detector = ChangeDetector(
        min_change_threshold=30.0,  # Adjust this based on your video
        window_size=5,
        low_res_width=320,
        low_res_height=180
    )
    
    # Process main video recording
    video_path = "calls/GMT20241125-094826_Recording_1920x1080.mp4"
    vtt_path = "calls/GMT20241125-094826_Recording.transcript.vtt"
    output_dir = Path("detected_changes")
    
    logger.info("Starting scene change detection...")
    
    # Detect changes
    changes = []
    for i, change in enumerate(detector.detect_changes(video_path, min_confidence=0.5)):
        logger.info(f"Detected change at {change.timestamp:.1f}s (confidence: {change.confidence:.2f})")
        
        # Save visualization
        save_change_visualization(output_dir, change, i)
        
        # Get transcript context
        transcript_segment = get_text_at_timestamp(vtt_path, change.timestamp)
        if transcript_segment:
            logger.info(f"Transcript context: [{transcript_segment.speaker}] {transcript_segment.text}")
        
        changes.append(change)
    
    logger.info(f"\nDetected {len(changes)} scene changes")
    
    # Write summary report
    with open(output_dir / "changes_summary.txt", "w") as f:
        f.write(f"Scene Changes Summary\n")
        f.write(f"===================\n\n")
        f.write(f"Total changes detected: {len(changes)}\n\n")
        
        for i, change in enumerate(changes):
            f.write(f"\nChange {i+1}:\n")
            f.write(f"  Timestamp: {change.timestamp:.1f}s\n")
            f.write(f"  Confidence: {change.confidence:.2f}\n")
            
            # Add transcript context
            transcript = get_text_at_timestamp(vtt_path, change.timestamp)
            if transcript:
                f.write(f"  Transcript: [{transcript.speaker}] {transcript.text}\n")
            
            f.write(f"  Files:\n")
            f.write(f"    - change_{i:04d}_before.jpg\n")
            f.write(f"    - change_{i:04d}_after.jpg\n")
            f.write(f"    - change_{i:04d}_visualization.jpg\n")

if __name__ == "__main__":
    main() 