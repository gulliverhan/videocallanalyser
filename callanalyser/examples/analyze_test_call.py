"""
Script to analyze test call activities and visualize the results.
"""

import cv2
import numpy as np
from pathlib import Path
import logging
from callanalyser.video.activity_detector import ActivityDetector, ActivityChange
from callanalyser.utils.transcript import get_text_at_timestamp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_activity_visualization(output_dir: Path, activity: ActivityChange, index: int):
    """Save visualization of detected activity."""
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Save individual frames
    cv2.imwrite(str(output_dir / f"activity_{index:04d}_before.jpg"), activity.frame_before)
    cv2.imwrite(str(output_dir / f"activity_{index:04d}_after.jpg"), activity.frame_after)
    
    # Create side-by-side visualization
    h, w = activity.frame_before.shape[:2]
    vis_width = 1920  # Fixed width for visualization
    scale = vis_width / (w * 2)  # Scale to fit two frames side by side
    vis_height = int(h * scale)
    
    # Resize frames
    frame_before_resized = cv2.resize(activity.frame_before, (vis_width//2, vis_height))
    frame_after_resized = cv2.resize(activity.frame_after, (vis_width//2, vis_height))
    
    # Create side-by-side visualization
    visualization = np.hstack([frame_before_resized, frame_after_resized])
    
    # Add text overlay
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    color = (255, 255, 255)
    thickness = 2
    
    # Add activity information
    cv2.putText(visualization, 
                f"Time: {activity.timestamp:.1f}s", 
                (10, 30), font, font_scale, color, thickness)
    cv2.putText(visualization, 
                f"Activity: {activity.activity_type}", 
                (10, 70), font, font_scale, color, thickness)
    cv2.putText(visualization, 
                f"Confidence: {activity.confidence:.2f}", 
                (10, 110), font, font_scale, color, thickness)
    
    # Add key metrics from details
    y_pos = 150
    for key, value in activity.details.items():
        if isinstance(value, (int, float)):
            cv2.putText(visualization,
                       f"{key}: {value:.2f}",
                       (10, y_pos), font, font_scale * 0.8, color, thickness)
            y_pos += 30
    
    # Save visualization
    cv2.imwrite(str(output_dir / f"activity_{index:04d}_visualization.jpg"), visualization)

def analyze_video(video_path: str, output_dir: Path, vtt_path: str = None):
    """
    Analyze video for activities, with optional transcript support.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save results
        vtt_path: Optional path to VTT transcript file
    """
    # Initialize detector with parameters tuned for Zoom
    detector = ActivityDetector(
        min_change_threshold=15.0,
        window_size=3,
        low_res_width=320,
        low_res_height=180
    )
    
    logger.info("Starting activity detection...")
    logger.info(f"Processing video: {video_path}")
    if vtt_path:
        logger.info(f"Using transcript: {vtt_path}")
    else:
        logger.info("No transcript provided - running pure visual analysis")
    
    # Track activities by type
    activities = []
    activity_types = set()
    
    # Detect and classify activities
    for i, activity in enumerate(detector.detect_activities(video_path, min_confidence=0.3)):
        logger.info(f"Detected {activity.activity_type} at {activity.timestamp:.1f}s "
                   f"(confidence: {activity.confidence:.2f})")
        
        # Save visualization
        save_activity_visualization(output_dir, activity, i)
        
        # Get transcript context if available
        if vtt_path:
            transcript_segment = get_text_at_timestamp(vtt_path, activity.timestamp)
            if transcript_segment:
                logger.info(f"Transcript context: [{transcript_segment.speaker}] {transcript_segment.text}")
        
        activities.append(activity)
        activity_types.add(activity.activity_type)
    
    # Write detailed report
    with open(output_dir / "activity_analysis.txt", "w") as f:
        f.write("Zoom Call Activity Analysis\n")
        f.write("=========================\n\n")
        
        # Overall statistics
        f.write(f"Total activities detected: {len(activities)}\n")
        f.write(f"Activity types found: {', '.join(sorted(activity_types))}\n\n")
        
        # Activity type breakdown
        f.write("Activity Type Breakdown:\n")
        for activity_type in sorted(activity_types):
            count = sum(1 for a in activities if a.activity_type == activity_type)
            f.write(f"  {activity_type}: {count} occurrences\n")
        f.write("\n")
        
        # Detailed activity log
        f.write("Activity Log:\n")
        f.write("-------------\n\n")
        
        for i, activity in enumerate(activities):
            f.write(f"\nActivity {i+1}:\n")
            f.write(f"  Time: {activity.timestamp:.1f}s\n")
            f.write(f"  Type: {activity.activity_type}\n")
            f.write(f"  Confidence: {activity.confidence:.2f}\n")
            
            # Add activity-specific details
            f.write("  Details:\n")
            for key, value in activity.details.items():
                f.write(f"    {key}: {value:.3f}\n")
            
            # Add transcript context if available
            if vtt_path:
                transcript = get_text_at_timestamp(vtt_path, activity.timestamp)
                if transcript:
                    f.write(f"  Transcript: [{transcript.speaker}] {transcript.text}\n")
            
            f.write(f"  Files:\n")
            f.write(f"    - activity_{i:04d}_before.jpg\n")
            f.write(f"    - activity_{i:04d}_after.jpg\n")
            f.write(f"    - activity_{i:04d}_visualization.jpg\n")
    
    logger.info(f"\nAnalysis complete. Detected {len(activities)} activities of {len(activity_types)} different types.")
    logger.info(f"Results saved to {output_dir}")

def main():
    # Process test call recording
    video_path = "test_call/GMT20250215-204908_Recording_1920x1108.mp4"
    output_dir = Path("detected_activities_no_transcript")
    
    # Run analysis without transcript
    analyze_video(video_path, output_dir)

if __name__ == "__main__":
    main() 