"""
Script to generate a visual timeline of Zoom call activities.
"""

import cv2
import numpy as np
from pathlib import Path
import logging
from callanalyser.video.activity_detector import ActivityDetector, ContentType, ActivityChange
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import timedelta
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS."""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def normalize_timestamp(timestamp: float, last_timestamp: float, duration: float) -> float:
    """
    Normalize timestamp to handle wrapping around video length.
    
    Args:
        timestamp: Current timestamp
        last_timestamp: Previous timestamp
        duration: Video duration
        
    Returns:
        float: Normalized timestamp
    """
    # If timestamp is less than last_timestamp, it might have wrapped
    if timestamp < last_timestamp:
        # Check if adding duration makes more sense
        if abs(timestamp + duration - last_timestamp) < abs(timestamp - last_timestamp):
            return timestamp + duration
    return timestamp

def create_timeline(activities: list, duration: float, output_path: Path):
    """
    Create a visual timeline of the call activities.
    
    Args:
        activities: List of detected activities
        duration: Total duration of the video in seconds
        output_path: Path to save the timeline image
    """
    # Set up the plot
    plt.figure(figsize=(15, 6))
    
    # Define colors for different content types
    colors = {
        ContentType.SPEAKER_VIEW: '#2ecc71',  # Green
        ContentType.SCREEN_SHARE: '#3498db',  # Blue
        ContentType.SLIDES: '#9b59b6',        # Purple
        ContentType.WHITEBOARD: '#e74c3c',    # Red
        ContentType.UNKNOWN: '#95a5a6'        # Gray
    }
    
    # Create main timeline axis
    ax = plt.gca()
    ax.set_xlim(0, duration)
    ax.set_ylim(0, 1)
    
    # Remove y-axis ticks and labels
    ax.set_yticks([])
    
    # Add grid for time markers
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)
    
    # Track content segments
    current_content = ContentType.SPEAKER_VIEW
    segment_start = 0
    last_timestamp = 0
    
    # Filter out very short segments and fix overlapping times
    filtered_activities = []
    min_segment_duration = 0.5  # Minimum segment duration in seconds
    
    for activity in activities:
        # Normalize timestamp
        normalized_time = normalize_timestamp(activity.timestamp, last_timestamp, duration)
        
        # Calculate segment duration
        segment_duration = normalized_time - last_timestamp
        
        # Only include segments longer than minimum duration
        if segment_duration >= min_segment_duration:
            # Create new activity with normalized timestamp
            filtered_activities.append(ActivityChange(
                timestamp=normalized_time,
                confidence=activity.confidence,
                frame_before=activity.frame_before,
                frame_after=activity.frame_after,
                activity_type=activity.activity_type,
                content_type=activity.content_type,
                details=activity.details
            ))
            last_timestamp = normalized_time
    
    # Plot content segments
    for i, activity in enumerate(filtered_activities):
        if activity.activity_type == "content_change" or i == len(filtered_activities) - 1:
            # Draw the segment up to this point
            width = activity.timestamp - segment_start
            
            # Skip invalid segments
            if width <= 0 or segment_start >= duration:
                continue
                
            # Ensure we don't exceed video duration
            if segment_start + width > duration:
                width = duration - segment_start
            
            rect = patches.Rectangle(
                (segment_start, 0), width, 1,
                facecolor=colors[current_content],
                alpha=0.5
            )
            ax.add_patch(rect)
            
            # Add label if segment is wide enough
            if width > duration * 0.05:  # Only label segments wider than 5% of total duration
                plt.text(
                    segment_start + width/2,
                    0.5,
                    current_content.replace('_', ' ').title(),
                    horizontalalignment='center',
                    verticalalignment='center'
                )
            
            # Start new segment
            segment_start = activity.timestamp
            current_content = activity.content_type
    
    # Draw final segment if needed
    if segment_start < duration:
        width = duration - segment_start
        rect = patches.Rectangle(
            (segment_start, 0), width, 1,
            facecolor=colors[current_content],
            alpha=0.5
        )
        ax.add_patch(rect)
        if width > duration * 0.05:
            plt.text(
                segment_start + width/2,
                0.5,
                current_content.replace('_', ' ').title(),
                horizontalalignment='center',
                verticalalignment='center'
            )
    
    # Add activity markers
    for activity in filtered_activities:
        if activity.activity_type in ["speaker_change", "content_update"]:
            plt.axvline(x=activity.timestamp, color='black', alpha=0.2, linewidth=1)
    
    # Customize x-axis
    plt.xlabel("Time (MM:SS)")
    ax.set_xticks(np.arange(0, duration, 30))  # Tick every 30 seconds
    ax.set_xticklabels([format_timestamp(t) for t in np.arange(0, duration, 30)])
    
    # Add title and legend
    plt.title("Zoom Call Timeline")
    
    # Create legend
    legend_elements = [
        patches.Patch(facecolor=colors[ct], alpha=0.5, label=ct.replace('_', ' ').title())
        for ct in colors if ct != ContentType.UNKNOWN
    ]
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15),
              ncol=len(legend_elements), frameon=False)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def main():
    # Get video path from command line argument
    if len(sys.argv) != 2:
        print("Usage: python generate_timeline.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    video_name = Path(video_path).stem
    
    output_dir = Path("timeline_output")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize detector with adjusted parameters
    detector = ActivityDetector(
        min_change_threshold=10.0,
        window_size=2,
        low_res_width=320,
        low_res_height=180
    )
    
    logger.info(f"Analyzing video {video_name}...")
    
    # Get video duration
    cap = cv2.VideoCapture(video_path)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    # Collect activities
    activities = list(detector.detect_activities(video_path, min_confidence=0.3))
    
    # Generate timeline with unique filename
    timeline_path = output_dir / f"timeline_{video_name}.png"
    create_timeline(activities, duration, timeline_path)
    
    logger.info(f"Timeline generated and saved to {timeline_path}")
    
    # Generate text summary with unique filename
    summary_path = output_dir / f"summary_{video_name}.txt"
    with open(summary_path, "w") as f:
        f.write(f"Call Timeline Summary - {video_name}\n")
        f.write("=" * (22 + len(video_name)) + "\n\n")
        
        current_content = ContentType.SPEAKER_VIEW
        segment_start = 0
        last_timestamp = 0
        
        for activity in activities:
            # Normalize timestamp
            normalized_time = normalize_timestamp(activity.timestamp, last_timestamp, duration)
            
            if activity.activity_type == "content_change":
                # Calculate segment duration
                segment_duration = normalized_time - segment_start
                
                # Only write segments longer than 0.5 seconds
                if segment_duration >= 0.5:
                    f.write(f"{format_timestamp(segment_start)} - {format_timestamp(normalized_time)}: "
                           f"{current_content.replace('_', ' ').title()} ({segment_duration:.1f}s)\n")
                    
                    # Update for next segment
                    segment_start = normalized_time
                    current_content = activity.content_type
                    last_timestamp = normalized_time
        
        # Write final segment if needed
        if segment_start < duration:
            final_duration = duration - segment_start
            if final_duration > 0:
                f.write(f"{format_timestamp(segment_start)} - {format_timestamp(duration)}: "
                       f"{current_content.replace('_', ' ').title()} ({final_duration:.1f}s)\n")
    
    logger.info(f"Summary saved to {summary_path}")

if __name__ == "__main__":
    main() 