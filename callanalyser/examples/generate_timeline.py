"""
Script to generate a visual timeline of Zoom call activities.
"""

import cv2
import numpy as np
from pathlib import Path
import logging
from callanalyser.video.activity_detector import ActivityDetector, ContentType, ActivityEvent
from callanalyser.video.metric_analyzer import analyze_video_metrics
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import argparse
import inquirer
from typing import Optional, List
import os

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

def interactive_mode() -> dict:
    """
    Run interactive mode to get parameters from user.
    
    Returns:
        dict: Parameters selected by user
    """
    questions = [
        inquirer.Path('video_path',
            message="Enter the path to your video file",
            exists=True,
            path_type=inquirer.Path.FILE),
        inquirer.List('llm_enabled',
            message="Would you like to use LLM verification?",
            choices=['No', 'Yes'],
            default='No'),
        inquirer.List('llm_provider',
            message="Select LLM provider",
            choices=['anthropic', 'openai', 'azure', 'gemini'],
            default='anthropic',
            when=lambda answers: answers['llm_enabled'] == 'Yes'),
        inquirer.Text('sample_interval',
            message="Enter sample interval in seconds (default: 10.0)",
            default="10.0",
            validate=lambda _, x: float(x) > 0),
        inquirer.Text('min_confidence',
            message="Enter minimum confidence threshold (0.0-1.0, default: 0.3)",
            default="0.3",
            validate=lambda _, x: 0 <= float(x) <= 1),
        inquirer.Text('min_duration',
            message="Enter minimum segment duration in seconds (default: 2.0)",
            default="2.0",
            validate=lambda _, x: float(x) > 0),
        inquirer.Path('output_dir',
            message="Enter output directory (default: timeline_output)",
            default="timeline_output",
            exists=False,
            path_type=inquirer.Path.DIRECTORY)
    ]
    
    answers = inquirer.prompt(questions)
    if not answers:
        sys.exit(1)
    
    # Convert answers to appropriate types
    return {
        'video_path': answers['video_path'],
        'use_llm': answers['llm_enabled'] == 'Yes',
        'llm_provider': answers.get('llm_provider'),
        'sample_interval': float(answers['sample_interval']),
        'min_confidence': float(answers['min_confidence']),
        'min_duration': float(answers['min_duration']),
        'output_dir': answers['output_dir']
    }

def create_timeline(activities: list, duration: float, output_path: Path):
    """
    Create a visual timeline of the call activities.
    
    Args:
        activities: List of ActivityEvent objects
        duration: Total duration of the video in seconds
        output_path: Path to save the timeline image
    """
    # Set up the plot
    plt.figure(figsize=(15, 6))
    
    # Define colors for different content types
    colors = {
        ContentType.SPEAKER: '#2ecc71',     # Green
        ContentType.SLIDES: '#9b59b6',      # Purple
        ContentType.INTERACTIVE: '#3498db',  # Blue
    }
    
    # Create main timeline axis
    ax = plt.gca()
    ax.set_xlim(0, duration)
    ax.set_ylim(0, 1)
    
    # Remove y-axis ticks and labels
    ax.set_yticks([])
    
    # Add grid for time markers
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)
    
    # Plot each activity segment
    for activity in activities:
        # Skip invalid segments
        if activity.start_time >= duration or activity.end_time is None:
            continue
            
        # Ensure we don't exceed video duration
        end_time = min(activity.end_time, duration)
        width = end_time - activity.start_time
        
        if width <= 0:
            continue
        
        # Draw the segment
        rect = patches.Rectangle(
            (activity.start_time, 0), width, 1,
            facecolor=colors.get(activity.content_type, colors[ContentType.SLIDES]),
            alpha=0.5
        )
        ax.add_patch(rect)
        
        # Add label if segment is wide enough
        if width > duration * 0.02:  # Only label segments wider than 2% of total duration
            label = activity.content_type.replace('_', ' ').title()
            if activity.content_type == ContentType.SLIDES and activity.slide_number is not None:
                label = f"Slide {activity.slide_number}"
            
            plt.text(
                activity.start_time + width/2,
                0.5,
                label,
                horizontalalignment='center',
                verticalalignment='center'
            )
    
    # Customize x-axis
    plt.xlabel("Time (MM:SS)")
    ax.set_xticks(np.arange(0, duration, 30))  # Tick every 30 seconds
    ax.set_xticklabels([format_timestamp(t) for t in np.arange(0, duration, 30)])
    
    # Add title and legend
    plt.title("Zoom Call Timeline")
    
    # Create legend
    legend_elements = [
        patches.Patch(facecolor=colors[ct], alpha=0.5, label=ct.replace('_', ' ').title())
        for ct in [ContentType.SPEAKER, ContentType.SLIDES, ContentType.INTERACTIVE]
    ]
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15),
              ncol=len(legend_elements), frameon=False)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def create_summary(video_name: str, activities: List[ActivityEvent], summary_path: str):
    """Create a text summary of the activities."""
    with open(summary_path, "w") as f:
        f.write(f"Call Timeline Summary - {video_name}\n")
        f.write("=" * (22 + len(video_name)) + "\n\n")
        
        for activity in activities:
            # Format timestamps as MM:SS
            start_time = format_timestamp(activity.start_time)
            end_time = format_timestamp(activity.end_time) if activity.end_time else "END"
            duration = activity.end_time - activity.start_time if activity.end_time else 0
            
            # Add slide number to description if available
            description = activity.content_type.replace('_', ' ').title()
            if activity.content_type == ContentType.SLIDES:
                if activity.slide_number is not None:
                    description = f"Slide {int(activity.slide_number)}"
                else:
                    description = "Slides"  # Default to "Slides" if no number available
            
            f.write(f"{start_time} - {end_time}: {description} "
                   f"({duration:.1f}s, confidence: {activity.confidence:.2f})\n")
            
            # Add key metrics
            f.write("  Key metrics:\n")
            for metric, value in activity.metrics.items():
                if metric == 'frame':  # Skip the frame data
                    continue
                # Convert numpy values to Python float
                if hasattr(value, 'dtype'):  # Check if it's a numpy type
                    value = float(value)
                if metric == 'slide_number' and value is not None:
                    value = int(value)  # Ensure slide numbers are integers
                try:
                    f.write(f"    {metric}: {value:.3f}\n")
                except (TypeError, ValueError):
                    # For non-numeric values, just print as is
                    f.write(f"    {metric}: {value}\n")
            f.write("\n")
    
    logging.info(f"Summary saved to {summary_path}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Generate activity timeline from video')
    parser.add_argument('video_path', help='Path to video file')
    parser.add_argument('--sample-interval', type=float, default=1.0,
                      help='Time interval between frame samples in seconds')
    parser.add_argument('--min-confidence', type=float, default=0.3,
                      help='Minimum confidence threshold for activity detection')
    parser.add_argument('--use-llm', action='store_true',
                      help='Use LLM for content verification')
    parser.add_argument('--llm-provider', choices=['openai', 'anthropic'], default='anthropic',
                      help='LLM provider to use for verification')
    args = parser.parse_args()

    # Convert video path to absolute path
    video_path = os.path.abspath(args.video_path)
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    # Create output directory if needed
    os.makedirs('timeline_output', exist_ok=True)

    # Get video filename without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Setup paths
    timeline_path = f'timeline_output/timeline_{video_name}.png'
    summary_path = f'timeline_output/summary_{video_name}.txt'

    # Initialize detector
    detector = ActivityDetector(
        use_llm=args.use_llm,
        llm_provider=args.llm_provider
    )

    # Detect activities
    logging.info(f"Analyzing video {os.path.basename(video_path)}...")
    logging.info(f"LLM verification: {'enabled' if args.use_llm else 'disabled'}")

    activities = detector.detect_activities(video_path, sample_interval=args.sample_interval)

    # Filter activities by confidence
    activities = [a for a in activities if a.confidence >= args.min_confidence]

    # Get video duration
    cap = cv2.VideoCapture(video_path)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # Create timeline visualization
    create_timeline(activities, duration, timeline_path)

    # Generate text summary
    create_summary(video_name, activities, summary_path)

if __name__ == "__main__":
    main() 