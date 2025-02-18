"""
Script to analyze video content metrics using ground truth data.
"""

import argparse
from pathlib import Path
import logging
from callanalyser.video.metric_analyzer import analyze_video_metrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Analyze video content metrics.')
    parser.add_argument('video_path', help='Path to the video file')
    parser.add_argument('--actions', help='Path to the actions.txt file (defaults to actions.txt in video directory)')
    parser.add_argument('--output-dir', default='metric_analysis', help='Directory to save metric analysis')
    
    args = parser.parse_args()
    
    video_path = Path(args.video_path)
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        return
    
    # Use provided actions file or look in video directory
    actions_file = args.actions if args.actions else video_path.parent / "actions.txt"
    if not Path(actions_file).exists():
        logger.error(f"Actions file not found: {actions_file}")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    logger.info(f"Analyzing video: {video_path}")
    logger.info(f"Using actions file: {actions_file}")
    logger.info(f"Output directory: {output_dir}")
    
    # Analyze metrics
    metrics = analyze_video_metrics(
        str(video_path),
        str(actions_file),
        output_dir
    )
    
    logger.info(f"\nMetric analysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main() 