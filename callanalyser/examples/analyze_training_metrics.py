"""
Script to analyze metrics from the training video and compare with ground truth.
"""

import argparse
from pathlib import Path
import logging
from callanalyser.video.metric_analyzer import analyze_training_videos
import json
import numpy as np
import sys
import inquirer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        inquirer.Path('truth_path',
            message="Enter the path to ground truth file (leave empty to auto-detect)",
            exists=False,
            path_type=inquirer.Path.FILE),
        inquirer.Text('sample_interval',
            message="Enter sample interval in seconds (default: 10.0)",
            default="10.0",
            validate=lambda _, x: float(x) > 0),
        inquirer.Text('min_duration',
            message="Enter minimum segment duration in seconds (default: 2.0)",
            default="2.0",
            validate=lambda _, x: float(x) > 0),
        inquirer.Path('output_dir',
            message="Enter output directory (default: metric_analysis)",
            default="metric_analysis",
            exists=False,
            path_type=inquirer.Path.DIRECTORY),
        inquirer.List('detail_level',
            message="Select analysis detail level",
            choices=['basic', 'detailed', 'full'],
            default='detailed')
    ]
    
    answers = inquirer.prompt(questions)
    if not answers:
        sys.exit(1)
    
    # If truth path not provided, try to auto-detect
    if not answers['truth_path']:
        video_path = Path(answers['video_path'])
        default_truth = video_path.parent / f"{video_path.stem}.txt"
        if default_truth.exists():
            answers['truth_path'] = str(default_truth)
        else:
            logger.error(f"Ground truth file not found at {default_truth}")
            sys.exit(1)
    
    return {
        'video_path': answers['video_path'],
        'truth_path': answers['truth_path'],
        'sample_interval': float(answers['sample_interval']),
        'min_duration': float(answers['min_duration']),
        'output_dir': answers['output_dir'],
        'detail_level': answers['detail_level']
    }

def print_analysis_results(metrics_file: Path, detail_level: str = 'detailed'):
    """
    Print a human-readable analysis of the metrics.
    
    Args:
        metrics_file: Path to the metrics JSON file
        detail_level: Level of detail to show ('basic', 'detailed', or 'full')
    """
    with open(metrics_file, 'r') as f:
        results = json.load(f)
    
    # Get the first (and only) video's data
    video_path = next(iter(results))
    video_data = results[video_path]
    metrics = video_data.get('metrics', [])
    
    print("\nVideo Analysis")
    print("=============")
    print(f"Total samples: {len(metrics)} (10-second intervals)")
    
    # Analyze content type distribution
    content_types = {}
    for metric in metrics:
        content = metric['ground_truth']
        content_types[content] = content_types.get(content, 0) + 1
    
    print("\nContent Type Distribution:")
    for content_type, count in content_types.items():
        percentage = (count / len(metrics)) * 100
        duration = count * 10  # Each sample represents 10 seconds
        minutes = duration // 60
        seconds = duration % 60
        print(f"{content_type:15s}: {count:4d} samples ({percentage:5.1f}%) - {minutes:.0f}m {seconds:.0f}s total")
    
    if detail_level == 'basic':
        return
    
    # Analyze metrics by content type
    print("\nDetailed Metrics by Content Type:")
    print("=" * 50)
    
    for content_type in content_types:
        # Get all metrics for this content type
        type_metrics = [m['metrics'] for m in metrics if m['ground_truth'] == content_type]
        
        print(f"\n{content_type} ({len(type_metrics)} samples):")
        
        # Key metrics to analyze
        key_metrics = {
            'face_count': 'Face detection',
            'edge_density': 'Edge density',
            'text_line_density': 'Text density',
            'motion_magnitude': 'Motion magnitude',
            'brightness': 'Brightness'
        }
        
        for metric_key, metric_name in key_metrics.items():
            values = [float(m[metric_key]) for m in type_metrics]
            mean = np.mean(values)
            std = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            
            print(f"  {metric_name}:")
            print(f"    Mean: {mean:.3f} ± {std:.3f}")
            print(f"    Range: {min_val:.3f} - {max_val:.3f}")
    
    if detail_level == 'detailed':
        return
        
    # Full analysis includes transitions
    print("\nContent Type Transitions:")
    print("-" * 80)
    
    prev_type = None
    transition_count = 0
    
    for i, metric in enumerate(metrics):
        current_type = metric['ground_truth']
        if prev_type and current_type != prev_type:
            transition_count += 1
            time = metric['time']
            minutes = int(time // 60)
            seconds = int(time % 60)
            
            # Get metrics before and after transition
            before_metrics = metrics[i-1]['metrics']
            after_metrics = metric['metrics']
            
            print(f"\nTransition {transition_count} at {minutes:02d}:{seconds:02d}")
            print(f"From: {prev_type} -> To: {current_type}")
            print("Before metrics:")
            print(f"  Face count: {before_metrics['face_count']}")
            print(f"  Edge density: {float(before_metrics['edge_density']):.3f}")
            print(f"  Text density: {float(before_metrics['text_line_density']):.3f}")
            print(f"  Motion magnitude: {float(before_metrics['motion_magnitude']):.3f}")
            print("After metrics:")
            print(f"  Face count: {after_metrics['face_count']}")
            print(f"  Edge density: {float(after_metrics['edge_density']):.3f}")
            print(f"  Text density: {float(after_metrics['text_line_density']):.3f}")
            print(f"  Motion magnitude: {float(after_metrics['motion_magnitude']):.3f}")
        
        prev_type = current_type

def main():
    parser = argparse.ArgumentParser(description='Analyze training video metrics.')
    parser.add_argument('video_path', nargs='?', help='Path to the video file')
    parser.add_argument('--interactive', '-i', action='store_true', help='Run in interactive mode')
    parser.add_argument('--truth-path', help='Path to ground truth file (defaults to video_name.txt)')
    parser.add_argument('--sample-interval', type=float, default=10.0,
                       help='Interval between frame samples in seconds (default: 10.0)')
    parser.add_argument('--min-duration', type=float, default=2.0,
                       help='Minimum duration for content segments in seconds (default: 2.0)')
    parser.add_argument('--output-dir', default='metric_analysis',
                       help='Directory to save metric analysis (default: metric_analysis)')
    parser.add_argument('--detail-level', choices=['basic', 'detailed', 'full'],
                       default='detailed', help='Level of detail in analysis output')
    
    args = parser.parse_args()
    
    # If interactive mode or no video path provided, run interactive mode
    if args.interactive or not args.video_path:
        params = interactive_mode()
    else:
        video_path = Path(args.video_path)
        # Try to find truth file if not specified
        truth_path = args.truth_path
        if not truth_path:
            default_truth = video_path.parent / f"{video_path.stem}.txt"
            if default_truth.exists():
                truth_path = str(default_truth)
            else:
                logger.error(f"Ground truth file not found at {default_truth}")
                return
        
        params = {
            'video_path': str(video_path),
            'truth_path': truth_path,
            'sample_interval': args.sample_interval,
            'min_duration': args.min_duration,
            'output_dir': args.output_dir,
            'detail_level': args.detail_level
        }
    
    # Create output directory
    output_dir = Path(params['output_dir'])
    output_dir.mkdir(exist_ok=True)
    
    logger.info("Analyzing video...")
    logger.info(f"Video: {params['video_path']}")
    logger.info(f"Truth: {params['truth_path']}")
    logger.info(f"Sample interval: {params['sample_interval']}s")
    logger.info(f"Minimum duration: {params['min_duration']}s")
    logger.info(f"Detail level: {params['detail_level']}")
    
    # Analyze video
    analyze_training_videos(
        [params['video_path']], 
        [params['truth_path']], 
        output_dir,
        sample_interval=params['sample_interval'],
        min_duration=params['min_duration']
    )
    
    # Print detailed analysis
    metrics_file = output_dir / 'metrics.json'
    if metrics_file.exists():
        print_analysis_results(metrics_file, params['detail_level'])
    
    logger.info(f"\nAnalysis complete. Full results saved to {output_dir}")

if __name__ == "__main__":
    main() 