"""
Script to analyze properties of a Zoom recording and its transcript.
"""

import cv2
import numpy as np
from pathlib import Path
from callanalyser.video.processor import VideoProcessor
from callanalyser.utils.transcript import parse_vtt, TranscriptSegment

def analyze_video_stream(video_path: Path):
    """Analyze properties of a video stream."""
    print(f"\nAnalyzing video: {video_path.name}")
    print("-" * 50)
    
    with VideoProcessor(str(video_path)) as processor:
        print(f"Resolution: {processor.frame_width}x{processor.frame_height}")
        print(f"Duration: {processor.duration:.2f} seconds ({processor.duration/60:.1f} minutes)")
        print(f"FPS: {processor.fps}")
        print(f"Total frames: {processor.frame_count}")
        
        # Analyze frame content at different points
        test_times = [0, 30, 60]  # Check at 0s, 30s, and 60s
        for t in test_times:
            frame = processor.get_frame_at_timestamp(t)
            if frame is not None:
                # Calculate average brightness
                brightness = np.mean(frame)
                # Calculate standard deviation as a measure of variation/detail
                variation = np.std(frame)
                print(f"\nFrame at {t}s:")
                print(f"  Average brightness: {brightness:.1f}")
                print(f"  Variation: {variation:.1f}")

def analyze_transcript(vtt_path: Path):
    """Analyze properties of the transcript."""
    print(f"\nAnalyzing transcript: {vtt_path.name}")
    print("-" * 50)
    
    segments = list(parse_vtt(vtt_path))
    
    # Basic statistics
    total_duration = segments[-1].end_time - segments[0].start_time
    total_segments = len(segments)
    segments_with_speaker = len([s for s in segments if s.speaker is not None])
    
    print(f"Total segments: {total_segments}")
    print(f"Duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    print(f"Average segment length: {total_duration/total_segments:.1f} seconds")
    print(f"Segments with speaker info: {segments_with_speaker} ({segments_with_speaker/total_segments:.1%})")
    
    # Analyze gaps between segments
    gaps = []
    for i in range(len(segments) - 1):
        gap = segments[i + 1].start_time - segments[i].end_time
        if gap > 0:
            gaps.append(gap)
    
    if gaps:
        print(f"\nGaps between segments:")
        print(f"  Minimum gap: {min(gaps):.3f}s")
        print(f"  Maximum gap: {max(gaps):.3f}s")
        print(f"  Average gap: {sum(gaps)/len(gaps):.3f}s")
        print(f"  Number of gaps > 1s: {sum(1 for g in gaps if g > 1.0)}")
    
    # Sample some text
    print("\nSample segments:")
    for segment in segments[:3]:  # First 3 segments
        speaker = f"[{segment.speaker}] " if segment.speaker else ""
        print(f"{segment.start_time:.1f}s - {segment.end_time:.1f}s: {speaker}{segment.text[:100]}...")

def main():
    # Analyze main video recording
    video_path = Path("calls/GMT20241125-094826_Recording_1920x1080.mp4")
    analyze_video_stream(video_path)
    
    # Analyze transcript
    vtt_path = Path("calls/GMT20241125-094826_Recording.transcript.vtt")
    analyze_transcript(vtt_path)

if __name__ == "__main__":
    main() 