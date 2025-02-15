"""
Example script demonstrating basic video frame extraction and processing.
"""

import cv2
import os
from pathlib import Path
from callanalyser.video.processor import VideoProcessor

def main():
    # Get the video path from command line or use a default
    video_path = input("Enter the path to your video file: ")
    if not os.path.exists(video_path):
        print(f"Error: File {video_path} does not exist")
        return

    # Create output directory for frames
    output_dir = Path("extracted_frames")
    output_dir.mkdir(exist_ok=True)

    # Process video and extract frames
    with VideoProcessor(video_path) as processor:
        print(f"Video duration: {processor.duration:.2f} seconds")
        print(f"FPS: {processor.fps}")
        print(f"Resolution: {processor.frame_width}x{processor.frame_height}")

        # Extract a frame every second
        print("\nExtracting frames...")
        for timestamp, frame in processor.extract_frames(interval=1.0):
            # Save original frame
            frame_path = output_dir / f"frame_{timestamp:.1f}s.jpg"
            cv2.imwrite(str(frame_path), frame)
            
            # Save a smaller version (320x180) for quick preview
            small_frame = processor.resize_frame(frame, 320, 180)
            small_frame_path = output_dir / f"frame_{timestamp:.1f}s_small.jpg"
            cv2.imwrite(str(small_frame_path), small_frame)
            
            print(f"Saved frame at {timestamp:.1f}s")

        print(f"\nFrames saved to {output_dir.absolute()}")

if __name__ == "__main__":
    main() 