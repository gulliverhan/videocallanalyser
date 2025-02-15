"""
Utility module for parsing VTT (WebVTT) transcript files.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Generator
import re
import logging

logger = logging.getLogger(__name__)

@dataclass
class TranscriptSegment:
    """Represents a single segment in the transcript."""
    start_time: float  # in seconds
    end_time: float    # in seconds
    text: str
    speaker: Optional[str] = None

def parse_timestamp(timestamp: str) -> float:
    """
    Convert WebVTT timestamp to seconds.
    
    Args:
        timestamp (str): Timestamp in format "HH:MM:SS.mmm"
        
    Returns:
        float: Time in seconds
    """
    # Handle optional hours
    if '.' not in timestamp:
        timestamp += '.000'
    
    parts = timestamp.split(':')
    if len(parts) == 2:
        parts.insert(0, '00')  # Add hours if missing
    
    hours, minutes, seconds = parts
    seconds, milliseconds = seconds.split('.')
    
    total_seconds = (
        int(hours) * 3600 +
        int(minutes) * 60 +
        int(seconds) +
        int(milliseconds.ljust(3, '0')) / 1000
    )
    
    return total_seconds

def parse_vtt(vtt_path: str) -> Generator[TranscriptSegment, None, None]:
    """
    Parse a WebVTT file and yield TranscriptSegments.
    
    Args:
        vtt_path (str): Path to the VTT file
        
    Yields:
        TranscriptSegment: Parsed transcript segments
    """
    vtt_path = Path(vtt_path)
    if not vtt_path.exists():
        raise FileNotFoundError(f"VTT file not found: {vtt_path}")
    
    with open(vtt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split into segments (split on double newline)
    segments = content.strip().split('\n\n')
    
    # Skip WebVTT header
    if segments[0].startswith('WEBVTT'):
        segments = segments[1:]
    
    # Regular expression for timestamp line
    timestamp_pattern = r'(\d{2}:)?(\d{2}:\d{2}\.\d{3}) --> (\d{2}:)?(\d{2}:\d{2}\.\d{3})'
    
    for segment in segments:
        lines = segment.strip().split('\n')
        if len(lines) < 2:
            continue
        
        # Find timestamp line
        timestamp_line = None
        for line in lines:
            if re.match(timestamp_pattern, line):
                timestamp_line = line
                break
        
        if not timestamp_line:
            continue
        
        # Parse timestamps
        match = re.match(timestamp_pattern, timestamp_line)
        if not match:
            continue
        
        start_time = parse_timestamp(match.group(2) if match.group(1) is None else match.group(1) + match.group(2))
        end_time = parse_timestamp(match.group(4) if match.group(3) is None else match.group(3) + match.group(4))
        
        # Get text content (everything after timestamp line)
        text_lines = lines[lines.index(timestamp_line) + 1:]
        text = ' '.join(text_lines)
        
        # Try to extract speaker if present (usually in format "Speaker: Text")
        speaker = None
        if ': ' in text:
            speaker_part, text = text.split(': ', 1)
            if not any(char.isdigit() for char in speaker_part):  # Avoid splitting on timestamps
                speaker = speaker_part
        
        yield TranscriptSegment(
            start_time=start_time,
            end_time=end_time,
            text=text,
            speaker=speaker
        )

def get_text_at_timestamp(vtt_path: str, timestamp: float) -> Optional[TranscriptSegment]:
    """
    Get the transcript segment at a specific timestamp.
    
    Args:
        vtt_path (str): Path to the VTT file
        timestamp (float): Time in seconds
        
    Returns:
        Optional[TranscriptSegment]: Matching transcript segment if found
    """
    for segment in parse_vtt(vtt_path):
        if segment.start_time <= timestamp <= segment.end_time:
            return segment
    return None 