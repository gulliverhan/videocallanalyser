"""
LLM API module for video content analysis.
Supports multiple LLM providers for frame analysis and content type verification.
"""

import os
from pathlib import Path
import sys
import base64
from typing import Optional, Union, List, Tuple
import mimetypes
import logging
from dotenv import load_dotenv

# Import LLM providers
from openai import OpenAI, AzureOpenAI
from anthropic import Anthropic
import google.generativeai as genai

logger = logging.getLogger(__name__)

def load_environment():
    """Load environment variables from .env files in order of precedence"""
    # Get the project root directory (where the .env file should be)
    project_root = Path(__file__).parent.parent.parent
    logger.info(f"Project root directory: {project_root.absolute()}")
    
    env_files = ['.env.local', '.env', '.env.example']
    env_loaded = False
    
    for env_file in env_files:
        env_path = project_root / env_file
        logger.info(f"Checking for {env_file} at: {env_path.absolute()}")
        if env_path.exists():
            logger.info(f"Loading environment from: {env_path.absolute()}")
            load_dotenv(dotenv_path=env_path, override=True)
            # Print loaded environment variables (excluding sensitive values)
            loaded_vars = {}
            with open(env_path) as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key = line.split('=')[0].strip()
                        loaded_vars[key] = '[HIDDEN]' if 'KEY' in key else os.getenv(key)
            logger.info(f"Loaded variables: {loaded_vars}")
            env_loaded = True
    
    if not env_loaded:
        logger.warning("No .env files found!")
    
    # Verify the loaded OpenAI key
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        logger.info(f"Loaded OpenAI key starting with: {openai_key[:8]}...")
        logger.info(f"Key length: {len(openai_key)}")
    else:
        logger.warning("No OpenAI API key found in environment!")

# Load environment variables at module import
load_environment()

def create_llm_client(provider: str = "anthropic"):
    """
    Create an LLM client for the specified provider.
    
    Args:
        provider (str): The LLM provider to use
        
    Returns:
        The LLM client instance
    """
    if provider == "anthropic":
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        return Anthropic(api_key=api_key)
    
    elif provider == "openai":
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        return OpenAI(api_key=api_key)
    
    elif provider == "azure":
        api_key = os.getenv('AZURE_OPENAI_API_KEY')
        if not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY not found in environment variables")
        return AzureOpenAI(
            api_key=api_key,
            api_version="2024-02-15-preview",
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT')
        )
    
    elif provider == "gemini":
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        genai.configure(api_key=api_key)
        return genai
    
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def encode_image(image_data: bytes, mime_type: str = "image/jpeg") -> str:
    """
    Encode image data to base64.
    
    Args:
        image_data (bytes): Raw image data
        mime_type (str): MIME type of the image
        
    Returns:
        str: Base64 encoded image string
    """
    return base64.b64encode(image_data).decode('utf-8')

def analyze_frames(
    frames: List[Tuple[str, str, str]],  # List of (frame_data, current_type, proposed_type)
    provider: str = None,
    model: Optional[str] = None,
    temperature: float = 0.3
) -> List[Tuple[str, float]]:
    """
    Analyze video frames using the specified LLM provider.
    
    Args:
        frames: List of (frame_data, current_type, proposed_type) tuples
        provider: LLM provider to use (defaults to LLM_PROVIDER from env)
        model: Model name (defaults to LLM_MODEL from env)
        temperature: Temperature for response generation
        
    Returns:
        List[Tuple[str, float]]: List of (content_type, confidence) tuples
    """
    try:
        # Use environment variables if not specified
        provider = provider or os.getenv('LLM_PROVIDER', 'anthropic')
        model = model or os.getenv('LLM_MODEL', 'claude-3-sonnet-20240229')
        
        logger.info(f"Using LLM provider: {provider}, model: {model}")
        client = create_llm_client(provider)
        
        # Prepare images
        encoded_frames = [encode_image(frame_data) for frame_data, _, _ in frames]
        
        # Load prompt template
        prompt_path = Path(__file__).parent.parent / "prompts" / "content_analysis.txt"
        if not prompt_path.exists():
            logger.warning(f"Prompt file not found at {prompt_path}, using default prompt")
            with open(prompt_path, 'r') as f:
                prompt = f.read()
        
        # Add frame-specific information
        for i, (_, current_type, proposed_type) in enumerate(frames):
            prompt += f"\nFrame {i+1}:\n"
            prompt += f"Current classification: {current_type}\n"
            prompt += f"Proposed classification: {proposed_type}\n"
        
        # Call appropriate provider
        if provider == "anthropic":
            messages = [{"role": "user", "content": []}]
            messages[0]["content"].append({"type": "text", "text": prompt})
            
            for frame in encoded_frames:
                messages[0]["content"].append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": frame
                    }
                })
            
            response = client.messages.create(
                model=model,
                max_tokens=1000,
                messages=messages,
                temperature=temperature
            )
            response_text = response.content[0].text
            
        elif provider in ["openai", "azure"]:
            messages = [{"role": "user", "content": []}]
            messages[0]["content"].append({"type": "text", "text": prompt})
            
            for frame in encoded_frames:
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{frame}"}
                })
            
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=1000
            )
            response_text = response.choices[0].message.content
            
        elif provider == "gemini":
            model = client.GenerativeModel(model)
            response = model.generate_content([prompt] + encoded_frames)
            response_text = response.text
        
        # Parse response
        results = []
        frame_responses = response_text.lower().split("frame")[1:]
        
        for resp in frame_responses:
            content_type = "slides"  # Default to slides
            confidence = 0.5
            
            # Look for content type line
            for line in resp.split('\n'):
                if "content type:" in line:
                    type_text = line.split("content type:")[1].strip()
                    if "speaker" in type_text:
                        content_type = "speaker"
                    elif "screen sharing" in type_text or "screenshare" in type_text:
                        content_type = "screen sharing"
                    # Note: If no clear match, keep the default "slides"
                elif "confidence:" in line:
                    try:
                        confidence = float(line.split("confidence:")[1].strip())
                    except ValueError:
                        confidence = 0.5
            
            results.append((content_type, confidence))
        
        return results
        
    except Exception as e:
        logger.error(f"Error during LLM analysis: {str(e)}")
        # Return conservative results on error (default to slides)
        return [("slides", 0.5) for _ in frames]

def verify_content_type(
    frame_data: bytes,
    current_type: str,
    proposed_type: str,
    provider: str = "anthropic"
) -> Tuple[str, float]:
    """
    Verify content type for a single frame.
    Convenience wrapper around analyze_frames for single-frame analysis.
    
    Args:
        frame_data: Raw frame image data
        current_type: Current content type classification
        proposed_type: Proposed new content type
        provider: LLM provider to use
        
    Returns:
        Tuple[str, float]: Verified content type and confidence
    """
    results = analyze_frames(
        [(frame_data, current_type, proposed_type)],
        provider=provider
    )
    return results[0] if results else ("unknown", 0.5) 