from typing import List, Dict, Any
from ..parsers.base import Document
import os


class AudioProcessor:
    """Audio processor with transcription capabilities using Whisper."""
    
    def __init__(self, model_size: str = "base"):
        """Initialize the audio processor.
        
        Args:
            model_size (str): Size of Whisper model to use ("tiny", "base", "small", "medium", "large")
        """
        self.model_size = model_size
        self.model = None
        self.processor = None
        
    def _load_model(self):
        """Load the Whisper model."""
        if self.model is not None:
            return
            
        try:
            import whisper
            self.model = whisper.load_model(self.model_size)
        except ImportError:
            raise ImportError(
                "To use audio transcription, you need to install OpenAI Whisper. "
                "Please run: pip install openai-whisper"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper model: {e}")
    
    def process_audio(self, audio_path: str) -> Document:
        """Process an audio file and generate transcription.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            Document: Document containing audio transcription and metadata
        """
        # Load model
        self._load_model()
        
        # Transcribe audio
        result = self.model.transcribe(audio_path)
        
        # Extract transcription and additional info
        transcription = result["text"]
        language = result.get("language", "unknown")
        segments = result.get("segments", [])
        
        # Create document
        metadata = {
            "source": audio_path,
            "content_type": "audio",
            "media_type": "audio",
            "language": language,
            "model_size": self.model_size,
            "segment_count": len(segments)
        }
        
        # Add segment information if available
        if segments:
            metadata["segments"] = [
                {
                    "start": seg.get("start", 0),
                    "end": seg.get("end", 0),
                    "text": seg.get("text", "")
                }
                for seg in segments[:5]  # Limit to first 5 segments for metadata
            ]
        
        return Document(content=transcription.strip(), metadata=metadata)
    
    def process_video(self, video_path: str) -> Document:
        """Process a video file by extracting audio and generating transcription.
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            Document: Document containing video transcription and metadata
        """
        try:
            import whisper
        except ImportError:
            raise ImportError(
                "To process video files, you need to install OpenAI Whisper. "
                "Please run: pip install openai-whisper"
            )
        
        # For now, we'll just transcribe the video directly
        # In a more advanced implementation, we might extract audio first
        return self.process_audio(video_path)
