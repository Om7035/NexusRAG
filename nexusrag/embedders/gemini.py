from typing import List
import os
from .base import BaseEmbedder


class GeminiEmbedder(BaseEmbedder):
    """Text embedder using Google Gemini's embedding models."""
    
    def __init__(self, model_name: str = "models/embedding-001"):
        """Initialize the Gemini embedder.
        
        Args:
            model_name (str): Name of the Gemini embedding model to use
        """
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "To use GeminiEmbedder, you need to install the google-generativeai library. "
                "Please run: pip install google-generativeai"
            )
        
        # Get API key from environment variable
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Gemini API key not found. Please set the GEMINI_API_KEY environment variable."
            )
        
        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.model = genai.embed_content
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of text strings using Gemini.
        
        Args:
            texts (List[str]): List of text strings to embed
            
        Returns:
            List[List[float]]: List of embeddings, one for each input text
        """
        embeddings = []
        
        for text in texts:
            response = self.model(
                model=self.model_name,
                content=text
            )
            embedding = response['embedding']
            embeddings.append(embedding)
            
        return embeddings
