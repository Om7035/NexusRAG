from typing import List
import os
from .base import BaseEmbedder


class CohereEmbedder(BaseEmbedder):
    """Text embedder using Cohere's embedding models."""
    
    def __init__(self, model_name: str = "embed-english-v3.0"):
        """Initialize the Cohere embedder.
        
        Args:
            model_name (str): Name of the Cohere embedding model to use
        """
        try:
            import cohere
        except ImportError:
            raise ImportError(
                "To use CohereEmbedder, you need to install the cohere library. "
                "Please run: pip install cohere"
            )
        
        # Get API key from environment variable
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError(
                "Cohere API key not found. Please set the COHERE_API_KEY environment variable."
            )
        
        self.client = cohere.Client(api_key)
        self.model_name = model_name
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of text strings using Cohere.
        
        Args:
            texts (List[str]): List of text strings to embed
            
        Returns:
            List[List[float]]: List of embeddings, one for each input text
        """
        response = self.client.embed(
            texts=texts,
            model=self.model_name,
            input_type="search_document"  # or "search_query" for queries
        )
        return response.embeddings
