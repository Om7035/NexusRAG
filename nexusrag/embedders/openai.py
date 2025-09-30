from typing import List
import os
from .base import BaseEmbedder


class OpenAIEmbedder(BaseEmbedder):
    """Text embedder using OpenAI's embedding models."""
    
    def __init__(self, model_name: str = "text-embedding-ada-002"):
        """Initialize the OpenAI embedder.
        
        Args:
            model_name (str): Name of the OpenAI embedding model to use
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "To use OpenAIEmbedder, you need to install the openai library. "
                "Please run: pip install openai"
            )
        
        # Get API key from environment variable
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
            )
        
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of text strings using OpenAI.
        
        Args:
            texts (List[str]): List of text strings to embed
            
        Returns:
            List[List[float]]: List of embeddings, one for each input text
        """
        # OpenAI has a limit of 2048 tokens per request
        # We'll process texts in batches
        embeddings = []
        
        for text in texts:
            response = self.client.embeddings.create(
                input=text,
                model=self.model_name
            )
            embedding = response.data[0].embedding
            embeddings.append(embedding)
            
        return embeddings
