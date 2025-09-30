from typing import List
import os
from .base import BaseEmbedder


class UniversalEmbedder(BaseEmbedder):
    """Universal embedder that can use different embedding models."""
    
    def __init__(self, provider: str = "sentence-transformers", model_name: str = None):
        """Initialize the universal embedder.
        
        Args:
            provider (str): Embedding provider ("sentence-transformers", "openai", "cohere", "gemini")
            model_name (str): Specific model name to use
        """
        self.provider = provider.lower()
        
        if self.provider == "sentence-transformers":
            from .sentence_transformers import SentenceTransformerEmbedder
            model_name = model_name or "all-MiniLM-L6-v2"
            self.embedder = SentenceTransformerEmbedder(model_name)
        elif self.provider == "openai":
            from .openai import OpenAIEmbedder
            model_name = model_name or "text-embedding-ada-002"
            self.embedder = OpenAIEmbedder(model_name)
        elif self.provider == "cohere":
            from .cohere import CohereEmbedder
            model_name = model_name or "embed-english-v3.0"
            self.embedder = CohereEmbedder(model_name)
        elif self.provider == "gemini":
            from .gemini import GeminiEmbedder
            model_name = model_name or "models/embedding-001"
            self.embedder = GeminiEmbedder(model_name)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of text strings.
        
        Args:
            texts (List[str]): List of text strings to embed
            
        Returns:
            List[List[float]]: List of embeddings, one for each input text
        """
        return self.embedder.embed(texts)
