from typing import List
from .base import BaseEmbedder


class SentenceTransformerEmbedder(BaseEmbedder):
    """Text embedder using Sentence Transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the embedder with a specific model.
        
        Args:
            model_name (str): Name of the Sentence Transformer model to use
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "To use SentenceTransformerEmbedder, you need to install the sentence-transformers library. "
                "Please run: pip install sentence-transformers"
            )
        
        self.model = SentenceTransformer(model_name)
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of text strings.
        
        Args:
            texts (List[str]): List of text strings to embed
            
        Returns:
            List[List[float]]: List of embeddings, one for each input text
        """
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
