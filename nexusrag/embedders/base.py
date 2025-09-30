from abc import ABC, abstractmethod
from typing import List


class BaseEmbedder(ABC):
    """Abstract base class for text embedders."""
    
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of text strings.
        
        Args:
            texts (List[str]): List of text strings to embed
            
        Returns:
            List[List[float]]: List of embeddings, one for each input text
        """
        pass
