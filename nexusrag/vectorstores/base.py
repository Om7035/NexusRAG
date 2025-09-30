from abc import ABC, abstractmethod
from typing import List, Dict, Any
from nexusrag.parsers.base import Document


class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    def add(self, docs: List[Document]) -> None:
        """Add documents to the vector store.
        
        Args:
            docs (List[Document]): List of documents to add
        """
        pass
    
    @abstractmethod
    def query(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query the vector store for similar documents.
        
        Args:
            text (str): Query text
            top_k (int): Number of top results to return
            
        Returns:
            List[Dict[str, Any]]: List of similar documents with scores
        """
        pass
