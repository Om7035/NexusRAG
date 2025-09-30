from typing import List, Dict, Any
from .base import BaseVectorStore
from ..parsers.base import Document


class UniversalVectorStore(BaseVectorStore):
    """Universal vector store that can use different vector store implementations."""
    
    def __init__(self, provider: str = "chroma", **kwargs):
        """Initialize the universal vector store.
        
        Args:
            provider (str): Vector store provider ("chroma", "pinecone", "weaviate")
            **kwargs: Additional arguments for the specific vector store
        """
        self.provider = provider.lower()
        
        if self.provider == "chroma":
            from .chroma import ChromaVectorStore
            self.vector_store = ChromaVectorStore(**kwargs)
        elif self.provider == "pinecone":
            from .pinecone import PineconeVectorStore
            self.vector_store = PineconeVectorStore(**kwargs)
        elif self.provider == "weaviate":
            from .weaviate import WeaviateVectorStore
            self.vector_store = WeaviateVectorStore(**kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def add(self, docs: List[Document]) -> None:
        """Add documents to the vector store.
        
        Args:
            docs (List[Document]): List of documents to add
        """
        self.vector_store.add(docs)
    
    def query(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query the vector store for similar documents.
        
        Args:
            text (str): Query text
            top_k (int): Number of top results to return
            
        Returns:
            List[Dict[str, Any]]: List of similar documents with scores
        """
        return self.vector_store.query(text, top_k)
