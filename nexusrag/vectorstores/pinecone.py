from typing import List, Dict, Any
import os
from .base import BaseVectorStore
from ..parsers.base import Document


class PineconeVectorStore(BaseVectorStore):
    """Vector store implementation using Pinecone."""
    
    def __init__(self, index_name: str = "nexusrag", dimension: int = 384):
        """Initialize the Pinecone vector store.
        
        Args:
            index_name (str): Name of the Pinecone index
            dimension (int): Dimension of the embeddings
        """
        try:
            import pinecone
        except ImportError:
            raise ImportError(
                "To use PineconeVectorStore, you need to install the pinecone-client library. "
                "Please run: pip install pinecone-client"
            )
        
        # Get API key from environment variable
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError(
                "Pinecone API key not found. Please set the PINECONE_API_KEY environment variable."
            )
        
        # Get environment from environment variable
        environment = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")
        
        # Initialize Pinecone
        pinecone.init(api_key=api_key, environment=environment)
        
        # Create or connect to index
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(index_name, dimension=dimension, metric="cosine")
        
        self.index = pinecone.Index(index_name)
        self.index_name = index_name
    
    def add(self, docs: List[Document]) -> None:
        """Add documents to the Pinecone vector store.
        
        Args:
            docs (List[Document]): List of documents to add
        """
        # In a real implementation, you would first embed the documents
        # For this example, we'll assume embeddings are already available
        # or generate them on the fly
        
        vectors = []
        for i, doc in enumerate(docs):
            # Generate a unique ID for each document
            doc_id = f"{self.index_name}_{i}"
            
            # In a real implementation, you would embed the document content
            # For now, we'll create a placeholder
            vector = [0.0] * 384  # Placeholder vector
            
            # Create metadata
            metadata = {
                "content": doc.content,
                **doc.metadata
            }
            
            vectors.append((doc_id, vector, metadata))
        
        # Upsert vectors to Pinecone
        self.index.upsert(vectors)
    
    def query(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query the Pinecone vector store for similar documents.
        
        Args:
            text (str): Query text
            top_k (int): Number of top results to return
            
        Returns:
            List[Dict[str, Any]]: List of similar documents with scores
        """
        # In a real implementation, you would first embed the query text
        # For this example, we'll create a placeholder query vector
        query_vector = [0.0] * 384  # Placeholder vector
        
        # Query Pinecone
        response = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )
        
        # Format results
        results = []
        for match in response.matches:
            result = {
                "content": match.metadata.get("content", ""),
                "metadata": {k: v for k, v in match.metadata.items() if k != "content"},
                "score": match.score
            }
            results.append(result)
            
        return results
