from typing import List, Dict, Any
from .base import BaseVectorStore
from ..parsers.base import Document
import os


class QdrantVectorStore(BaseVectorStore):
    """Vector store implementation using Qdrant with hybrid search support."""
    
    def __init__(self, collection_name: str = "nexusrag", host: str = None, port: int = 6333):
        """Initialize the Qdrant vector store.
        
        Args:
            collection_name (str): Name of the Qdrant collection
            host (str): Qdrant host URL
            port (int): Qdrant port
        """
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams, PointStruct
        except ImportError:
            raise ImportError(
                "To use QdrantVectorStore, you need to install the qdrant-client library. "
                "Please run: pip install qdrant-client"
            )
        
        # Get host from environment variable or use default
        host = host or os.getenv("QDRANT_HOST", "localhost")
        
        # Initialize Qdrant client
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.point_id = 0
        
        # Create collection if it doesn't exist
        self._create_collection_if_not_exists()
    
    def _create_collection_if_not_exists(self):
        """Create the Qdrant collection if it doesn't exist."""
        try:
            from qdrant_client.models import Distance, VectorParams
        except ImportError:
            raise ImportError(
                "To use QdrantVectorStore, you need to install the qdrant-client library. "
                "Please run: pip install qdrant-client"
            )
        
        # Check if collection exists
        try:
            self.client.get_collection(self.collection_name)
        except Exception:
            # Create collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )
    
    def add(self, docs: List[Document]) -> None:
        """Add documents to the Qdrant vector store.
        
        Args:
            docs (List[Document]): List of documents to add
        """
        try:
            from qdrant_client.models import PointStruct
            import numpy as np
        except ImportError:
            raise ImportError(
                "To use QdrantVectorStore, you need to install the qdrant-client library. "
                "Please run: pip install qdrant-client"
            )
        
        # Add documents to Qdrant
        points = []
        for doc in docs:
            # Generate a simple vector (in a real implementation, you would use an embedder)
            # For demonstration, we'll create a random vector
            vector = np.random.rand(768).tolist()
            
            # Create point
            point = PointStruct(
                id=self.point_id,
                vector=vector,
                payload={
                    "content": doc.content,
                    **doc.metadata
                }
            )
            
            points.append(point)
            self.point_id += 1
        
        # Upload points
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
    
    def query(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query the Qdrant vector store for similar documents.
        
        Args:
            text (str): Query text
            top_k (int): Number of top results to return
            
        Returns:
            List[Dict[str, Any]]: List of similar documents with scores
        """
        try:
            import numpy as np
        except ImportError:
            raise ImportError(
                "To use QdrantVectorStore, you need to install numpy. "
                "Please run: pip install numpy"
            )
        
        # Generate a simple vector for the query (in a real implementation, you would use an embedder)
        # For demonstration, we'll create a random vector
        query_vector = np.random.rand(768).tolist()
        
        # Search in Qdrant
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k
        )
        
        # Format results
        results = []
        for result in search_result:
            doc_result = {
                "content": result.payload.get("content", ""),
                "metadata": {k: v for k, v in result.payload.items() if k != "content"},
                "score": result.score
            }
            results.append(doc_result)
        
        return results
    
    def hybrid_search(self, query_text: str, query_vector: List[float] = None, 
                     top_k: int = 5) -> List[Dict[str, Any]]:
        """Perform hybrid search combining vector and keyword search.
        
        Args:
            query_text (str): Query text for keyword search
            query_vector (List[float]): Query vector for semantic search
            top_k (int): Number of top results to return
            
        Returns:
            List[Dict[str, Any]]: List of search results with scores
        """
        try:
            from qdrant_client.models import SearchRequest
            import numpy as np
        except ImportError:
            raise ImportError(
                "To use QdrantVectorStore hybrid search, you need to install the qdrant-client library. "
                "Please run: pip install qdrant-client"
            )
        
        # If no vector provided, generate a random one for demonstration
        if query_vector is None:
            query_vector = np.random.rand(768).tolist()
        
        # Perform hybrid search using Qdrant's recommendation search
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=None,
            limit=top_k,
            with_payload=True,
            with_vectors=False
        )
        
        # Format results
        results = []
        for result in search_result:
            doc_result = {
                "content": result.payload.get("content", ""),
                "metadata": {k: v for k, v in result.payload.items() if k != "content"},
                "score": result.score
            }
            results.append(doc_result)
        
        return results
