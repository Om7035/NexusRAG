from typing import List, Dict, Any
from .base import BaseVectorStore
from ..parsers.base import Document


class ChromaVectorStore(BaseVectorStore):
    """Vector store implementation using ChromaDB."""
    
    def __init__(self, collection_name: str = "nexusrag", persist_directory: str = None):
        """Initialize the Chroma vector store.
        
        Args:
            collection_name (str): Name of the Chroma collection
            persist_directory (str): Directory to persist the database (optional)
        """
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "To use ChromaVectorStore, you need to install the chromadb library. "
                "Please run: pip install chromadb"
            )
        
        # Initialize Chroma client
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()
            
        # Get or create collection
        self.collection = self.client.get_or_create_collection(name=collection_name)
    
    def add(self, docs: List[Document]) -> None:
        """Add documents to the vector store.
        
        Args:
            docs (List[Document]): List of documents to add
        """
        # Extract content and metadata
        contents = [doc.content for doc in docs]
        metadatas = [doc.metadata for doc in docs]
        ids = [f"doc_{i}" for i in range(len(docs))]
        
        # Add to collection
        self.collection.add(
            documents=contents,
            metadatas=metadatas,
            ids=ids
        )
    
    def query(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query the vector store for similar documents.
        
        Args:
            text (str): Query text
            top_k (int): Number of top results to return
            
        Returns:
            List[Dict[str, Any]]: List of similar documents with scores
        """
        # Query the collection
        results = self.collection.query(
            query_texts=[text],
            n_results=top_k
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            result = {
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "score": results['distances'][0][i] if 'distances' in results else None
            }
            formatted_results.append(result)
            
        return formatted_results
