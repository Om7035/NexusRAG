from typing import List, Dict, Any
from ..parsers.base import Document
from ..vectorstores.base import BaseVectorStore
from .bm25 import BM25Retriever
import numpy as np


class HybridRetriever:
    """Hybrid retriever that combines vector search and keyword search."""
    
    def __init__(self, vector_store: BaseVectorStore, keyword_weight: float = 0.5):
        """Initialize the hybrid retriever.
        
        Args:
            vector_store (BaseVectorStore): Vector store for semantic search
            keyword_weight (float): Weight for keyword search (0.0 to 1.0)
        """
        self.vector_store = vector_store
        self.keyword_weight = keyword_weight
        self.bm25_retriever = BM25Retriever()
    
    def add_documents(self, docs: List[Document]) -> None:
        """Add documents to both vector store and keyword index.
        
        Args:
            docs (List[Document]): List of documents to add
        """
        # Add to vector store
        self.vector_store.add(docs)
        
        # Add to BM25 retriever
        self.bm25_retriever.add_documents(docs)
    
    def hybrid_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Perform hybrid search combining vector and keyword search.
        
        Args:
            query (str): Query text
            top_k (int): Number of top results to return
            
        Returns:
            List[Dict[str, Any]]: List of search results with scores
        """
        # Perform vector search
        vector_results = self.vector_store.query(query, top_k * 2)  # Get more results for re-ranking
        
        # Perform keyword search using BM25
        keyword_results = self.bm25_retriever.search(query, top_k * 2)
        
        # Combine results
        combined_results = self._combine_results(vector_results, keyword_results, top_k)
        
        return combined_results
    
    
    def _combine_results(self, vector_results: List[Dict[str, Any]], 
                        keyword_results: List[Dict[str, Any]], 
                        top_k: int) -> List[Dict[str, Any]]:
        """Combine vector and keyword search results.
        
        Args:
            vector_results (List[Dict[str, Any]]): Vector search results
            keyword_results (List[Dict[str, Any]]): Keyword search results
            top_k (int): Number of top results to return
            
        Returns:
            List[Dict[str, Any]]: Combined search results with scores
        """
        # Normalize scores
        vector_scores = [result.get("score", 0.0) for result in vector_results]
        keyword_scores = [result.get("score", 0.0) for result in keyword_results]
        
        # Handle empty results
        if not vector_scores and not keyword_scores:
            return []
        
        # Normalize scores to 0-1 range
        max_vector_score = max(vector_scores) if vector_scores else 1.0
        max_keyword_score = max(keyword_scores) if keyword_scores else 1.0
        
        if max_vector_score == 0:
            max_vector_score = 1.0
        if max_keyword_score == 0:
            max_keyword_score = 1.0
        
        # Create a map of results by content for deduplication
        result_map = {}
        
        # Process vector results
        for result in vector_results:
            content = result["content"]
            normalized_score = result.get("score", 0.0) / max_vector_score
            hybrid_score = (1 - self.keyword_weight) * normalized_score
            
            result_map[content] = {
                "content": content,
                "metadata": result["metadata"],
                "vector_score": normalized_score,
                "keyword_score": 0.0,
                "hybrid_score": hybrid_score
            }
        
        # Process keyword results
        for result in keyword_results:
            content = result["content"]
            normalized_score = result.get("score", 0.0) / max_keyword_score
            keyword_score = self.keyword_weight * normalized_score
            
            if content in result_map:
                # Update existing result
                result_map[content]["keyword_score"] = normalized_score
                result_map[content]["hybrid_score"] += keyword_score
            else:
                # Add new result
                result_map[content] = {
                    "content": content,
                    "metadata": result["metadata"],
                    "vector_score": 0.0,
                    "keyword_score": normalized_score,
                    "hybrid_score": keyword_score
                }
        
        # Convert to list and sort by hybrid score
        combined_results = list(result_map.values())
        combined_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        
        # Return top_k results
        return combined_results[:top_k]
