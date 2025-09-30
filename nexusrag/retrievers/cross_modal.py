from typing import List, Dict, Any
from ..parsers.base import Document
from ..vectorstores.base import BaseVectorStore
from ..embedders.base import BaseEmbedder
import numpy as np


class CrossModalRetriever:
    """Cross-modal retriever for retrieving across different modalities."""
    
    def __init__(self, vector_store: BaseVectorStore, embedder: BaseEmbedder):
        """Initialize the cross-modal retriever.
        
        Args:
            vector_store (BaseVectorStore): Vector store for retrieval
            embedder (BaseEmbedder): Embedder for different modalities
        """
        self.vector_store = vector_store
        self.embedder = embedder
    
    def add_multimodal_documents(self, documents: List[Document]) -> None:
        """Add multimodal documents to the vector store.
        
        Args:
            documents (List[Document]): List of multimodal documents to add
        """
        # Add documents to vector store
        self.vector_store.add(documents)
    
    def cross_modal_search(self, query: str, modality: str = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """Perform cross-modal search across different modalities.
        
        Args:
            query (str): Query text
            modality (str): Target modality to retrieve ("text", "image", "audio", "video", "table", etc.)
            top_k (int): Number of top results to return
            
        Returns:
            List[Dict[str, Any]]: List of search results with scores
        """
        # Perform standard vector search
        results = self.vector_store.query(query, top_k * 2)  # Get more results for filtering
        
        # Filter by modality if specified
        if modality:
            filtered_results = []
            for result in results:
                metadata = result.get("metadata", {})
                content_type = metadata.get("content_type", "unknown")
                media_type = metadata.get("media_type", "unknown")
                
                # Check if result matches the target modality
                if self._matches_modality(content_type, media_type, modality):
                    filtered_results.append(result)
            
            results = filtered_results
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return results[:top_k]
    
    def _matches_modality(self, content_type: str, media_type: str, target_modality: str) -> bool:
        """Check if a document matches the target modality.
        
        Args:
            content_type (str): Content type from metadata
            media_type (str): Media type from metadata
            target_modality (str): Target modality to match
            
        Returns:
            bool: True if document matches target modality
        """
        # Normalize modality names
        content_type = content_type.lower()
        media_type = media_type.lower()
        target_modality = target_modality.lower()
        
        # Check for matches
        if target_modality == "text":
            return content_type in ["text", "table"] or media_type in ["text"]
        elif target_modality == "image":
            return content_type == "image" or media_type == "image"
        elif target_modality == "audio":
            return content_type == "audio" or media_type == "audio"
        elif target_modality == "video":
            return content_type == "video" or media_type == "video"
        elif target_modality == "table":
            return content_type == "table"
        else:
            # If no specific modality, return True
            return True
    
    def multimodal_fusion_search(self, queries: Dict[str, str], weights: Dict[str, float] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """Perform fusion search across multiple modalities.
        
        Args:
            queries (Dict[str, str]): Dictionary of queries for different modalities
            weights (Dict[str, float]): Weights for each modality (optional)
            top_k (int): Number of top results to return
            
        Returns:
            List[Dict[str, Any]]: List of fused search results with scores
        """
        # If no weights provided, use equal weights
        if weights is None:
            weights = {modality: 1.0/len(queries) for modality in queries}
        
        # Perform search for each modality
        modality_results = {}
        for modality, query in queries.items():
            results = self.vector_store.query(query, top_k * 2)
            modality_results[modality] = results
        
        # Fuse results
        fused_results = self._fuse_results(modality_results, weights, top_k)
        
        return fused_results
    
    def _fuse_results(self, modality_results: Dict[str, List[Dict[str, Any]]], 
                     weights: Dict[str, float], 
                     top_k: int) -> List[Dict[str, Any]]:
        """Fuse results from different modalities.
        
        Args:
            modality_results (Dict[str, List[Dict[str, Any]]]): Results from each modality
            weights (Dict[str, float]): Weights for each modality
            top_k (int): Number of top results to return
            
        Returns:
            List[Dict[str, Any]]: Fused search results with scores
        """
        # Create a map of results by content for deduplication
        result_map = {}
        
        # Process results from each modality
        for modality, results in modality_results.items():
            weight = weights.get(modality, 1.0/len(modality_results))
            
            for result in results:
                content = result["content"]
                score = result.get("score", 0.0)
                weighted_score = score * weight
                
                if content in result_map:
                    # Update existing result
                    result_map[content]["fused_score"] += weighted_score
                    result_map[content]["modalities"].append(modality)
                else:
                    # Add new result
                    result_map[content] = {
                        "content": content,
                        "metadata": result["metadata"],
                        "original_score": score,
                        "fused_score": weighted_score,
                        "modalities": [modality]
                    }
        
        # Convert to list and sort by fused score
        fused_results = list(result_map.values())
        fused_results.sort(key=lambda x: x["fused_score"], reverse=True)
        
        # Return top_k results
        return fused_results[:top_k]
