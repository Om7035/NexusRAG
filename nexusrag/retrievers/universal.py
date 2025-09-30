from typing import List, Dict, Any
from ..parsers.base import Document
from ..vectorstores.base import BaseVectorStore
from ..embedders.base import BaseEmbedder
from .hybrid import HybridRetriever
from .reranker import BGEReranker, LightweightReranker
from .cross_modal import CrossModalRetriever


class UniversalRetriever:
    """Universal retriever that combines hybrid search, re-ranking, and cross-modal retrieval."""
    
    def __init__(self, vector_store: BaseVectorStore, embedder: BaseEmbedder, 
                 use_bge_reranker: bool = False, keyword_weight: float = 0.3):
        """Initialize the universal retriever.
        
        Args:
            vector_store (BaseVectorStore): Vector store for retrieval
            embedder (BaseEmbedder): Embedder for different modalities
            use_bge_reranker (bool): Whether to use BGE re-ranker or lightweight re-ranker
            keyword_weight (float): Weight for keyword search in hybrid retrieval
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.use_bge_reranker = use_bge_reranker
        self.keyword_weight = keyword_weight
        
        # Initialize components
        self.hybrid_retriever = HybridRetriever(vector_store, keyword_weight)
        self.cross_modal_retriever = CrossModalRetriever(vector_store, embedder)
        
        if use_bge_reranker:
            self.reranker = BGEReranker()
        else:
            self.reranker = LightweightReranker()
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the retriever.
        
        Args:
            documents (List[Document]): List of documents to add
        """
        # Add to hybrid retriever (which also adds to vector store)
        self.hybrid_retriever.add_documents(documents)
        
        # Add to cross-modal retriever
        self.cross_modal_retriever.add_multimodal_documents(documents)
    
    def search(self, query: str, top_k: int = 5, use_reranking: bool = True) -> List[Dict[str, Any]]:
        """Perform standard search with optional re-ranking.
        
        Args:
            query (str): Query text
            top_k (int): Number of top results to return
            use_reranking (bool): Whether to apply re-ranking
            
        Returns:
            List[Dict[str, Any]]: List of search results
        """
        # Perform hybrid search
        results = self.hybrid_retriever.hybrid_search(query, top_k * 2)
        
        # Apply re-ranking if requested
        if use_reranking and results:
            results = self.reranker.rerank(query, results, top_k)
        else:
            # Sort by hybrid score and return top_k
            results.sort(key=lambda x: x.get("hybrid_score", x.get("score", 0.0)), reverse=True)
            results = results[:top_k]
        
        return results
    
    def cross_modal_search(self, query: str, modality: str = None, top_k: int = 5, 
                          use_reranking: bool = True) -> List[Dict[str, Any]]:
        """Perform cross-modal search with optional re-ranking.
        
        Args:
            query (str): Query text
            modality (str): Target modality to retrieve
            top_k (int): Number of top results to return
            use_reranking (bool): Whether to apply re-ranking
            
        Returns:
            List[Dict[str, Any]]: List of cross-modal search results
        """
        # Perform cross-modal search
        results = self.cross_modal_retriever.cross_modal_search(query, modality, top_k * 2)
        
        # Apply re-ranking if requested
        if use_reranking and results:
            results = self.reranker.rerank(query, results, top_k)
        else:
            # Sort by score and return top_k
            results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            results = results[:top_k]
        
        return results
    
    def multimodal_fusion_search(self, queries: Dict[str, str], weights: Dict[str, float] = None, 
                                top_k: int = 5, use_reranking: bool = True) -> List[Dict[str, Any]]:
        """Perform multimodal fusion search with optional re-ranking.
        
        Args:
            queries (Dict[str, str]): Dictionary of queries for different modalities
            weights (Dict[str, float]): Weights for each modality
            top_k (int): Number of top results to return
            use_reranking (bool): Whether to apply re-ranking
            
        Returns:
            List[Dict[str, Any]]: List of fused search results
        """
        # Perform multimodal fusion search
        results = self.cross_modal_retriever.multimodal_fusion_search(queries, weights, top_k * 2)
        
        # Apply re-ranking if requested
        if use_reranking and results:
            # Use the first query for re-ranking
            first_query = next(iter(queries.values()))
            results = self.reranker.rerank(first_query, results, top_k)
        else:
            # Sort by fused score and return top_k
            results.sort(key=lambda x: x.get("fused_score", 0.0), reverse=True)
            results = results[:top_k]
        
        return results
    
    def get_search_metadata(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get metadata about the search results.
        
        Args:
            results (List[Dict[str, Any]]): Search results
            
        Returns:
            Dict[str, Any]: Metadata about the search results
        """
        if not results:
            return {"total_results": 0}
        
        # Count modalities
        modality_counts = {}
        for result in results:
            metadata = result.get("metadata", {})
            content_type = metadata.get("content_type", "unknown")
            media_type = metadata.get("media_type", "unknown")
            
            modality = f"{content_type}/{media_type}"
            modality_counts[modality] = modality_counts.get(modality, 0) + 1
        
        return {
            "total_results": len(results),
            "modalities": modality_counts,
            "avg_score": sum(result.get("score", 0.0) for result in results) / len(results)
        }
