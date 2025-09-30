from typing import List, Dict, Any
from ..parsers.base import Document
import numpy as np


class BGEReranker:
    """BGE-based re-ranker for improving search result precision."""
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        """Initialize the BGE re-ranker.
        
        Args:
            model_name (str): Name of the BGE model to use
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
    def _load_model(self):
        """Load the BGE re-ranker model."""
        if self.model is not None:
            return
            
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.eval()
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.model.to("cuda")
                
        except ImportError as e:
            raise ImportError(
                f"To use BGE re-ranker, you need to install transformers and torch. "
                f"Please run: pip install transformers torch"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load BGE re-ranker model: {e}")
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """Re-rank documents based on their relevance to the query.
        
        Args:
            query (str): Query text
            documents (List[Dict[str, Any]]): List of documents to re-rank
            top_k (int): Number of top results to return
            
        Returns:
            List[Dict[str, Any]]: Re-ranked documents with scores
        """
        # Load model if not already loaded
        self._load_model()
        
        # If no documents, return empty list
        if not documents:
            return []
        
        # Prepare pairs for re-ranking
        pairs = [[query, doc.get("content", "")] for doc in documents]
        
        # Tokenize inputs
        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512
            )
            
            # Move to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Get scores
            scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()
            scores = torch.sigmoid(scores).cpu().numpy()
        
        # Add scores to documents
        for i, doc in enumerate(documents):
            doc["rerank_score"] = float(scores[i])
        
        # Sort by re-rank score
        documents.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        
        return documents[:top_k]


class LightweightReranker:
    """Lightweight re-ranker using simple scoring methods."""
    
    def __init__(self):
        """Initialize the lightweight re-ranker."""
        pass
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """Re-rank documents using lightweight scoring.
        
        Args:
            query (str): Query text
            documents (List[Dict[str, Any]]): List of documents to re-rank
            top_k (int): Number of top results to return
            
        Returns:
            List[Dict[str, Any]]: Re-ranked documents with scores
        """
        # If no documents, return empty list
        if not documents:
            return []
        
        query_terms = query.lower().split()
        
        # Calculate relevance scores
        for doc in documents:
            content = doc.get("content", "").lower()
            score = 0.0
            
            # Count matching terms
            for term in query_terms:
                score += content.count(term)
            
            # Normalize by document length
            if len(content.split()) > 0:
                score = score / len(content.split())
            
            doc["rerank_score"] = score
        
        # Sort by re-rank score
        documents.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        
        return documents[:top_k]
