from typing import List, Dict, Any
from ..parsers.base import Document
import math


class BM25Retriever:
    """BM25-based keyword retriever for precise keyword search."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """Initialize the BM25 retriever.
        
        Args:
            k1 (float): BM25 k1 parameter
            b (float): BM25 b parameter
        """
        self.k1 = k1
        self.b = b
        self.documents = []
        self.avg_doc_length = 0.0
        self.doc_count = 0
        self.term_freqs = {}  # term -> doc_id -> frequency
        self.doc_freqs = {}   # term -> document frequency
        self.doc_lengths = {} # doc_id -> length
    
    def add_documents(self, docs: List[Document]) -> None:
        """Add documents to the BM25 index.
        
        Args:
            docs (List[Document]): List of documents to add
        """
        for i, doc in enumerate(docs):
            doc_id = len(self.documents)
            self.documents.append(doc)
            
            # Tokenize content
            tokens = self._tokenize(doc.content)
            doc_length = len(tokens)
            self.doc_lengths[doc_id] = doc_length
            
            # Calculate term frequencies
            term_freq = {}
            for token in tokens:
                term_freq[token] = term_freq.get(token, 0) + 1
            
            # Update term frequencies and document frequencies
            for term, freq in term_freq.items():
                if term not in self.term_freqs:
                    self.term_freqs[term] = {}
                self.term_freqs[term][doc_id] = freq
                
                if term not in self.doc_freqs:
                    self.doc_freqs[term] = 0
                self.doc_freqs[term] += 1
        
        # Update statistics
        self.doc_count = len(self.documents)
        if self.doc_lengths:
            self.avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search documents using BM25 scoring.
        
        Args:
            query (str): Query text
            top_k (int): Number of top results to return
            
        Returns:
            List[Dict[str, Any]]: List of search results with scores
        """
        if not self.documents:
            return []
        
        # Tokenize query
        query_terms = self._tokenize(query)
        
        # Calculate scores for each document
        scores = {}
        for doc_id in range(len(self.documents)):
            score = 0.0
            for term in query_terms:
                if term in self.term_freqs and doc_id in self.term_freqs[term]:
                    # Calculate BM25 score for this term
                    term_freq = self.term_freqs[term][doc_id]
                    doc_freq = self.doc_freqs.get(term, 0)
                    doc_length = self.doc_lengths.get(doc_id, 0)
                    
                    # BM25 formula
                    idf = math.log((self.doc_count - doc_freq + 0.5) / (doc_freq + 0.5))
                    numerator = term_freq * (self.k1 + 1)
                    denominator = term_freq + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
                    score += idf * (numerator / denominator)
            
            scores[doc_id] = score
        
        # Sort by score and return top_k
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Format results
        results = []
        for doc_id, score in sorted_docs:
            if score > 0:  # Only include documents with positive scores
                doc = self.documents[doc_id]
                result = {
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "score": score
                }
                results.append(result)
        
        return results
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer that splits text into terms.
        
        Args:
            text (str): Text to tokenize
            
        Returns:
            List[str]: List of terms
        """
        # Simple tokenization by splitting on whitespace and removing punctuation
        import re
        terms = re.findall(r'\b\w+\b', text.lower())
        return terms
