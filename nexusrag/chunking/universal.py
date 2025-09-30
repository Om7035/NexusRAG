from typing import List
from ..parsers.base import Document
from .document_chunker import DocumentChunker
from .semantic import SemanticChunker
from .sentence import SentenceChunker

class UniversalChunker:
    """Universal chunking utility that can use different chunking strategies."""
    
    def __init__(self, strategy: str = "character", **kwargs):
        """Initialize the universal chunker.
        
        Args:
            strategy (str): Chunking strategy ("character", "semantic", "sentence")
            **kwargs: Additional arguments for the chunker
        """
        self.strategy = strategy
        
        if strategy == "character":
            self.chunker = DocumentChunker(**kwargs)
        elif strategy == "semantic":
            self.chunker = SemanticChunker(**kwargs)
        elif strategy == "sentence":
            self.chunker = SentenceChunker(**kwargs)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
    
    def chunk_document(self, document: Document) -> List[Document]:
        """Split a document using the selected strategy.
        
        Args:
            document (Document): Document to chunk
            
        Returns:
            List[Document]: List of chunked documents
        """
        if self.strategy == "character":
            return self.chunker.chunk_document(document)
        elif self.strategy == "semantic":
            return self.chunker.chunk_document(document)
        elif self.strategy == "sentence":
            return self.chunker.chunk_document(document)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.strategy}")
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split multiple documents using the selected strategy.
        
        Args:
            documents (List[Document]): Documents to chunk
            
        Returns:
            List[Document]: List of chunked documents
        """
        chunked_documents = []
        
        for doc in documents:
            chunks = self.chunk_document(doc)
            chunked_documents.extend(chunks)
            
        return chunked_documents
