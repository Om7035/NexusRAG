from typing import List
from ..parsers.base import Document
import re

class SentenceChunker:
    """Sentence-based chunking utility that splits documents by sentences."""
    
    def __init__(self, max_chunk_size: int = 1000, min_chunk_size: int = 100):
        """Initialize the sentence chunker.
        
        Args:
            max_chunk_size (int): Maximum size of each chunk in characters
            min_chunk_size (int): Minimum size of each chunk in characters
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
    
    def chunk_document(self, document: Document) -> List[Document]:
        """Split a document into sentence-based chunks.
        
        Args:
            document (Document): Document to chunk
            
        Returns:
            List[Document]: List of chunked documents
        """
        content = document.content
        
        # If document is already small enough, return as is
        if len(content) <= self.max_chunk_size:
            return [document]
        
        # Split by sentences
        sentences = self._split_by_sentences(content)
        
        # Group sentences into chunks
        chunks = self._group_sentences(sentences, document.metadata)
        
        return chunks
    
    def _split_by_sentences(self, content: str) -> List[str]:
        """Split content by sentences.
        
        Args:
            content (str): Content to split
            
        Returns:
            List[str]: List of sentences
        """
        # Split by sentence endings
        sentences = re.split(r'[.!?]+\s+', content)
        return [s.strip() for s in sentences if s.strip()]
    
    def _group_sentences(self, sentences: List[str], base_metadata: dict) -> List[Document]:
        """Group sentences into chunks respecting size limits.
        
        Args:
            sentences (List[str]): List of sentences
            base_metadata (dict): Base metadata for chunks
            
        Returns:
            List[Document]: List of chunked documents
        """
        chunks = []
        current_chunk = []
        current_size = 0
        
        for i, sentence in enumerate(sentences):
            # If adding this sentence would exceed max size
            if current_size + len(sentence) > self.max_chunk_size and current_chunk:
                # Create chunk from current sentences
                chunk_content = " ".join(current_chunk)
                
                # Only create chunk if it meets minimum size requirement
                if len(chunk_content) >= self.min_chunk_size:
                    chunk_metadata = base_metadata.copy()
                    chunk_metadata["chunk_index"] = len(chunks)
                    chunk_metadata["chunk_type"] = "sentence"
                    
                    chunk_doc = Document(content=chunk_content, metadata=chunk_metadata)
                    chunks.append(chunk_doc)
                
                # Start new chunk
                current_chunk = [sentence]
                current_size = len(sentence)
            else:
                # Add sentence to current chunk
                current_chunk.append(sentence)
                current_size += len(sentence) + 1  # +1 for space separator
        
        # Add remaining sentences as final chunk
        if current_chunk:
            chunk_content = " ".join(current_chunk)
            
            # Only create chunk if it meets minimum size requirement
            if len(chunk_content) >= self.min_chunk_size:
                chunk_metadata = base_metadata.copy()
                chunk_metadata["chunk_index"] = len(chunks)
                chunk_metadata["chunk_type"] = "sentence"
                
                chunk_doc = Document(content=chunk_content, metadata=chunk_metadata)
                chunks.append(chunk_doc)
        
        return chunks
