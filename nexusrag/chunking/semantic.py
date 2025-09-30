from typing import List
from ..parsers.base import Document
import re

class SemanticChunker:
    """Semantic chunking utility that splits documents based on content structure."""
    
    def __init__(self, max_chunk_size: int = 1000):
        """Initialize the semantic chunker.
        
        Args:
            max_chunk_size (int): Maximum size of each chunk in characters
        """
        self.max_chunk_size = max_chunk_size
    
    def chunk_document(self, document: Document) -> List[Document]:
        """Split a document into semantic chunks.
        
        Args:
            document (Document): Document to chunk
            
        Returns:
            List[Document]: List of chunked documents
        """
        content = document.content
        
        # If document is already small enough, return as is
        if len(content) <= self.max_chunk_size:
            return [document]
        
        # Split by paragraphs first
        paragraphs = self._split_by_paragraphs(content)
        
        # Group paragraphs into chunks
        chunks = self._group_paragraphs(paragraphs, document.metadata)
        
        return chunks
    
    def _split_by_paragraphs(self, content: str) -> List[str]:
        """Split content by paragraphs.
        
        Args:
            content (str): Content to split
            
        Returns:
            List[str]: List of paragraphs
        """
        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', content)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _group_paragraphs(self, paragraphs: List[str], base_metadata: dict) -> List[Document]:
        """Group paragraphs into chunks respecting max size.
        
        Args:
            paragraphs (List[str]): List of paragraphs
            base_metadata (dict): Base metadata for chunks
            
        Returns:
            List[Document]: List of chunked documents
        """
        chunks = []
        current_chunk = []
        current_size = 0
        
        for i, paragraph in enumerate(paragraphs):
            # If adding this paragraph would exceed max size
            if current_size + len(paragraph) > self.max_chunk_size and current_chunk:
                # Create chunk from current paragraphs
                chunk_content = "\n\n".join(current_chunk)
                chunk_metadata = base_metadata.copy()
                chunk_metadata["chunk_index"] = len(chunks)
                chunk_metadata["chunk_type"] = "semantic"
                
                chunk_doc = Document(content=chunk_content, metadata=chunk_metadata)
                chunks.append(chunk_doc)
                
                # Start new chunk
                current_chunk = [paragraph]
                current_size = len(paragraph)
            else:
                # Add paragraph to current chunk
                current_chunk.append(paragraph)
                current_size += len(paragraph) + 2  # +2 for \n\n separator
        
        # Add remaining paragraphs as final chunk
        if current_chunk:
            chunk_content = "\n\n".join(current_chunk)
            chunk_metadata = base_metadata.copy()
            chunk_metadata["chunk_index"] = len(chunks)
            chunk_metadata["chunk_type"] = "semantic"
            
            chunk_doc = Document(content=chunk_content, metadata=chunk_metadata)
            chunks.append(chunk_doc)
        
        return chunks
