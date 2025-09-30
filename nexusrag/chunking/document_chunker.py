from typing import List
from ..parsers.base import Document


class DocumentChunker:
    """Document chunking utility for splitting documents into smaller pieces."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize the document chunker.
        
        Args:
            chunk_size (int): Maximum size of each chunk in characters
            chunk_overlap (int): Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_document(self, document: Document) -> List[Document]:
        """Split a document into smaller chunks.
        
        Args:
            document (Document): Document to chunk
            
        Returns:
            List[Document]: List of chunked documents
        """
        content = document.content
        
        # If document is already small enough, return as is
        if len(content) <= self.chunk_size:
            return [document]
        
        # Split content into chunks
        chunks = []
        start = 0
        
        while start < len(content):
            # Calculate end position
            end = min(start + self.chunk_size, len(content))
            
            # Extract chunk
            chunk_content = content[start:end]
            
            # Create new document for chunk
            chunk_metadata = document.metadata.copy()
            chunk_metadata["chunk_index"] = len(chunks)
            chunk_metadata["chunk_start"] = start
            chunk_metadata["chunk_end"] = end
            
            chunk_doc = Document(content=chunk_content, metadata=chunk_metadata)
            chunks.append(chunk_doc)
            
            # Move start position
            start = end - self.chunk_overlap
            
            # If we're at the end, break
            if end == len(content):
                break
        
        return chunks
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split multiple documents into smaller chunks.
        
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
