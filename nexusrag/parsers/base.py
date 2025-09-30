from abc import ABC, abstractmethod
from typing import List


class Document:
    """Represents a document with its content and metadata."""
    def __init__(self, content: str, metadata: dict = None):
        self.content = content
        self.metadata = metadata or {}


class BaseParser(ABC):
    """Abstract base class for document parsers."""
    
    @abstractmethod
    def parse(self, file_path: str) -> List[Document]:
        """Parse a document file and return a list of Document objects.
        
        Args:
            file_path (str): Path to the document file to parse
            
        Returns:
            List[Document]: List of parsed documents
        """
        pass
