from typing import List, Dict, Any, Callable
from .parsers.base import Document


class MetadataFilter:
    """Utility for filtering documents based on metadata."""
    
    @staticmethod
    def filter_by_metadata(documents: List[Document], 
                          filter_func: Callable[[Dict[str, Any]], bool]) -> List[Document]:
        """Filter documents based on a metadata filter function.
        
        Args:
            documents (List[Document]): Documents to filter
            filter_func (Callable): Function that takes metadata and returns True/False
            
        Returns:
            List[Document]: Filtered documents
        """
        return [doc for doc in documents if filter_func(doc.metadata)]
    
    @staticmethod
    def filter_by_source(documents: List[Document], source_pattern: str) -> List[Document]:
        """Filter documents by source pattern.
        
        Args:
            documents (List[Document]): Documents to filter
            source_pattern (str): Pattern to match in source metadata
            
        Returns:
            List[Document]: Filtered documents
        """
        return [doc for doc in documents if source_pattern in doc.metadata.get("source", "")]
    
    @staticmethod
    def filter_by_content_type(documents: List[Document], content_type: str) -> List[Document]:
        """Filter documents by content type.
        
        Args:
            documents (List[Document]): Documents to filter
            content_type (str): Content type to filter by
            
        Returns:
            List[Document]: Filtered documents
        """
        return [doc for doc in documents if doc.metadata.get("content_type") == content_type]
    
    @staticmethod
    def filter_by_custom_field(documents: List[Document], 
                              field_name: str, 
                              field_value: Any) -> List[Document]:
        """Filter documents by a custom metadata field.
        
        Args:
            documents (List[Document]): Documents to filter
            field_name (str): Name of the metadata field
            field_value (Any): Value to match
            
        Returns:
            List[Document]: Filtered documents
        """
        return [doc for doc in documents if doc.metadata.get(field_name) == field_value]
