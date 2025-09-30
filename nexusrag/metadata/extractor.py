from typing import Dict, Any
from ..parsers.base import Document
import os
from datetime import datetime


class MetadataExtractor:
    """Metadata extraction utility for documents."""
    
    @staticmethod
    def extract_file_metadata(file_path: str) -> Dict[str, Any]:
        """Extract metadata from a file.
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            Dict[str, Any]: File metadata
        """
        if not os.path.exists(file_path):
            return {}
        
        stat = os.stat(file_path)
        
        metadata = {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "file_extension": os.path.splitext(file_path)[1],
            "file_size": stat.st_size,
            "created_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "accessed_time": datetime.fromtimestamp(stat.st_atime).isoformat()
        }
        
        return metadata
    
    @staticmethod
    def enhance_document_metadata(document: Document, file_path: str = None) -> Document:
        """Enhance document metadata with file information.
        
        Args:
            document (Document): Document to enhance
            file_path (str): Optional file path
            
        Returns:
            Document: Document with enhanced metadata
        """
        # Copy existing metadata
        enhanced_metadata = document.metadata.copy()
        
        # Add file metadata if path is provided
        if file_path:
            file_metadata = MetadataExtractor.extract_file_metadata(file_path)
            enhanced_metadata.update(file_metadata)
        
        # Add content statistics
        content = document.content
        enhanced_metadata["content_length"] = len(content)
        enhanced_metadata["word_count"] = len(content.split())
        enhanced_metadata["line_count"] = len(content.splitlines())
        
        # Add extraction timestamp
        enhanced_metadata["extraction_timestamp"] = datetime.now().isoformat()
        
        # Create new document with enhanced metadata
        enhanced_document = Document(
            content=document.content,
            metadata=enhanced_metadata
        )
        
        return enhanced_document
    
    @staticmethod
    def extract_content_metadata(content: str) -> Dict[str, Any]:
        """Extract metadata from content.
        
        Args:
            content (str): Content to analyze
            
        Returns:
            Dict[str, Any]: Content metadata
        """
        metadata = {
            "content_length": len(content),
            "word_count": len(content.split()),
            "line_count": len(content.splitlines()),
            "paragraph_count": len(content.split("\n\n")),
            "sentence_count": len(content.split(". "))
        }
        
        return metadata
