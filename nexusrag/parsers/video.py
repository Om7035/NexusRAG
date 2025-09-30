from typing import List
from .base import BaseParser, Document
import os
from ..metadata.extractor import MetadataExtractor


class VideoParser(BaseParser):
    """Video parser that extracts metadata from video files."""
    
    def parse(self, file_path: str) -> List[Document]:
        """Parse video file and extract metadata.
        
        Args:
            file_path (str): Path to the video file
            
        Returns:
            List[Document]: List of parsed documents
        """
        documents = []
        
        # Get file metadata
        file_stats = os.stat(file_path)
        
        # Create document with metadata
        document = Document(
            content=f"[Video file: {os.path.basename(file_path)}]",
            metadata={
                "source": file_path,
                "file_type": "video",
                "file_size": file_stats.st_size,
                "file_extension": os.path.splitext(file_path)[1]
            }
        )
        
        documents.append(document)
        
        # Enhance metadata for all documents
        enhanced_documents = []
        for doc in documents:
            enhanced_doc = MetadataExtractor.enhance_document_metadata(doc, file_path)
            enhanced_documents.append(enhanced_doc)
        
        return enhanced_documents
