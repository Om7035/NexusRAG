from typing import List
import os
from .base import BaseParser, Document
from ..metadata.extractor import MetadataExtractor


class UniversalParser(BaseParser):
    """Universal document parser that automatically detects file type and uses appropriate parser."""
    
    def parse(self, file_path: str) -> List[Document]:
        """Parse a document based on its file extension.
        
        Args:
            file_path (str): Path to the document file to parse
            
        Returns:
            List[Document]: List of parsed documents
        """
        # Get file extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        # Select appropriate parser based on file extension
        if ext == '.pdf':
            from .pdf import PDFParser
            parser = PDFParser()
        elif ext in ['.docx', '.doc']:
            from .word import WordParser
            parser = WordParser()
        elif ext in ['.html', '.htm']:
            from .html import HTMLParser
            parser = HTMLParser()
        elif ext in ['.md', '.markdown']:
            from .markdown import MarkdownParser
            parser = MarkdownParser()
        elif ext == '.txt':
            from .text import TextParser
            parser = TextParser()
        elif ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']:
            from .image import ImageParser
            parser = ImageParser()
        elif ext in ['.mp3', '.wav', '.flac', '.aac', '.ogg']:
            from .audio import AudioParser
            parser = AudioParser()
        elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']:
            from .video import VideoParser
            parser = VideoParser()
        else:
            # Default to text parser for unknown formats
            from .text import TextParser
            parser = TextParser()
            
        # Parse the document
        documents = parser.parse(file_path)
        
        # Enhance metadata for all documents
        enhanced_documents = []
        for doc in documents:
            enhanced_doc = MetadataExtractor.enhance_document_metadata(doc, file_path)
            enhanced_documents.append(enhanced_doc)
        
        return enhanced_documents
