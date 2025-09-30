from typing import List
from .base import BaseParser, Document
from PIL import Image
import pytesseract
import os
from ..metadata.extractor import MetadataExtractor

class ImageParser(BaseParser):
    """Image parser that extracts text from images using OCR."""
    
    def parse(self, file_path: str) -> List[Document]:
        """Parse image file and extract text using OCR.
        
        Args:
            file_path (str): Path to the image file
            
        Returns:
            List[Document]: List of parsed documents
        """
        documents = []
        
        # Open image
        image = Image.open(file_path)
        
        # Extract text using OCR
        text = pytesseract.image_to_string(image)
        
        # Create document
        document = Document(
            content=text.strip(),
            metadata={
                "source": file_path,
                "file_type": "image",
                "width": image.width,
                "height": image.height,
                "mode": image.mode
            }
        )
        
        documents.append(document)
        
        # Enhance metadata for all documents
        enhanced_documents = []
        for doc in documents:
            enhanced_doc = MetadataExtractor.enhance_document_metadata(doc, file_path)
            enhanced_documents.append(enhanced_doc)
        
        return enhanced_documents
