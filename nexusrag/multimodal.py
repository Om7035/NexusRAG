from typing import List, Dict, Any
from .parsers.base import Document


class MultimodalProcessor:
    """Processor for handling multimodal data (text, images, tables, etc.)."""
    
    def __init__(self):
        """Initialize the multimodal processor."""
        pass
    
    def process_image(self, image_path: str) -> Document:
        """Process an image and extract text content.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            Document: Document containing extracted text
        """
        try:
            from PIL import Image
            import pytesseract
        except ImportError:
            raise ImportError(
                "To process images, you need to install PIL and pytesseract. "
                "Please run: pip install Pillow pytesseract"
            )
        
        # Open and process image
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        
        # Create document
        metadata = {
            "source": image_path,
            "content_type": "image",
            "media_type": "image"
        }
        
        return Document(content=text.strip(), metadata=metadata)
    
    def process_table(self, table_data: List[List[str]]) -> Document:
        """Process table data and convert to text.
        
        Args:
            table_data (List[List[str]]): 2D list representing table data
            
        Returns:
            Document: Document containing table as text
        """
        # Convert table to text format
        table_text = ""
        for row in table_data:
            table_text += "\t".join(str(cell) for cell in row) + "\n"
        
        # Create document
        metadata = {
            "content_type": "table",
            "media_type": "text"
        }
        
        return Document(content=table_text.strip(), metadata=metadata)
    
    def process_multimodal_document(self, file_path: str) -> List[Document]:
        """Process a document that may contain multiple modalities.
        
        Args:
            file_path (str): Path to the document file
            
        Returns:
            List[Document]: List of documents for each modality
        """
        import os
        
        # Get file extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        documents = []
        
        # Process based on file type
        if ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
            # Image file
            doc = self.process_image(file_path)
            documents.append(doc)
        elif ext == '.pdf':
            # PDF may contain images and text
            # For now, we'll use the existing PDF parser
            # In a more advanced implementation, we would extract images separately
            from .parsers.pdf import PDFParser
            parser = PDFParser()
            documents = parser.parse(file_path)
        else:
            # For other file types, use appropriate parser
            from .parsers.universal import UniversalParser
            parser = UniversalParser()
            documents = parser.parse(file_path)
            
        return documents
