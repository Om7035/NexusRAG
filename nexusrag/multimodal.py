from typing import List, Dict, Any
from nexusrag.parsers.base import Document
from nexusrag.multimodal.universal import UniversalMultimodalProcessor


class MultimodalProcessor:
    """Processor for handling multimodal data (text, images, tables, etc.)."""
    
    def __init__(self):
        """Initialize the multimodal processor."""
        self.processor = UniversalMultimodalProcessor()
    
    def process_image(self, image_path: str) -> Document:
        """Process an image and extract content.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            Document: Document containing image content
        """
        return self.processor.process_image(image_path)
    
    def process_table(self, table_data: List[List[str]]) -> Document:
        """Process table data and convert to structured format.
        
        Args:
            table_data (List[List[str]]): 2D list representing table data
            
        Returns:
            Document: Document containing table as structured text
        """
        return self.processor.process_table_data(table_data)
    
    def process_multimodal_document(self, file_path: str) -> List[Document]:
        """Process a document that may contain multiple modalities.
        
        Args:
            file_path (str): Path to the document file
            
        Returns:
            List[Document]: List of documents for each modality
        """
        return self.processor.process_file(file_path)
    
    def process_audio(self, audio_path: str) -> Document:
        """Process an audio file and generate transcription.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            Document: Document containing audio transcription
        """
        return self.processor.process_audio(audio_path)
    
    def process_video(self, video_path: str) -> Document:
        """Process a video file and generate transcription.
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            Document: Document containing video transcription
        """
        return self.processor.process_video(video_path)
    
    def process_pdf(self, pdf_path: str) -> List[Document]:
        """Process a PDF file with advanced capabilities.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            List[Document]: List of documents containing PDF content
        """
        return self.processor.process_pdf(pdf_path)
    
    def process_html_table(self, html_content: str) -> Document:
        """Process HTML table and convert to structured format.
        
        Args:
            html_content (str): HTML content containing table
            
        Returns:
            Document: Document containing table as structured text
        """
        return self.processor.process_html_table(html_content)
    
    def extract_tables_from_document(self, document_content: str) -> List[Document]:
        """Extract tables from document content.
        
        Args:
            document_content (str): Document content that may contain tables
            
        Returns:
            List[Document]: List of documents containing extracted tables
        """
        return self.processor.extract_tables_from_document(document_content)
