from typing import List, Dict, Any
from ..parsers.base import Document
from .image_processor import ImageProcessor
from .audio_processor import AudioProcessor
from .table_processor import TableProcessor
from .pdf_processor import PDFProcessor
import os


class UniversalMultimodalProcessor:
    """Universal multimodal processor that integrates all modalities."""
    
    def __init__(self):
        """Initialize the universal multimodal processor."""
        self.image_processor = ImageProcessor()
        self.audio_processor = AudioProcessor()
        self.table_processor = TableProcessor()
        self.pdf_processor = PDFProcessor()
    
    def process_file(self, file_path: str) -> List[Document]:
        """Process a file based on its type.
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            List[Document]: List of documents containing processed content
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Get file extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        # Process based on file type
        if ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']:
            # Image file
            doc = self.image_processor.process_image(file_path)
            return [doc]
        elif ext in ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a']:
            # Audio file
            doc = self.audio_processor.process_audio(file_path)
            return [doc]
        elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']:
            # Video file
            doc = self.audio_processor.process_video(file_path)
            return [doc]
        elif ext == '.pdf':
            # PDF file
            return self.pdf_processor.process_pdf(file_path)
        else:
            # For other file types, use existing parsers
            from ..parsers.universal import UniversalParser
            parser = UniversalParser()
            return parser.parse(file_path)
    
    def process_image(self, image_path: str) -> Document:
        """Process an image file.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            Document: Document containing image content
        """
        return self.image_processor.process_image(image_path)
    
    def process_audio(self, audio_path: str) -> Document:
        """Process an audio file.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            Document: Document containing audio transcription
        """
        return self.audio_processor.process_audio(audio_path)
    
    def process_video(self, video_path: str) -> Document:
        """Process a video file.
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            Document: Document containing video transcription
        """
        return self.audio_processor.process_video(video_path)
    
    def process_pdf(self, pdf_path: str) -> List[Document]:
        """Process a PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            List[Document]: List of documents containing PDF content
        """
        return self.pdf_processor.process_pdf(pdf_path)
    
    def process_table_data(self, table_data: List[List[str]]) -> Document:
        """Process table data.
        
        Args:
            table_data (List[List[str]]): 2D list representing table data
            
        Returns:
            Document: Document containing table content
        """
        return self.table_processor.process_table_data(table_data)
    
    def process_html_table(self, html_content: str) -> Document:
        """Process HTML table.
        
        Args:
            html_content (str): HTML content containing table
            
        Returns:
            Document: Document containing table content
        """
        return self.table_processor.process_html_table(html_content)
    
    def extract_tables_from_document(self, document_content: str) -> List[Document]:
        """Extract tables from document content.
        
        Args:
            document_content (str): Document content that may contain tables
            
        Returns:
            List[Document]: List of documents containing extracted tables
        """
        return self.table_processor.extract_tables_from_document(document_content)
    
    def process_multimodal_content(self, content: Dict[str, Any]) -> List[Document]:
        """Process multimodal content from various sources.
        
        Args:
            content (Dict[str, Any]): Dictionary containing multimodal content
            
        Returns:
            List[Document]: List of documents containing processed content
        """
        documents = []
        
        # Process different content types
        if "image_path" in content:
            doc = self.process_image(content["image_path"])
            documents.append(doc)
        
        if "audio_path" in content:
            doc = self.process_audio(content["audio_path"])
            documents.append(doc)
        
        if "video_path" in content:
            doc = self.process_video(content["video_path"])
            documents.append(doc)
        
        if "pdf_path" in content:
            docs = self.process_pdf(content["pdf_path"])
            documents.extend(docs)
        
        if "table_data" in content:
            doc = self.process_table_data(content["table_data"])
            documents.append(doc)
        
        if "html_table" in content:
            doc = self.process_html_table(content["html_table"])
            documents.append(doc)
        
        return documents
