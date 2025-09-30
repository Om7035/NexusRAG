from typing import List
from .base import BaseParser, Document
import fitz  # PyMuPDF
from ..metadata.extractor import MetadataExtractor


class AdvancedPDFParser(BaseParser):
    """Advanced PDF parser with layout analysis and structured content extraction."""
    
    def __init__(self):
        super().__init__()
    
    def parse(self, file_path: str) -> List[Document]:
        """Parse PDF with advanced layout analysis.
        
        Args:
            file_path (str): Path to the PDF file
            
        Returns:
            List[Document]: List of extracted documents with metadata
        """
        documents = []
        
        # Open PDF
        pdf_document = fitz.open(file_path)
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            
            # Extract text with layout information
            blocks = page.get_text("dict")
            
            # Process text blocks
            for block_idx, block in enumerate(blocks.get("blocks", [])):
                if "lines" in block:  # Text block
                    text_lines = []
                    for line in block.get("lines", []):
                        line_text = ""
                        for span in line.get("spans", []):
                            line_text += span.get("text", "")
                        if line_text.strip():
                            text_lines.append(line_text)
                    
                    if text_lines:
                        text_content = "\n".join(text_lines)
                        document = Document(
                            content=text_content.strip(),
                            metadata={
                                "source": file_path,
                                "page": page_num + 1,
                                "block_type": "text",
                                "block_index": block_idx,
                                "bbox": block.get("bbox", [])
                            }
                        )
                        documents.append(document)
                
                elif "image" in block:  # Image block
                    # Extract image information
                    document = Document(
                        content=f"[Image at page {page_num + 1}, block {block_idx}]",
                        metadata={
                            "source": file_path,
                            "page": page_num + 1,
                            "block_type": "image",
                            "block_index": block_idx,
                            "bbox": block.get("bbox", [])
                        }
                    )
                    documents.append(document)
        
        pdf_document.close()
        
        # Enhance metadata for all documents
        enhanced_documents = []
        for doc in documents:
            enhanced_doc = MetadataExtractor.enhance_document_metadata(doc, file_path)
            enhanced_documents.append(enhanced_doc)
        
        return enhanced_documents
