from typing import List
from .base import BaseParser, Document


class PDFParser(BaseParser):
    """PDF document parser using the unstructured library."""
    
    def parse(self, file_path: str) -> List[Document]:
        """Parse a PDF document and return a list of Document objects.
        
        Args:
            file_path (str): Path to the PDF file to parse
            
        Returns:
            List[Document]: List of parsed documents
        """
        try:
            from unstructured.partition.pdf import partition_pdf
        except ImportError:
            raise ImportError(
                "To use PDFParser, you need to install the unstructured library. "
                "Please run: pip install unstructured"
            )
        
        # Partition the PDF document
        elements = partition_pdf(filename=file_path)
        
        # Convert elements to Document objects
        documents = []
        for element in elements:
            # Get the text content
            content = str(element)
            
            # Get metadata
            metadata = {
                "source": file_path,
                "page_number": getattr(element, "page_number", None),
                "element_type": type(element).__name__,
            }
            
            # Create Document object
            document = Document(content=content, metadata=metadata)
            documents.append(document)
            
        return documents
