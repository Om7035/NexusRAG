from typing import List
from .base import BaseParser, Document


class TextParser(BaseParser):
    """Plain text document parser."""
    
    def parse(self, file_path: str) -> List[Document]:
        """Parse a plain text document and return a list of Document objects.
        
        Args:
            file_path (str): Path to the text file to parse
            
        Returns:
            List[Document]: List of parsed documents
        """
        # Read the text file
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Split into paragraphs (separated by double newlines)
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        # Create Document objects
        documents = []
        for i, paragraph in enumerate(paragraphs):
            # Only include substantial paragraphs
            if len(paragraph) > 50:
                document = Document(
                    content=paragraph,
                    metadata={
                        "source": file_path,
                        "content_type": "paragraph",
                        "paragraph_index": i
                    }
                )
                documents.append(document)
        
        # If no substantial paragraphs found, treat entire document as one
        if not documents:
            document = Document(
                content=content.strip(),
                metadata={
                    "source": file_path,
                    "content_type": "document"
                }
            )
            documents.append(document)
            
        return documents
