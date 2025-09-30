from typing import List
from .base import BaseParser, Document


class MarkdownParser(BaseParser):
    """Markdown document parser."""
    
    def parse(self, file_path: str) -> List[Document]:
        """Parse a Markdown document and return a list of Document objects.
        
        Args:
            file_path (str): Path to the Markdown file to parse
            
        Returns:
            List[Document]: List of parsed documents
        """
        # Read the Markdown file
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Split into sections by headers
        sections = self._split_by_headers(content)
        
        # Create Document objects
        documents = []
        for i, (header, section_content) in enumerate(sections):
            # Combine header and content
            full_content = f"{header}\n\n{section_content}" if header else section_content
            
            document = Document(
                content=full_content.strip(),
                metadata={
                    "source": file_path,
                    "content_type": "section",
                    "section_index": i,
                    "section_title": header if header else "Untitled Section"
                }
            )
            documents.append(document)
            
        # If no sections found, treat entire document as one
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
    
    def _split_by_headers(self, content: str) -> List[tuple]:
        """Split markdown content by headers.
        
        Args:
            content (str): Markdown content
            
        Returns:
            List[tuple]: List of (header, content) tuples
        """
        lines = content.split('\n')
        sections = []
        current_header = ""
        current_content = []
        
        for line in lines:
            # Check if line is a header (starts with #)
            if line.strip().startswith('#'):
                # Save previous section if it exists
                if current_header or current_content:
                    sections.append((current_header, '\n'.join(current_content).strip()))
                
                # Start new section
                current_header = line.strip()
                current_content = []
            else:
                current_content.append(line)
        
        # Add the last section
        if current_header or current_content:
            sections.append((current_header, '\n'.join(current_content).strip()))
            
        return sections
