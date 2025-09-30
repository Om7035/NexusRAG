from typing import List
from .base import BaseParser, Document
from bs4 import BeautifulSoup


class HTMLParser(BaseParser):
    """HTML document parser using BeautifulSoup."""
    
    def parse(self, file_path: str) -> List[Document]:
        """Parse an HTML document and return a list of Document objects.
        
        Args:
            file_path (str): Path to the HTML file to parse
            
        Returns:
            List[Document]: List of parsed documents
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError(
                "To use HTMLParser, you need to install the beautifulsoup4 library. "
                "Please run: pip install beautifulsoup4"
            )
        
        # Read and parse the HTML file
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        soup = BeautifulSoup(content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extract title
        title = soup.find('title')
        title_text = title.get_text().strip() if title else ""
        
        # Extract main content
        # Try to find main content areas
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=['content', 'main', 'article'])
        
        if main_content:
            # If we found a main content area, extract from there
            text_content = main_content.get_text(separator='\n', strip=True)
        else:
            # Otherwise, extract from the body
            body = soup.find('body')
            if body:
                text_content = body.get_text(separator='\n', strip=True)
            else:
                # Fallback to entire document
                text_content = soup.get_text(separator='\n', strip=True)
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in text_content.split('\n') if p.strip()]
        
        # Create Document objects
        documents = []
        
        # Add title as a separate document
        if title_text:
            title_doc = Document(
                content=title_text,
                metadata={
                    "source": file_path,
                    "content_type": "title"
                }
            )
            documents.append(title_doc)
        
        # Add paragraphs
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph) > 20:  # Only include substantial paragraphs
                paragraph_doc = Document(
                    content=paragraph,
                    metadata={
                        "source": file_path,
                        "content_type": "paragraph",
                        "paragraph_index": i
                    }
                )
                documents.append(paragraph_doc)
        
        return documents
