from typing import List, Dict, Any
from ..parsers.base import Document
import pandas as pd


class TableProcessor:
    """Table processor for structured data extraction and comprehension."""
    
    def __init__(self):
        """Initialize the table processor."""
        pass
    
    def process_table_data(self, table_data: List[List[str]]) -> Document:
        """Process table data and convert to structured format.
        
        Args:
            table_data (List[List[str]]): 2D list representing table data
            
        Returns:
            Document: Document containing table as structured text
        """
        # Convert to DataFrame for easier processing
        try:
            df = pd.DataFrame(table_data[1:], columns=table_data[0])
        except Exception:
            # If conversion fails, treat as simple text table
            table_text = ""
            for row in table_data:
                table_text += "\t".join(str(cell) for cell in row) + "\n"
            
            metadata = {
                "content_type": "table",
                "media_type": "text",
                "processing_method": "text"
            }
            
            return Document(content=table_text.strip(), metadata=metadata)
        
        # Convert DataFrame to structured text
        table_text = df.to_string(index=False)
        
        # Create document
        metadata = {
            "content_type": "table",
            "media_type": "text",
            "processing_method": "dataframe",
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns)
        }
        
        return Document(content=table_text.strip(), metadata=metadata)
    
    def process_html_table(self, html_content: str) -> Document:
        """Process HTML table and convert to structured format.
        
        Args:
            html_content (str): HTML content containing table
            
        Returns:
            Document: Document containing table as structured text
        """
        try:
            import pandas as pd
            from io import StringIO
        except ImportError:
            raise ImportError(
                "To process HTML tables, you need to install pandas. "
                "Please run: pip install pandas"
            )
        
        try:
            # Read HTML tables
            tables = pd.read_html(StringIO(html_content))
            
            if not tables:
                raise ValueError("No tables found in HTML content")
            
            # Use the first table
            df = tables[0]
            
            # Convert to structured text
            table_text = df.to_string(index=False)
            
            # Create document
            metadata = {
                "content_type": "table",
                "media_type": "text",
                "processing_method": "html",
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": list(df.columns)
            }
            
            return Document(content=table_text.strip(), metadata=metadata)
            
        except Exception as e:
            # Fallback to simple text processing
            metadata = {
                "content_type": "table",
                "media_type": "text",
                "processing_method": "html-fallback",
                "error": str(e)
            }
            
            return Document(content=html_content.strip(), metadata=metadata)
    
    def extract_tables_from_document(self, document_content: str) -> List[Document]:
        """Extract tables from document content.
        
        Args:
            document_content (str): Document content that may contain tables
            
        Returns:
            List[Document]: List of documents containing extracted tables
        """
        documents = []
        
        # Simple table detection based on delimiters
        lines = document_content.split('\n')
        table_lines = []
        in_table = False
        
        for line in lines:
            # Simple heuristic: lines with multiple tabs or pipes might be tables
            if '\t' in line or '|' in line:
                if not in_table:
                    in_table = True
                    table_lines = [line]
                else:
                    table_lines.append(line)
            else:
                if in_table:
                    # Process collected table lines
                    if len(table_lines) > 1:  # Need at least header and one row
                        # Try to parse as table
                        table_data = []
                        for table_line in table_lines:
                            # Split by tabs or pipes
                            if '\t' in table_line:
                                row = table_line.split('\t')
                            elif '|' in table_line:
                                # Remove leading/trailing pipes and split
                                row = [cell.strip() for cell in table_line.strip('|').split('|')]
                            else:
                                row = [table_line]
                            
                            # Clean up cells
                            row = [cell.strip() for cell in row]
                            table_data.append(row)
                        
                        # Process table data
                        if table_data:
                            doc = self.process_table_data(table_data)
                            documents.append(doc)
                    
                    in_table = False
                    table_lines = []
        
        # Handle case where document ends with a table
        if in_table and len(table_lines) > 1:
            table_data = []
            for table_line in table_lines:
                if '\t' in table_line:
                    row = table_line.split('\t')
                elif '|' in table_line:
                    row = [cell.strip() for cell in table_line.strip('|').split('|')]
                else:
                    row = [table_line]
                
                row = [cell.strip() for cell in row]
                table_data.append(row)
            
            if table_data:
                doc = self.process_table_data(table_data)
                documents.append(doc)
        
        return documents
