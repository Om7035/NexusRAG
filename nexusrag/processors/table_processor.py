from typing import List, Dict, Any
import pandas as pd
from ..parsers.base import Document


class TableProcessor:
    """Process tables from documents with advanced extraction capabilities."""
    
    def __init__(self):
        pass
    
    def extract_tables(self, document: Document) -> List[Dict[str, Any]]:
        """Extract tables from document content.
        
        Args:
            document (Document): Document to extract tables from
            
        Returns:
            List[Dict[str, Any]]: List of extracted tables with metadata
        """
        # This is a simplified implementation
        # In a production environment, you would use libraries like camelot or tabula
        tables = []
        
        # Look for table-like patterns in text
        lines = document.content.split('\n')
        table_data = []
        
        for line in lines:
            # Simple heuristic: lines with multiple tabs or pipes
            if '\t' in line or '|' in line:
                # Split by delimiter
                if '|' in line:
                    columns = [col.strip() for col in line.split('|') if col.strip()]
                else:
                    columns = [col.strip() for col in line.split('\t')]
                table_data.append(columns)
        
        if table_data:
            tables.append({
                'data': table_data,
                'source_document': document.metadata.get('source', 'unknown'),
                'page': document.metadata.get('page', 0),
                'type': 'extracted_table'
            })
        
        return tables
    
    def convert_to_structured(self, table_data: List[List[str]]) -> pd.DataFrame:
        """Convert extracted table data to structured format.
        
        Args:
            table_data (List[List[str]]): Raw table data
            
        Returns:
            pd.DataFrame: Structured table as DataFrame
        """
        if not table_data:
            return pd.DataFrame()
        
        # Use first row as headers if it looks like headers
        if len(table_data) > 1:
            # Simple heuristic: first row has fewer non-numeric values
            first_row = table_data[0]
            if len(first_row) > 0 and all(isinstance(item, str) for item in first_row):
                headers = first_row
                data_rows = table_data[1:]
                try:
                    return pd.DataFrame(data_rows, columns=headers)
                except ValueError:
                    # If column count doesn't match, use default indexing
                    return pd.DataFrame(table_data)
        
        # If no clear headers, use default indexing
        return pd.DataFrame(table_data)
    
    def extract_structured_data(self, document: Document) -> List[pd.DataFrame]:
        """Extract all structured data from a document.
        
        Args:
            document (Document): Document to extract structured data from
            
        Returns:
            List[pd.DataFrame]: List of structured data tables
        """
        tables = self.extract_tables(document)
        dataframes = []
        
        for table in tables:
            df = self.convert_to_structured(table['data'])
            dataframes.append(df)
        
        return dataframes
