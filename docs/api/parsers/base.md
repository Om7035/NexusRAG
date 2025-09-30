# nexusrag.parsers.base

Abstract base class for document parsers.

## Classes

### BaseParser

```python
BaseParser()
```

Abstract base class for document parsers.

#### Methods

##### parse

```python
parse(self, file_path: str) -> List[Document]
```

Parse a document file and return a list of Document objects.

This method must be implemented by subclasses.

**Args:**
- `file_path (str)`: Path to the document file to parse

**Returns:**
- `List[Document]`: List of parsed documents
