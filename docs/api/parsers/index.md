# nexusrag.parsers

Document parsing components.

## Modules

- [base](base.md): Abstract base class for parsers
- pdf: PDF parsing implementation (coming soon)

## Classes

### Document

```python
Document(content: str, metadata: dict = None)
```

Represents a document with its content and metadata.

**Attributes:**
- `content (str)`: The document content
- `metadata (dict)`: Additional information about the document

#### \_\_init\_\_

```python
__init__(self, content: str, metadata: dict = None)
```

Initialize a Document.

**Args:**
- `content (str)`: The document content
- `metadata (dict, optional)`: Additional information about the document
