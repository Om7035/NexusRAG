# nexusrag.vectorstores.base

Abstract base class for vector stores.

## Classes

### BaseVectorStore

```python
BaseVectorStore()
```

Abstract base class for vector stores.

#### Methods

##### add

```python
add(self, docs: List[Document]) -> None
```

Add documents to the vector store.

This method must be implemented by subclasses.

**Args:**
- `docs (List[Document])`: List of documents to add

##### query

```python
query(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]
```

Query the vector store for similar documents.

This method must be implemented by subclasses.

**Args:**
- `text (str)`: Query text
- `top_k (int)`: Number of top results to return

**Returns:**
- `List[Dict[str, Any]]`: List of similar documents with scores
