# nexusrag.embedders.base

Abstract base class for text embedders.

## Classes

### BaseEmbedder

```python
BaseEmbedder()
```

Abstract base class for text embedders.

#### Methods

##### embed

```python
embed(self, texts: List[str]) -> List[List[float]]
```

Generate embeddings for a list of text strings.

This method must be implemented by subclasses.

**Args:**
- `texts (List[str])`: List of text strings to embed

**Returns:**
- `List[List[float]]`: List of embeddings, one for each input text
