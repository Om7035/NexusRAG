# nexusrag.llms.base

Abstract base class for language models.

## Classes

### BaseLLM

```python
BaseLLM()
```

Abstract base class for large language models.

#### Methods

##### generate

```python
generate(self, prompt: str, context: List[Dict[str, Any]] = None) -> str
```

Generate a response based on a prompt and optional context.

This method must be implemented by subclasses.

**Args:**
- `prompt (str)`: The prompt to generate a response for
- `context (List[Dict[str, Any]])`: Optional context documents

**Returns:**
- `str`: Generated response
