# nexusrag.pipeline

Main pipeline orchestration module.

## Classes

### RAGPipeline

```python
RAGPipeline(parser: BaseParser, embedder: BaseEmbedder, vector_store: BaseVectorStore, llm: BaseLLM)
```

Main RAG pipeline that orchestrates all components.

#### Methods

##### \_\_init\_\_

```python
__init__(self, parser: BaseParser, embedder: BaseEmbedder, vector_store: BaseVectorStore, llm: BaseLLM)
```

Initialize the RAG pipeline with components.

**Args:**
- `parser (BaseParser)`: Document parser
- `embedder (BaseEmbedder)`: Text embedder
- `vector_store (BaseVectorStore)`: Vector store
- `llm (BaseLLM)`: Language model

##### process_document

```python
process_document(self, file_path: str) -> None
```

Process a document file and add it to the vector store.

**Args:**
- `file_path (str)`: Path to the document file

##### ask

```python
ask(self, question: str) -> str
```

Ask a question about the processed documents.

**Args:**
- `question (str)`: Question to ask

**Returns:**
- `str`: Answer to the question
