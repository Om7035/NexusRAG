# Core Concepts

This document explains the core concepts and architecture of NexusRAG.

## Architecture Overview

NexusRAG follows a modular architecture with four main components:

1. **Parsers**: Extract content and metadata from documents
2. **Embedders**: Convert text into numerical embeddings
3. **Vector Stores**: Store and retrieve embeddings efficiently
4. **LLMs**: Generate responses based on prompts and context

These components are orchestrated by the **Pipeline**, which provides a unified interface for processing documents and answering questions.

## Parsers

Parsers are responsible for extracting content from various document formats. They take a file path as input and return a list of `Document` objects, each containing:

- `content`: The extracted text content
- `metadata`: Additional information about the document (source, page number, etc.)

### Available Parsers

- `PDFParser`: Extracts text and tables from PDF documents
- (More parsers will be added)

### Creating Custom Parsers

To create a custom parser, inherit from `BaseParser` and implement the `parse` method:

```python
from nexusrag.parsers.base import BaseParser, Document

class MyCustomParser(BaseParser):
    def parse(self, file_path: str) -> List[Document]:
        # Your parsing logic here
        pass
```

## Embedders

Embedders convert text into numerical vectors that can be used for similarity comparisons. They take a list of text strings as input and return a list of embeddings.

### Available Embedders

- `SentenceTransformerEmbedder`: Uses pre-trained models from Sentence Transformers
- (More embedders will be added)

### Creating Custom Embedders

To create a custom embedder, inherit from `BaseEmbedder` and implement the `embed` method:

```python
from nexusrag.embedders.base import BaseEmbedder

class MyCustomEmbedder(BaseEmbedder):
    def embed(self, texts: List[str]) -> List[List[float]]:
        # Your embedding logic here
        pass
```

## Vector Stores

Vector stores are responsible for storing embeddings and retrieving similar documents based on a query. They provide efficient similarity search capabilities.

### Available Vector Stores

- `ChromaVectorStore`: Uses ChromaDB for storage and retrieval
- (More vector stores will be added)

### Creating Custom Vector Stores

To create a custom vector store, inherit from `BaseVectorStore` and implement the `add` and `query` methods:

```python
from nexusrag.vectorstores.base import BaseVectorStore

class MyCustomVectorStore(BaseVectorStore):
    def add(self, docs: List[Document]) -> None:
        # Your storage logic here
        pass
    
    def query(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        # Your retrieval logic here
        pass
```

## LLMs

LLMs generate responses based on prompts and context documents. They take a prompt and optional context as input and return a generated response.

### Available LLMs

- `HuggingFaceLLM`: Uses models from Hugging Face
- (More LLMs will be added)

### Creating Custom LLMs

To create a custom LLM, inherit from `BaseLLM` and implement the `generate` method:

```python
from nexusrag.llms.base import BaseLLM

class MyCustomLLM(BaseLLM):
    def generate(self, prompt: str, context: List[Dict[str, Any]] = None) -> str:
        # Your generation logic here
        pass
```

## Pipeline

The Pipeline orchestrates all components and provides a unified interface for processing documents and answering questions. It handles the flow of data between components and manages the overall RAG process.

### Usage

```python
# Initialize pipeline with components
pipeline = RAGPipeline(parser, embedder, vector_store, llm)

# Process documents
pipeline.process_document("document.pdf")

# Ask questions
answer = pipeline.ask("What is this document about?")
```

## Modularity Benefits

The modular architecture of NexusRAG provides several benefits:

1. **Flexibility**: Easily swap components to experiment with different technologies
2. **Extensibility**: Add new components without modifying existing code
3. **Maintainability**: Isolated components are easier to test and debug
4. **Reusability**: Components can be used in different combinations for different use cases
