# NexusRAG

NexusRAG is an open-source framework for building autonomous AI agents that reason over complex, multimodal data. It combines high-fidelity document parsing with a fully modular architecture, enabling developers to create powerful, data-aware applications.

## Features

- **Multimodal Parsing**: High-fidelity extraction from PDFs, Word documents, HTML, Markdown, and other formats
- **Modular Design**: Pluggable components for parsers, embedders, vector stores, and LLMs
- **Agent-Ready**: Built for creating autonomous AI agents that can reason over complex data
- **Extensible**: Easy to add new components and customize existing ones
- **Multiple Providers**: Support for OpenAI, Cohere, Pinecone, Weaviate, Anthropic, Google Gemini, and more
- **Free API Support**: Built-in support for free APIs like Google Gemini
- **Local LLM Support**: Built-in support for local LLMs via Ollama
- **Knowledge Graph**: Entity extraction and relationship mapping (future enhancement)
- **Advanced Features**: Document chunking, metadata filtering, and multimodal processing

## Architecture

NexusRAG follows a modular architecture with several key components:

1. **Parsers**: Extract content and metadata from documents (PDF, Word, HTML, Markdown, Text)
2. **Embedders**: Convert text into numerical embeddings (Sentence Transformers, OpenAI, Cohere, Google Gemini)
3. **Vector Stores**: Store and retrieve embeddings efficiently (Chroma, Pinecone, Weaviate)
4. **LLMs**: Generate responses based on prompts and context (Hugging Face, OpenAI, Anthropic, Google Gemini, Ollama)
5. **Chunking**: Break large documents into smaller, manageable pieces
6. **Metadata Filtering**: Filter documents based on metadata criteria
7. **Multimodal Processing**: Handle text, images, and tables within documents

These components are orchestrated by the **Pipeline**, which provides a unified interface for processing documents and answering questions.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation Steps

1. Clone the repository:

```bash
git clone https://github.com/your-username/NexusRAG.git
cd NexusRAG
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package in development mode:

```bash
pip install -e .
```

This will install all dependencies specified in `pyproject.toml`.

## Quickstart

### Using the Streamlit Demo

NexusRAG includes an advanced Streamlit demo application that you can run locally:

```bash
streamlit run app.py
```

This will start a web server where you can upload multiple document types and ask questions about them.

### Using the Python API

```python
from nexusrag.enhanced_pipeline import EnhancedRAGPipeline
from nexusrag.parsers.universal import UniversalParser
from nexusrag.embedders.universal import UniversalEmbedder
from nexusrag.vectorstores.universal import UniversalVectorStore
from nexusrag.llms.universal import UniversalLLM

# Initialize components
parser = UniversalParser()
embedder = UniversalEmbedder(provider="sentence-transformers")
vector_store = UniversalVectorStore(provider="chroma")
llm = UniversalLLM(provider="huggingface")

# Initialize the enhanced pipeline
pipeline = EnhancedRAGPipeline(
    parser=parser,
    embedder=embedder,
    vector_store=vector_store,
    llm=llm,
    chunk_size=1000,
    chunk_overlap=200
)

# Process documents
pipeline.process_documents(["path/to/your/document.pdf", "path/to/your/document.docx"])

# Ask questions about the documents
answer = pipeline.ask("What is this document about?")
print(answer)

# Ask questions with metadata filtering
filtered_answer = pipeline.ask(
    "What is this document about?",
    filter_metadata={"content_type": "paragraph"}
)
print(filtered_answer)
```

## Components

### Parsers

- `UniversalParser`: Automatically detects file type and uses appropriate parser
- `PDFParser`: Extracts text and tables from PDF documents
- `WordParser`: Extracts text and tables from Word documents
- `HTMLParser`: Extracts text from HTML documents
- `MarkdownParser`: Extracts text from Markdown documents
- `TextParser`: Extracts text from plain text files

### Embedders

- `UniversalEmbedder`: Supports multiple embedding providers
- `SentenceTransformerEmbedder`: Uses pre-trained models from Sentence Transformers
- `OpenAIEmbedder`: Uses OpenAI's embedding models
- `CohereEmbedder`: Uses Cohere's embedding models
- `GeminiEmbedder`: Uses Google Gemini's embedding models

### Vector Stores

- `UniversalVectorStore`: Supports multiple vector store providers
- `ChromaVectorStore`: Uses ChromaDB for storage and retrieval
- `PineconeVectorStore`: Uses Pinecone for storage and retrieval
- `WeaviateVectorStore`: Uses Weaviate for storage and retrieval

### LLMs

- `UniversalLLM`: Supports multiple LLM providers
- `HuggingFaceLLM`: Uses models from Hugging Face
- `OpenAILLM`: Uses OpenAI's language models
- `AnthropicLLM`: Uses Anthropic's language models
- `GeminiLLM`: Uses Google Gemini's language models
- `OllamaLLM`: Uses local LLMs via Ollama

### Advanced Features

- **Document Chunking**: Break large documents into smaller pieces
- **Metadata Filtering**: Filter documents based on metadata criteria
- **Multimodal Processing**: Handle text, images, and tables within documents

## Examples

See the [examples](examples/) directory for detailed usage examples:

- [Basic Usage](examples/basic_usage.py): Simple example with default components
- [Advanced Usage](examples/advanced_usage.py): Example with all advanced features
- [Custom Components](examples/custom_components.py): Example of creating custom components

## Documentation

For more detailed information, please see the [documentation](docs/).

## Contributing

We welcome contributions! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to get started.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
