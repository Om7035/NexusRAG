# NexusRAG - The "It Just Works" RAG Framework

NexusRAG is an open-source framework that works out of the box but is highly customizable. It combines high-fidelity document parsing with advanced reasoning capabilities to create powerful, data-aware applications.

## 🚀 Quick Start - "It Just Works" Experience

### Option 1: One-Command Docker Setup

```bash
# Clone and run everything with one command
git clone https://github.com/Om7035/NexusRAG.git
cd NexusRAG
docker-compose up

# Open http://localhost:8501 in your browser
```

### Option 2: CLI Usage

```bash
# Install NexusRAG
pip install nexusrag

# Process documents
nexusrag process document.pdf document.docx

# Ask questions
nexusrag ask "What is this document about?"
```

### Option 3: Python Library

```python
from nexusrag import RAG

# Initialize RAG
rag = RAG()

# Process documents
rag.process(["document.pdf", "document.docx"])

# Ask questions
answer = rag.ask("What is this document about?")
print(answer)
```

## 🌟 Key Features

### Multiple Usage Patterns
- **Ready-to-use application**: `docker-compose up` → works!
- **Python library**: `from nexusrag import RAG`
- **REST API**: For web apps/mobile apps
- **CLI tool**: `nexusrag query "your question"`

### Advanced Capabilities
- **Multimodal Processing**: Text, PDF, images, audio, video, tables
- **Smart Chunking**: Character, semantic, and sentence-based strategies
- **Hybrid Search**: Vector + keyword search with re-ranking
- **Multi-Step Reasoning**: Iterative refinement with citations
- **Local LLM Support**: Full Ollama integration
- **Configurable Components**: Swap models, databases, and processors

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- Docker (for one-command deployment)

### Quick Installation

```bash
# Clone the repository
git clone https://github.com/Om7035/NexusRAG.git
cd NexusRAG

# Install with pip
pip install -e .

# Or install with all dependencies
pip install -e .[all]
```

## 🛠️ Usage Patterns

### 1. Web Application

```bash
# Run the Streamlit web app
streamlit run app.py

# Or with Docker
docker-compose up
```

### 2. Python Library

```python
from nexusrag import RAG

# Initialize with default settings
rag = RAG()

# Or configure components
rag = RAG(
    embedder="sentence-transformers",
    vector_store="chroma",
    llm="ollama"
)

# Process documents
rag.process(["document1.pdf", "document2.docx"])

# Ask questions
answer = rag.ask("Summarize the key points")
print(answer)
```

### 3. Command Line Interface

```bash
# Process documents
nexusrag process document.pdf document.docx

# Ask questions
nexusrag ask "What is the main topic?"

# Show configuration
nexusrag config

# Get version
nexusrag version
```

### 4. REST API

```bash
# Start the API server
nexusrag serve --port 8000

# Process documents (POST /process)
curl -X POST http://localhost:8000/process \
  -F "files=@document.pdf"

# Ask questions (POST /ask)
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this about?"}'
```

## ⚙️ Configuration

NexusRAG uses a simple YAML configuration system:

```yaml
# nexusrag.yaml
pipeline:
  chunk_size: 1000
  chunk_overlap: 200

components:
  embedder:
    type: sentence-transformers
    model: all-MiniLM-L6-v2
  
  vector_store:
    type: chroma
    collection_name: nexusrag
  
  llm:
    type: ollama
    model: llama2
```

## 🧩 Modular Architecture

NexusRAG follows a modular architecture that makes it easy to extend:

1. **Document Processing Layer**: Universal file loader, smart chunking, metadata extraction
2. **Multi-Modal Understanding Layer**: Image understanding, audio/video transcription, table comprehension
3. **Retrieval Engine Layer**: Hybrid search, re-ranking, cross-modal retrieval
4. **Generation & Reasoning Layer**: Local LLM orchestration, multi-step reasoning, citation & verification

## 📚 Examples

See the [examples](examples/) directory for detailed usage examples:

- [Document Processing](examples/document_processing_example.py)
- [Multimodal Understanding](examples/multimodal_example.py)
- [Retrieval Engine](examples/retrieval_example.py)
- [Generation & Reasoning](examples/generation_reasoning_example.py)

## 🤝 Contributing

We welcome contributions! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 🎯 Why NexusRAG?

NexusRAG is designed to be:

- **Easy to use**: "It just works" out of the box
- **Highly customizable**: Modular architecture for advanced users
- **Production ready**: Docker deployment, configuration management
- **Community friendly**: Clear contribution guidelines, good documentation

Compared to other RAG frameworks, NexusRAG offers:
- More advanced multi-modal capabilities
- Sophisticated reasoning with citations
- Local LLM support with Ollama
- Comprehensive evaluation tools
- Better documentation and examples
