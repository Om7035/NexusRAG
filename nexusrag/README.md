# NexusRAG

NexusRAG is an open-source framework for building autonomous AI agents that reason over complex, multimodal data. It combines high-fidelity document parsing with a fully modular architecture, enabling developers to create powerful, data-aware applications.

## Features

- **Multimodal Parsing**: High-fidelity extraction from PDFs, documents, and other formats
- **Modular Design**: Pluggable components for parsers, embedders, vector stores, and LLMs
- **Agent-Ready**: Built for creating autonomous AI agents that can reason over complex data

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/NexusRAG.git
cd NexusRAG

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quickstart

```python
from nexusrag.pipeline import RAGPipeline

# Initialize the pipeline with default components
pipeline = RAGPipeline()

# Process a document
pipeline.process_document("path/to/your/document.pdf")

# Ask questions about the document
answer = pipeline.ask("What is this document about?")
print(answer)
```

## Contributing

We welcome contributions! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to get started.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
