# Quickstart

This guide will help you get started with NexusRAG quickly.

## Basic Usage

Here's a simple example of how to use NexusRAG:

```python
from nexusrag.pipeline import RAGPipeline
# TODO: Import default components when implemented

# Initialize the pipeline with default components
# pipeline = RAGPipeline(parser, embedder, vector_store, llm)

# Process a document
# pipeline.process_document("path/to/your/document.pdf")

# Ask questions about the document
# answer = pipeline.ask("What is this document about?")
# print(answer)
```

## Using Custom Components

NexusRAG's modular design allows you to easily swap components:

```python
from nexusrag.pipeline import RAGPipeline
from nexusrag.parsers import MyCustomParser
from nexusrag.embedders import MyCustomEmbedder
from nexusrag.vectorstores import MyCustomVectorStore
from nexusrag.llms import MyCustomLLM

# Initialize custom components
parser = MyCustomParser()
embedder = MyCustomEmbedder()
vector_store = MyCustomVectorStore()
llm = MyCustomLLM()

# Initialize pipeline with custom components
pipeline = RAGPipeline(parser, embedder, vector_store, llm)

# Use the pipeline as before
# pipeline.process_document("path/to/your/document.pdf")
# answer = pipeline.ask("What is this document about?")
```

## Running the Demo Application

NexusRAG includes a Streamlit demo application that you can run locally:

```bash
streamlit run app.py
```

This will start a web server where you can upload documents and ask questions about them.

## Next Steps

- Check out the [API Reference](api/) for detailed information about classes and methods
- Learn about [Core Concepts](concepts.md)
- See how to [Contribute](contributing.md) to the project
