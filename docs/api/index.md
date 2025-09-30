# API Reference

This is the API reference for NexusRAG.

## Modules

- [nexusrag.pipeline](pipeline.md): Main pipeline orchestration
- [nexusrag.parsers](parsers/): Document parsing components
- [nexusrag.embedders](embedders/): Text embedding components
- [nexusrag.vectorstores](vectorstores/): Vector storage and retrieval components
- [nexusrag.llms](llms/): Language model components

## Core Classes

- `RAGPipeline`: Main pipeline class that orchestrates all components
- `Document`: Represents a document with content and metadata

## Base Classes

All components inherit from abstract base classes that define their interfaces:

- `BaseParser`: Base class for document parsers
- `BaseEmbedder`: Base class for text embedders
- `BaseVectorStore`: Base class for vector stores
- `BaseLLM`: Base class for language models
