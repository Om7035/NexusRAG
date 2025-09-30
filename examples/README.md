# NexusRAG Examples

This directory contains example scripts demonstrating how to use the NexusRAG framework.

## Basic Usage

The [basic_usage.py](basic_usage.py) script demonstrates how to use NexusRAG with its default components:

- PDF parsing with `PDFParser`
- Text embedding with `SentenceTransformerEmbedder`
- Vector storage with `ChromaVectorStore`
- Language modeling with `HuggingFaceLLM`

To run this example:

```bash
python examples/basic_usage.py
```

## Advanced Usage

The [advanced_usage.py](advanced_usage.py) script demonstrates the advanced features of NexusRAG:

- Universal parsers for multiple document types (PDF, Word, HTML, Markdown)
- Universal embedders with multiple providers (Sentence Transformers, OpenAI, Cohere)
- Universal vector stores with multiple backends (Chroma, Pinecone, Weaviate)
- Universal LLMs with multiple providers (Hugging Face, OpenAI, Anthropic)
- Document chunking
- Metadata filtering
- Multimodal processing

To run this example:

```bash
python examples/advanced_usage.py
```

## Comprehensive Example

The [comprehensive_example.py](comprehensive_example.py) script demonstrates all features of NexusRAG:

- All advanced features from advanced_usage.py
- Detailed demonstrations of individual components
- Complete pipeline workflow

To run this example:

```bash
python examples/comprehensive_example.py
```

## Free API Example

The [free_api_example.py](free_api_example.py) script demonstrates how to use free APIs like Google Gemini:

- Using Google Gemini for embeddings and language modeling
- No cost experimentation and development
- Easy switching between free and paid providers

To run this example:

```bash
export GEMINI_API_KEY="your-google-ai-key"
python examples/free_api_example.py
```

## Multimodal Processing Example (Coming Soon)

The [multimodal_example.py](multimodal_example.py) script will demonstrate the enhanced multimodal processing capabilities of NexusRAG:

- Image understanding & captioning (OCR fallback)
- Audio/video transcription (with Whisper)
- Table & chart comprehension
- PDF with math/formula understanding (with Nougat)
- Universal multimodal processing

*Note: This example is temporarily disabled due to import issues.*

## Document Processing Example

The [document_processing_example.py](document_processing_example.py) script demonstrates the enhanced document processing capabilities of NexusRAG:

- Universal file parsing (text, PDF, images, audio, video)
- Advanced PDF parsing with layout analysis
- Smart chunking strategies (character, semantic, sentence)
- Comprehensive metadata extraction

To run this example:

```bash
python examples/document_processing_example.py
```

## Comprehensive Example

The [comprehensive_example.py](comprehensive_example.py) script demonstrates all advanced features of NexusRAG:

- Advanced document parsing with layout analysis
- Knowledge graph with entity extraction
- Multi-step reasoning capabilities
- Table processing
- Local LLM support via Ollama

To run this example:

```bash
python examples/comprehensive_example.py
```

## Ollama Example

The [ollama_example.py](ollama_example.py) script demonstrates how to use Ollama for local LLM processing:

- Using local LLMs with complete privacy
- No internet required for processing
- No API costs for inference

To run this example:

```bash
# First install and set up Ollama
# Download from: https://ollama.com/download
# Run: ollama pull llama2

python examples/ollama_example.py
```

## Enhanced Knowledge Graph Example

The [enhanced_knowledge_graph_example.py](enhanced_knowledge_graph_example.py) script demonstrates the enhanced knowledge graph features:

- Automatic entity extraction from documents
- Relationship discovery between entities
- Entity search and querying
- Document-entity linking

To run this example:

```bash
python examples/enhanced_knowledge_graph_example.py
```

## Knowledge Graph Example

The [knowledge_graph_example.py](knowledge_graph_example.py) script demonstrates how to use knowledge graph features:

- Building knowledge graphs from documents
- Enhanced reasoning with entity relationships
- Future-ready architecture for advanced AI features

To run this example:

```bash
python examples/knowledge_graph_example.py
```

## Custom Components

The [custom_components.py](custom_components.py) script demonstrates how to create and use custom components with NexusRAG:

- Custom parser implementation
- Custom embedder implementation
- Custom vector store implementation
- Custom LLM implementation

To run this example:

```bash
python examples/custom_components.py
```

## Creating Your Own Examples

To create your own examples:

1. Create a new Python file in this directory
2. Import the necessary components from `nexusrag`
3. Initialize your components
4. Create a `RAGPipeline` or `EnhancedRAGPipeline` with your components
5. Process documents and ask questions

For more information, see the [documentation](../docs/) and [API reference](../docs/api/).
