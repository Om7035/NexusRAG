# Advanced Features

NexusRAG includes several advanced features that make it a powerful framework for building AI applications.

## Document Chunking

Document chunking is the process of breaking large documents into smaller, manageable pieces. This is useful for:

1. Staying within token limits of embedding models and LLMs
2. Improving retrieval precision by matching specific parts of documents
3. Reducing processing time for large documents

### How It Works

The `DocumentChunker` class splits documents based on character count with configurable overlap:

```python
from nexusrag.chunking import DocumentChunker
from nexusrag.parsers.base import Document

# Create a chunker with custom settings
chunker = DocumentChunker(chunk_size=500, chunk_overlap=100)

doc = Document(content="Very long document content...", metadata={"source": "example.txt"})

# Split document into chunks
chunks = chunker.chunk_document(doc)

# Split multiple documents
documents = [doc1, doc2, doc3]
chunked_documents = chunker.chunk_documents(documents)
```

### Configuration

- `chunk_size`: Maximum size of each chunk in characters (default: 1000)
- `chunk_overlap`: Number of characters to overlap between chunks (default: 200)

## Metadata Filtering

Metadata filtering allows you to search within specific subsets of your documents based on metadata criteria.

### How It Works

The `MetadataFilter` class provides several filtering methods:

```python
from nexusrag.metadata_filter import MetadataFilter
from nexusrag.parsers.base import Document

# Create sample documents with metadata
doc1 = Document("Content 1", {"source": "file1.txt", "content_type": "paragraph"})
doc2 = Document("Content 2", {"source": "file2.pdf", "content_type": "table"})
doc3 = Document("Content 3", {"source": "file1.txt", "content_type": "paragraph"})

documents = [doc1, doc2, doc3]

# Filter by source
filtered_by_source = MetadataFilter.filter_by_source(documents, "file1.txt")

# Filter by content type
filtered_by_type = MetadataFilter.filter_by_content_type(documents, "paragraph")

# Filter by custom field
filtered_by_field = MetadataFilter.filter_by_custom_field(documents, "content_type", "table")

# Filter with custom function
filtered_custom = MetadataFilter.filter_by_metadata(
    documents, 
    lambda metadata: metadata.get("content_type") == "paragraph" and "file1" in metadata.get("source", "")
)
```

### Usage in Pipeline

The `EnhancedRAGPipeline` supports metadata filtering in the `ask` method:

```python
from nexusrag.enhanced_pipeline import EnhancedRAGPipeline

# Initialize pipeline
pipeline = EnhancedRAGPipeline(parser, embedder, vector_store, llm)

# Ask with metadata filtering
answer = pipeline.ask(
    "What is the main topic?",
    filter_metadata={"content_type": "paragraph", "source": "important_doc.pdf"}
)
```

## Multimodal Processing

Multimodal processing allows NexusRAG to handle different types of content within documents, including text, images, and tables.

### Supported Modalities

1. **Text**: Standard text content
2. **Images**: OCR processing to extract text from images
3. **Tables**: Structured data in tabular format

### How It Works

The `MultimodalProcessor` class handles different content types:

```python
from nexusrag.multimodal import MultimodalProcessor

processor = MultimodalProcessor()

# Process an image file
image_doc = processor.process_image("diagram.png")

# Process table data
table_data = [
    ["Name", "Age", "City"],
    ["Alice", "30", "New York"],
    ["Bob", "25", "Los Angeles"]
]
table_doc = processor.process_table(table_data)

# Process a multimodal document
# Automatically detects file type and processes accordingly
documents = processor.process_multimodal_document("complex_document.pdf")
```

### Requirements

- **Images**: Requires `Pillow` and `pytesseract` libraries
- **OCR**: Requires Tesseract OCR engine to be installed on the system

## Universal Components

NexusRAG provides universal components that can work with multiple providers.

### Universal Parser

The `UniversalParser` automatically detects file types and uses the appropriate parser:

```python
from nexusrag.parsers.universal import UniversalParser

parser = UniversalParser()

# Automatically uses PDFParser for .pdf files
documents = parser.parse("document.pdf")

# Automatically uses WordParser for .docx files
documents = parser.parse("document.docx")

# Automatically uses HTMLParser for .html files
documents = parser.parse("document.html")
```

### Universal Embedder

The `UniversalEmbedder` supports multiple embedding providers:

```python
from nexusrag.embedders.universal import UniversalEmbedder

# Use Sentence Transformers (free, local)
embedder = UniversalEmbedder(provider="sentence-transformers", model_name="all-MiniLM-L6-v2")

# Use OpenAI (paid)
embedder = UniversalEmbedder(provider="openai", model_name="text-embedding-ada-002")

# Use Cohere (paid)
embedder = UniversalEmbedder(provider="cohere", model_name="embed-english-v3.0")

# Use Google Gemini (free tier available)
embedder = UniversalEmbedder(provider="gemini", model_name="models/embedding-001")

# Generate embeddings
embeddings = embedder.embed(["Text to embed"])
```

### Universal Vector Store

The `UniversalVectorStore` supports multiple vector store backends:

```python
from nexusrag.vectorstores.universal import UniversalVectorStore

# Use Chroma (free, local)
vector_store = UniversalVectorStore(provider="chroma")

# Use Pinecone (paid)
vector_store = UniversalVectorStore(provider="pinecone")

# Use Weaviate (paid)
vector_store = UniversalVectorStore(provider="weaviate")

# Add documents
vector_store.add(documents)

# Query documents
results = vector_store.query("search text")
```

### Universal LLM

The `UniversalLLM` supports multiple language model providers:

```python
from nexusrag.llms.universal import UniversalLLM

# Use Hugging Face (free, local)
llm = UniversalLLM(provider="huggingface", model_name="google/flan-t5-base")

# Use OpenAI (paid)
llm = UniversalLLM(provider="openai", model_name="gpt-3.5-turbo")

# Use Anthropic (paid)
llm = UniversalLLM(provider="anthropic", model_name="claude-3-haiku-20240307")

# Use Google Gemini (free tier available)
llm = UniversalLLM(provider="gemini", model_name="gemini-pro")

# Use Ollama (free, local)
llm = UniversalLLM(provider="ollama", model_name="llama2")

# Generate response
response = llm.generate("Prompt text", context=documents)
```

## Environment Variables

NexusRAG uses environment variables for API keys and configuration:

```bash
# OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# Cohere
export COHERE_API_KEY="your-cohere-api-key"

# Pinecone
export PINECONE_API_KEY="your-pinecone-api-key"
export PINECONE_ENVIRONMENT="your-pinecone-environment"

# Weaviate
export WEAVIATE_HOST="http://localhost:8080"

# Anthropic
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# Google Gemini
export GEMINI_API_KEY="your-google-ai-key"

# Ollama (optional, if using custom host)
export OLLAMA_HOST="http://localhost:11434"
```

## Knowledge Graph Features

NexusRAG includes enhanced knowledge graph support with automatic entity extraction and relationship discovery.

### Current Implementation

The current implementation includes:

1. **Entity and Relationship Models**: Classes for representing entities and relationships
2. **Knowledge Graph Builder**: Component that automatically builds knowledge graphs from documents
3. **Entity Extraction**: Automatic extraction of persons, organizations, and technologies
4. **Relationship Discovery**: Identification of relationships between entities
5. **Entity Search**: Search functionality for finding entities by name
6. **Integration with Pipeline**: Enhanced pipeline that automatically builds knowledge graphs

### Usage Example

```python
from nexusrag.enhanced_pipeline import EnhancedRAGPipeline

# Initialize pipeline with knowledge graph support
pipeline = EnhancedRAGPipeline(parser, embedder, vector_store, llm)

# Process documents (automatically builds knowledge graph)
pipeline.process_documents(["doc1.pdf", "doc2.pdf"])

# Access the knowledge graph
if pipeline.knowledge_graph:
    print(f"Entities: {len(pipeline.knowledge_graph.entities)}")
    print(f"Relationships: {len(pipeline.knowledge_graph.relationships)}")
    
    # Search for entities
    entities = pipeline.knowledge_graph.search_entities("Apple")
    print(f"Found {len(entities)} entities matching 'Apple'")

# Ask questions with knowledge graph enhancement
answer = pipeline.ask(
    "How are these companies related?",
    use_knowledge_graph=True
)
```

## Agentic Features

NexusRAG now includes basic agentic capabilities with multi-step reasoning.

### Current Implementation

The current implementation includes:

1. **BasicAgent**: Agent with reasoning and tool usage capabilities
2. **Multi-step Reasoning**: Iterative refinement of answers
3. **Memory Management**: Storage and retrieval of reasoning steps
4. **Tool Integration**: Extensible tool system

### Usage Example

```python
from nexusrag.enhanced_pipeline import EnhancedRAGPipeline

# Initialize pipeline with agent support
pipeline = EnhancedRAGPipeline(parser, embedder, vector_store, llm)

# Process documents
pipeline.process_documents(["doc1.pdf", "doc2.pdf"])

# Ask questions with multi-step reasoning
answer = pipeline.ask_with_reasoning(
    "Compare the leadership changes at these companies",
    max_steps=3
)
print(answer)
```

## Advanced Document Parsing

NexusRAG includes advanced document parsing capabilities with layout analysis.

### Current Implementation

The current implementation includes:

1. **AdvancedPDFParser**: PDF parser with layout analysis
2. **Structured Content Extraction**: Preservation of document hierarchy
3. **Image Detection**: Identification of image blocks
4. **Metadata Preservation**: Detailed metadata for each content block

### Usage Example

```python
from nexusrag.parsers.advanced_pdf import AdvancedPDFParser

# Initialize advanced PDF parser
parser = AdvancedPDFParser()

# Parse PDF with layout analysis
documents = parser.parse("document.pdf")

# Access detailed metadata
for doc in documents:
    print(f"Page: {doc.metadata['page']}")
    print(f"Block type: {doc.metadata['block_type']}")
    print(f"Content: {doc.content[:100]}...")
```

## Table Processing

NexusRAG includes table processing capabilities for structured data extraction.

### Current Implementation

The current implementation includes:

1. **TableProcessor**: Extracts tables from documents
2. **Structured Data Conversion**: Converts to pandas DataFrames
3. **Delimiter Detection**: Handles various table formats
4. **Metadata Preservation**: Preserves table source information

### Usage Example

```python
from nexusrag.processors.table_processor import TableProcessor
from nexusrag.parsers.base import Document

# Initialize table processor
table_processor = TableProcessor()

# Create a document with table-like content
doc = Document(
    content="| Name | Age | City |\n|------|-----|------|\n| John | 30  | NYC  |",
    metadata={"source": "sample"}
)

# Extract tables
tables = table_processor.extract_tables(doc)

# Convert to structured format
if tables:
    df = table_processor.convert_to_structured(tables[0]['data'])
    print(df)
```

## Performance Considerations

When using advanced features, consider the following performance factors:

1. **Document Chunking**: Smaller chunks improve retrieval precision but increase storage requirements
2. **Embedding Models**: Cloud-based models (OpenAI, Cohere) may have rate limits and costs
3. **Vector Stores**: Cloud-based stores (Pinecone, Weaviate) offer scalability but may have latency
4. **LLMs**: Cloud-based models (OpenAI, Anthropic) offer powerful capabilities but may have costs and rate limits

## Best Practices

1. **Choose appropriate chunk sizes** based on your use case and model limitations
2. **Use metadata filtering** to narrow down search results and improve accuracy
3. **Select the right providers** based on your requirements for cost, performance, and capabilities
4. **Handle API keys securely** using environment variables or secure configuration management
5. **Test with representative data** to optimize chunking and filtering parameters
