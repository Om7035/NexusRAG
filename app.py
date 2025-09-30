import streamlit as st
import tempfile
import os
from dotenv import load_dotenv
from nexusrag.enhanced_pipeline import EnhancedRAGPipeline
from nexusrag.parsers.universal import UniversalParser
from nexusrag.embedders.universal import UniversalEmbedder
from nexusrag.vectorstores.universal import UniversalVectorStore
from nexusrag.llms.universal import UniversalLLM

# Load environment variables
load_dotenv()

st.title("NexusRAG Advanced Demo")

st.markdown("""
Welcome to NexusRAG! This advanced demo allows you to upload multiple document types and ask questions about their content.

Supported document types: PDF, Word (.docx), HTML, Markdown, Text
""")

# Sidebar for configuration
st.sidebar.header("Configuration")

# Embedder selection
embedder_provider = st.sidebar.selectbox(
    "Embedder Provider",
    ["sentence-transformers", "openai", "cohere", "gemini"],
    index=0
)

# Vector store selection
vector_store_provider = st.sidebar.selectbox(
    "Vector Store Provider",
    ["chroma", "pinecone", "weaviate"],
    index=0
)

# LLM selection
llm_provider = st.sidebar.selectbox(
    "LLM Provider",
    ["huggingface", "openai", "anthropic", "gemini", "ollama"],
    index=0
)

# Chunking options
st.sidebar.header("Chunking Options")
chunk_size = st.sidebar.slider("Chunk Size", 100, 2000, 1000)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 500, 200)

# Initialize pipeline with configurable components
@st.cache_resource
def get_pipeline():
    # Initialize components
    parser = UniversalParser()
    embedder = UniversalEmbedder(provider=embedder_provider)
    vector_store = UniversalVectorStore(provider=vector_store_provider)
    llm = UniversalLLM(provider=llm_provider)
    
    # Initialize enhanced pipeline
    pipeline = EnhancedRAGPipeline(
        parser=parser,
        embedder=embedder,
        vector_store=vector_store,
        llm=llm,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return pipeline

pipeline = get_pipeline()

# File uploader (support multiple file types)
uploaded_files = st.file_uploader(
    "Upload documents", 
    type=["pdf", "docx", "html", "htm", "md", "txt"],
    accept_multiple_files=True
)

documents_processed = False

if uploaded_files:
    processed_files = []
    
    for uploaded_file in uploaded_files:
        # Save uploaded file to a temporary location
        file_extension = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            # Process the uploaded file
            with st.spinner(f"Processing {uploaded_file.name}..."):
                pipeline.process_document(tmp_file_path)
            processed_files.append(uploaded_file.name)
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
    
    if processed_files:
        st.success(f"Successfully processed: {', '.join(processed_files)}")
        documents_processed = True
    
    # Question input
    question = st.text_input("Ask a question about the documents:")
    
    # Metadata filtering options
    st.sidebar.header("Metadata Filtering")
    filter_by_content_type = st.sidebar.checkbox("Filter by content type")
    content_type_filter = ""
    if filter_by_content_type:
        content_type_filter = st.sidebar.selectbox(
            "Content Type",
            ["paragraph", "table", "title", "section"]
        )
    
    if question and documents_processed:
        with st.spinner("Generating answer..."):
            try:
                # Apply metadata filter if requested
                filter_metadata = None
                if filter_by_content_type and content_type_filter:
                    filter_metadata = {"content_type": content_type_filter}
                
                answer = pipeline.ask(question, filter_metadata=filter_metadata)
                st.write(answer)
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")
else:
    st.info("Please upload documents to get started.")
