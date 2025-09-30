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

# Set page config
st.set_page_config(
    page_title="NexusRAG - Advanced RAG Framework",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
.header {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
}
.feature-card {
    background-color: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="header">', unsafe_allow_html=True)
st.title("🧠 NexusRAG - Advanced RAG Framework")
st.markdown("""
NexusRAG is an open-source framework that works out of the box but is highly customizable. 
It combines high-fidelity document parsing with advanced reasoning capabilities.
""")
st.markdown('</div>', unsafe_allow_html=True)

# Quick start guide
st.subheader("🚀 Quick Start")
with st.expander("How to use NexusRAG", expanded=True):
    st.markdown("""
    1. **Upload documents** using the file uploader below
    2. **Configure components** in the sidebar (optional)
    3. **Ask questions** about your documents
    4. **Get intelligent answers** with citations
    """)

# Feature highlights
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.subheader("📄 Multimodal Support")
    st.markdown("""
    - Text, PDF, Word, HTML, Markdown
    - Images with OCR
    - Audio/Video transcription
    - Tables and structured data
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.subheader("🔍 Advanced Retrieval")
    st.markdown("""
    - Hybrid search (vector + keyword)
    - Re-ranking for precision
    - Cross-modal retrieval
    - Metadata filtering
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.subheader("🤖 Smart Generation")
    st.markdown("""
    - Multi-step reasoning
    - Citation & verification
    - Local LLM support
    - Constrained generation
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")

# Component selection
st.sidebar.subheader("Components")
embedder_provider = st.sidebar.selectbox(
    "Embedder Provider",
    ["sentence-transformers", "openai", "cohere", "gemini"],
    index=0,
    help="Select the embedding model provider"
)

vector_store_provider = st.sidebar.selectbox(
    "Vector Store Provider",
    ["chroma", "pinecone", "weaviate", "qdrant"],
    index=0,
    help="Select the vector database provider"
)

llm_provider = st.sidebar.selectbox(
    "LLM Provider",
    ["huggingface", "openai", "anthropic", "gemini", "ollama"],
    index=4,  # Default to Ollama for local usage
    help="Select the language model provider"
)

# Advanced options
st.sidebar.subheader("Advanced Options")
chunk_size = st.sidebar.slider("Chunk Size", 100, 2000, 1000, help="Size of document chunks")
chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 500, 200, help="Overlap between chunks")

reasoning_steps = st.sidebar.slider("Reasoning Steps", 1, 10, 3, help="Number of reasoning steps")

# Initialize pipeline with configurable components
@st.cache_resource
def get_pipeline():
    try:
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
    except Exception as e:
        st.error(f"Error initializing pipeline: {str(e)}")
        return None

pipeline = get_pipeline()

if pipeline is None:
    st.stop()

# File uploader (support multiple file types)
st.subheader("📁 Upload Documents")
uploaded_files = st.file_uploader(
    "Upload documents to analyze", 
    type=["pdf", "docx", "html", "htm", "md", "txt", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
    help="Supported formats: PDF, Word, HTML, Markdown, Text, Images"
)

documents_processed = False

if uploaded_files:
    st.subheader("📊 Processing Status")
    processed_files = []
    failed_files = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        # Update progress
        progress = (i + 1) / len(uploaded_files)
        progress_bar.progress(progress)
        status_text.text(f"Processing {uploaded_file.name}...")
        
        # Save uploaded file to a temporary location
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            # Process the uploaded file
            pipeline.process_document(tmp_file_path)
            processed_files.append(uploaded_file.name)
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            failed_files.append(uploaded_file.name)
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except:
                pass
    
    progress_bar.empty()
    status_text.empty()
    
    if processed_files:
        st.success(f"✅ Successfully processed: {', '.join(processed_files)}")
        documents_processed = True
    
    if failed_files:
        st.error(f"❌ Failed to process: {', '.join(failed_files)}")
    
    # Question input
    st.subheader("💬 Ask Questions")
    question = st.text_input("Ask a question about the documents:")
    
    # Question type selection
    question_type = st.radio(
        "Question Type",
        ["Simple Query", "Reasoning Query", "Cross-modal Query"],
        help="Select the type of question you want to ask"
    )
    
    # Metadata filtering options
    with st.expander("🔍 Advanced Search Options"):
        filter_by_content_type = st.checkbox("Filter by content type")
        content_type_filter = ""
        if filter_by_content_type:
            content_type_filter = st.selectbox(
                "Content Type",
                ["paragraph", "table", "title", "section", "image", "audio", "video"]
            )
        
        use_knowledge_graph = st.checkbox("Use knowledge graph", value=False)
    
    if question and documents_processed:
        with st.spinner("🧠 Generating answer..."):
            try:
                # Apply metadata filter if requested
                filter_metadata = None
                if filter_by_content_type and content_type_filter:
                    filter_metadata = {"content_type": content_type_filter}
                
                # Generate answer based on question type
                if question_type == "Reasoning Query":
                    answer = pipeline.ask_with_reasoning(question, max_steps=reasoning_steps)
                else:
                    answer = pipeline.ask(
                        question, 
                        filter_metadata=filter_metadata,
                        use_knowledge_graph=use_knowledge_graph
                    )
                
                # Display answer
                st.subheader("📝 Answer")
                st.write(answer)
                
                # Display additional information
                with st.expander("ℹ️ Answer Details"):
                    st.info("This answer was generated using NexusRAG's advanced RAG capabilities.")
                    
            except Exception as e:
                st.error(f"❌ Error generating answer: {str(e)}")
                st.info("💡 Try rephrasing your question or check if your documents were processed correctly.")
else:
    st.info("📁 Please upload documents to get started. Supported formats: PDF, Word, HTML, Markdown, Text, Images")
    
    # Quick example
    with st.expander("💡 Quick Example"):
        st.markdown("""
        Try uploading a PDF document and asking questions like:
        - "What is this document about?"
        - "Summarize the key points"
        - "Find specific information about [topic]"
        """)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #888;'>NexusRAG - Open Source RAG Framework</div>", unsafe_allow_html=True)
