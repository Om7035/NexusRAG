from typing import List, Dict, Any, Optional
from nexusrag.parsers.universal import UniversalParser
from nexusrag.embedders.universal import UniversalEmbedder
from nexusrag.vectorstores.universal import UniversalVectorStore
from nexusrag.llms.universal import UniversalLLM


class RAG:
    """Simple RAG interface for easy usage."""
    
    def __init__(self, 
                 embedder: str = "sentence-transformers",
                 vector_store: str = "chroma",
                 llm: str = "huggingface",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """Initialize the RAG interface.
        
        Args:
            embedder (str): Embedder provider
            vector_store (str): Vector store provider
            llm (str): LLM provider
            chunk_size (int): Chunk size for document processing
            chunk_overlap (int): Chunk overlap for document processing
        """
        # Import EnhancedRAGPipeline here to avoid circular imports
        from .enhanced_pipeline import EnhancedRAGPipeline
        
        # Initialize components
        parser = UniversalParser()
        embedder_obj = UniversalEmbedder(provider=embedder)
        vector_store_obj = UniversalVectorStore(provider=vector_store)
        llm_obj = UniversalLLM(provider=llm)
        
        # Initialize pipeline
        self.pipeline = EnhancedRAGPipeline(
            parser=parser,
            embedder=embedder_obj,
            vector_store=vector_store_obj,
            llm=llm_obj,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def process(self, files: List[str]) -> None:
        """Process documents.
        
        Args:
            files (List[str]): List of file paths to process
        """
        self.pipeline.process_documents(files)
    
    def ask(self, question: str, 
            filter_metadata: Dict[str, Any] = None,
            top_k: int = 5) -> str:
        """Ask a question about processed documents.
        
        Args:
            question (str): Question to ask
            filter_metadata (Dict[str, Any]): Metadata filter criteria
            top_k (int): Number of top results to return
            
        Returns:
            str: Answer to the question
        """
        return self.pipeline.ask(question, top_k, filter_metadata)
    
    def ask_with_reasoning(self, question: str, max_steps: int = 3) -> str:
        """Ask a question with multi-step reasoning.
        
        Args:
            question (str): Question to ask
            max_steps (int): Maximum number of reasoning steps
            
        Returns:
            str: Answer to the question
        """
        return self.pipeline.ask_with_reasoning(question, max_steps)
    
    def clear(self) -> None:
        """Clear processed documents."""
        # This would require implementing a clear method in the pipeline
        # For now, we'll just note that this is a placeholder
        pass
    
    def __str__(self) -> str:
        """String representation of the RAG interface.
        
        Returns:
            str: String representation
        """
        return f"RAG(embedder=..., vector_store=..., llm=...)"
    
    def __repr__(self) -> str:
        """Detailed representation of the RAG interface.
        
        Returns:
            str: Detailed representation
        """
        return f"RAG(embedder=..., vector_store=..., llm=...)"
