from typing import List, Dict, Any, Optional
from .parsers.base import BaseParser, Document
from .embedders.base import BaseEmbedder
from .vectorstores.base import BaseVectorStore
from .llms.base import BaseLLM
from .chunking import DocumentChunker
from .metadata_filter import MetadataFilter
from .multimodal import MultimodalProcessor
from .knowledge_graph import KnowledgeGraphBuilder, KnowledgeGraph
from .agents.basic_agent import BasicAgent


class EnhancedRAGPipeline:
    """Enhanced RAG pipeline with advanced features."""
    
    def __init__(self, 
                 parser: BaseParser,
                 embedder: BaseEmbedder,
                 vector_store: BaseVectorStore,
                 llm: BaseLLM,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """Initialize the enhanced RAG pipeline.
        
        Args:
            parser (BaseParser): Document parser
            embedder (BaseEmbedder): Text embedder
            vector_store (BaseVectorStore): Vector store
            llm (BaseLLM): Language model
            chunk_size (int): Maximum size of each chunk in characters
            chunk_overlap (int): Number of characters to overlap between chunks
        """
        self.parser = parser
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm
        
        # Initialize enhanced components
        self.chunker = DocumentChunker(chunk_size, chunk_overlap)
        self.metadata_filter = MetadataFilter()
        self.multimodal_processor = MultimodalProcessor()
        self.knowledge_graph_builder = KnowledgeGraphBuilder()
        self.knowledge_graph: Optional[KnowledgeGraph] = None
        
        # Initialize agent
        self.agent = BasicAgent(llm, vector_store)
        
    def process_document(self, file_path: str, chunk: bool = True) -> None:
        """Process a document file and add it to the vector store.
        
        Args:
            file_path (str): Path to the document file
            chunk (bool): Whether to chunk the document
        """
        # Process multimodal document
        documents = self.multimodal_processor.process_multimodal_document(file_path)
        
        # Chunk documents if requested
        if chunk:
            documents = self.chunker.chunk_documents(documents)
        
        # Add to vector store
        self.vector_store.add(documents)
        
    def process_documents(self, file_paths: List[str], chunk: bool = True) -> None:
        """Process multiple document files and add them to the vector store.
        
        Args:
            file_paths (List[str]): Paths to the document files
            chunk (bool): Whether to chunk the documents
        """
        all_documents = []
        for file_path in file_paths:
            # Process multimodal document
            documents = self.multimodal_processor.process_multimodal_document(file_path)
            
            # Chunk documents if requested
            if chunk:
                documents = self.chunker.chunk_documents(documents)
            
            all_documents.extend(documents)
        
        # Add all documents to vector store
        self.vector_store.add(all_documents)
        
        # Build knowledge graph from all documents
        self.knowledge_graph = self.knowledge_graph_builder.build_from_documents(all_documents)
    
    def ask(self, question: str, 
            top_k: int = 5,
            filter_metadata: Dict[str, Any] = None,
            use_knowledge_graph: bool = False) -> str:
        """Ask a question about the processed documents.
        
        Args:
            question (str): Question to ask
            top_k (int): Number of top results to return
            filter_metadata (Dict[str, Any]): Metadata filter criteria
            use_knowledge_graph (bool): Whether to use knowledge graph for enhanced reasoning
            
        Returns:
            str: Answer to the question
        """
        # Retrieve relevant documents
        context = self.vector_store.query(question, top_k)
        
        # Apply metadata filter if provided
        if filter_metadata:
            # Create a filter function based on the provided metadata
            def metadata_filter_func(metadata):
                for key, value in filter_metadata.items():
                    if metadata.get(key) != value:
                        return False
                return True
            
            # Filter context
            filtered_context = [
                doc for doc in context 
                if metadata_filter_func(doc.get("metadata", {}))
            ]
            context = filtered_context
        
        # Enhance context with knowledge graph information if requested
        if use_knowledge_graph and self.knowledge_graph:
            # In a more advanced implementation, we would use the knowledge graph
            # to provide additional context and relationships
            # For now, we'll just note that it's being used
            enhanced_context = self._enhance_context_with_knowledge_graph(context, question)
            context = enhanced_context
        
        # Generate answer using LLM
        answer = self.llm.generate(question, context)
        return answer
    
    def ask_with_reasoning(self, question: str, max_steps: int = 3) -> str:
        """Ask a question with multi-step reasoning.
        
        Args:
            question (str): Question to ask
            max_steps (int): Maximum number of reasoning steps
            
        Returns:
            str: Answer to the question
        """
        return self.agent.think(question, max_steps)
    
    def _enhance_context_with_knowledge_graph(self, context: List[Dict[str, Any]], question: str) -> List[Dict[str, Any]]:
        """Enhance context with knowledge graph information.
        
        Args:
            context (List[Dict[str, Any]]): Original context
            question (str): Question being asked
            
        Returns:
            List[Dict[str, Any]]: Enhanced context
        """
        # In a more advanced implementation, we would:
        # 1. Extract entities from the question
        # 2. Find related entities in the knowledge graph
        # 3. Add relationships and additional context
        
        # For now, we'll just return the original context
        # This is a placeholder for future enhancement
        return context
    
    def filter_and_ask(self, question: str,
                      filter_func: callable,
                      top_k: int = 5,
                      use_knowledge_graph: bool = False) -> str:
        """Ask a question with a custom filter function.
        
        Args:
            question (str): Question to ask
            filter_func (callable): Function to filter documents by metadata
            top_k (int): Number of top results to return
            use_knowledge_graph (bool): Whether to use knowledge graph for enhanced reasoning
            
        Returns:
            str: Answer to the question
        """
        # Retrieve all documents (we'll filter after)
        all_context = self.vector_store.query(question, top_k * 3)  # Get more results to filter
        
        # Apply custom filter
        filtered_context = [
            doc for doc in all_context 
            if filter_func(doc.get("metadata", {}))
        ][:top_k]  # Limit to top_k after filtering
        
        # Enhance context with knowledge graph information if requested
        if use_knowledge_graph and self.knowledge_graph:
            enhanced_context = self._enhance_context_with_knowledge_graph(filtered_context, question)
            filtered_context = enhanced_context
        
        # Generate answer using LLM
        answer = self.llm.generate(question, filtered_context)
        return answer
