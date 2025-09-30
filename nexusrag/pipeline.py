from typing import List, Dict, Any
from nexusrag.parsers.base import BaseParser, Document
from nexusrag.embedders.base import BaseEmbedder
from nexusrag.vectorstores.base import BaseVectorStore
from nexusrag.llms.base import BaseLLM


class RAGPipeline:
    """Main RAG pipeline that orchestrates all components."""
    
    def __init__(self, 
                 parser: BaseParser,
                 embedder: BaseEmbedder,
                 vector_store: BaseVectorStore,
                 llm: BaseLLM):
        """Initialize the RAG pipeline with components.
        
        Args:
            parser (BaseParser): Document parser
            embedder (BaseEmbedder): Text embedder
            vector_store (BaseVectorStore): Vector store
            llm (BaseLLM): Language model
        """
        self.parser = parser
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm
        
    def process_document(self, file_path: str) -> None:
        """Process a document file and add it to the vector store.
        
        Args:
            file_path (str): Path to the document file
        """
        documents = self.parser.parse(file_path)
        self.vector_store.add(documents)
        
    def ask(self, question: str) -> str:
        """Ask a question about the processed documents.
        
        Args:
            question (str): Question to ask
            
        Returns:
            str: Answer to the question
        """
        # Retrieve relevant documents
        context = self.vector_store.query(question)
        
        # Generate answer using LLM
        answer = self.llm.generate(question, context)
        return answer
