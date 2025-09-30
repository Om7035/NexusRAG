import time
import tempfile
import os
from pathlib import Path
import sys
from typing import List, Dict, Any

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from nexusrag.parsers.universal import UniversalParser
from nexusrag.embedders.universal import UniversalEmbedder
from nexusrag.vectorstores.universal import UniversalVectorStore
from nexusrag.llms.universal import UniversalLLM
from nexusrag.enhanced_pipeline import EnhancedRAGPipeline
from nexusrag.parsers.base import Document
from .metrics import EvaluationMetrics


class BenchmarkSuite:
    """Comprehensive benchmark suite for NexusRAG."""
    
    def __init__(self):
        """Initialize the benchmark suite."""
        self.results = {}
    
    def benchmark_document_processing(self) -> Dict[str, Any]:
        """Benchmark document processing components."""
        print("Benchmarking Document Processing Components...")
        
        results = {}
        
        # Test with different file types
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test documents
            test_docs = self._create_test_documents(temp_dir)
            
            # Benchmark parser
            parser_time = self._benchmark_parser(test_docs)
            results["parser"] = {"time": parser_time}
            
            # Benchmark chunking
            chunking_time = self._benchmark_chunking(test_docs)
            results["chunking"] = {"time": chunking_time}
            
            # Benchmark metadata extraction
            metadata_time = self._benchmark_metadata_extraction(test_docs)
            results["metadata_extraction"] = {"time": metadata_time}
        
        return results
    
    def benchmark_retrieval(self) -> Dict[str, Any]:
        """Benchmark retrieval components."""
        print("Benchmarking Retrieval Components...")
        
        results = {}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test documents
            test_docs = self._create_test_documents(temp_dir)
            
            # Benchmark different vector stores
            vector_stores = ["chroma", "qdrant"]
            for store_type in vector_stores:
                try:
                    store_results = self._benchmark_vector_store(store_type, test_docs)
                    results[store_type] = store_results
                except Exception as e:
                    print(f"  Error benchmarking {store_type}: {e}")
                    results[store_type] = {"error": str(e)}
            
            # Benchmark hybrid search
            hybrid_results = self._benchmark_hybrid_search(test_docs)
            results["hybrid_search"] = hybrid_results
        
        return results
    
    def benchmark_generation(self) -> Dict[str, Any]:
        """Benchmark generation components."""
        print("Benchmarking Generation Components...")
        
        results = {}
        
        # Benchmark different LLMs
        llms = ["huggingface", "ollama"]
        for llm_type in llms:
            try:
                llm_results = self._benchmark_llm(llm_type)
                results[llm_type] = llm_results
            except Exception as e:
                print(f"  Error benchmarking {llm_type}: {e}")
                results[llm_type] = {"error": str(e)}
        
        # Benchmark reasoning
        reasoning_results = self._benchmark_reasoning()
        results["reasoning"] = reasoning_results
        
        return results
    
    def benchmark_multimodal(self) -> Dict[str, Any]:
        """Benchmark multimodal components."""
        print("Benchmarking Multimodal Components...")
        
        results = {}
        
        # Benchmark image processing
        image_results = self._benchmark_image_processing()
        results["image_processing"] = image_results
        
        # Benchmark audio processing
        audio_results = self._benchmark_audio_processing()
        results["audio_processing"] = audio_results
        
        # Benchmark table processing
        table_results = self._benchmark_table_processing()
        results["table_processing"] = table_results
        
        return results
    
    def run_full_benchmark(self) -> Dict[str, Any]:
        """Run the full benchmark suite."""
        print("Running Full NexusRAG Benchmark Suite")
        print("=" * 40)
        
        start_time = time.time()
        
        # Run all benchmarks
        document_results = self.benchmark_document_processing()
        retrieval_results = self.benchmark_retrieval()
        generation_results = self.benchmark_generation()
        multimodal_results = self.benchmark_multimodal()
        
        end_time = time.time()
        
        # Compile results
        self.results = {
            "document_processing": document_results,
            "retrieval": retrieval_results,
            "generation": generation_results,
            "multimodal": multimodal_results,
            "total_time": end_time - start_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return self.results
    
    def print_report(self) -> None:
        """Print a formatted benchmark report."""
        if not self.results:
            print("No benchmark results available. Run benchmarks first.")
            return
        
        print("\nNexusRAG Benchmark Report")
        print("=" * 40)
        print(f"Timestamp: {self.results['timestamp']}")
        print(f"Total Time: {self.results['total_time']:.2f} seconds")
        
        # Document processing results
        print("\nDocument Processing:")
        doc_results = self.results.get("document_processing", {})
        for component, metrics in doc_results.items():
            if "time" in metrics:
                print(f"  {component}: {metrics['time']:.2f} seconds")
            elif "error" in metrics:
                print(f"  {component}: Error - {metrics['error']}")
        
        # Retrieval results
        print("\nRetrieval:")
        retrieval_results = self.results.get("retrieval", {})
        for component, metrics in retrieval_results.items():
            if "time" in metrics:
                print(f"  {component}: {metrics['time']:.2f} seconds")
            elif "error" in metrics:
                print(f"  {component}: Error - {metrics['error']}")
        
        # Generation results
        print("\nGeneration:")
        generation_results = self.results.get("generation", {})
        for component, metrics in generation_results.items():
            if "time" in metrics:
                print(f"  {component}: {metrics['time']:.2f} seconds")
            elif "error" in metrics:
                print(f"  {component}: Error - {metrics['error']}")
        
        print("\n" + "=" * 40)
    
    def _create_test_documents(self, temp_dir: str) -> List[str]:
        """Create test documents for benchmarking.
        
        Args:
            temp_dir (str): Temporary directory path
            
        Returns:
            List[str]: List of document file paths
        """
        # Create a simple text document
        text_path = os.path.join(temp_dir, "test_document.txt")
        with open(text_path, 'w') as f:
            f.write("""This is a test document for benchmarking NexusRAG.
            
            NexusRAG is an open-source framework for building autonomous AI agents.
            It combines high-fidelity document parsing with a fully modular architecture.
            The framework enables developers to create powerful, data-aware applications.
            
            Key features include:
            - Multimodal parsing
            - Modular design
            - Agent-ready capabilities
            - Extensible components
            
            The capital of France is Paris.
            The largest planet in our solar system is Jupiter.
            Water boils at 100 degrees Celsius at sea level.
            """ * 10)  # Repeat to make it larger
        
        return [text_path]
    
    def _benchmark_parser(self, test_docs: List[str]) -> float:
        """Benchmark the parser component.
        
        Args:
            test_docs (List[str]): List of document file paths
            
        Returns:
            float: Time taken in seconds
        """
        parser = UniversalParser()
        
        start_time = time.time()
        for doc_path in test_docs:
            documents = parser.parse(doc_path)
        end_time = time.time()
        
        return end_time - start_time
    
    def _benchmark_chunking(self, test_docs: List[str]) -> float:
        """Benchmark the chunking component.
        
        Args:
            test_docs (List[str]): List of document file paths
            
        Returns:
            float: Time taken in seconds
        """
        from nexusrag.chunking.universal import UniversalChunker
        
        # Parse documents first
        parser = UniversalParser()
        all_documents = []
        for doc_path in test_docs:
            documents = parser.parse(doc_path)
            all_documents.extend(documents)
        
        # Benchmark chunking
        chunker = UniversalChunker(strategy="character", chunk_size=500, chunk_overlap=100)
        
        start_time = time.time()
        for doc in all_documents:
            chunked_docs = chunker.chunk_document(doc)
        end_time = time.time()
        
        return end_time - start_time
    
    def _benchmark_metadata_extraction(self, test_docs: List[str]) -> float:
        """Benchmark metadata extraction.
        
        Args:
            test_docs (List[str]): List of document file paths
            
        Returns:
            float: Time taken in seconds
        """
        from nexusrag.metadata.extractor import MetadataExtractor
        
        # Parse documents first
        parser = UniversalParser()
        all_documents = []
        for doc_path in test_docs:
            documents = parser.parse(doc_path)
            all_documents.extend(documents)
        
        # Benchmark metadata extraction
        start_time = time.time()
        for doc in all_documents:
            enhanced_doc = MetadataExtractor.enhance_document_metadata(doc, test_docs[0])
        end_time = time.time()
        
        return end_time - start_time
    
    def _benchmark_vector_store(self, store_type: str, test_docs: List[str]) -> Dict[str, Any]:
        """Benchmark a vector store.
        
        Args:
            store_type (str): Type of vector store
            test_docs (List[str]): List of document file paths
            
        Returns:
            Dict[str, Any]: Benchmark results
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize vector store
            if store_type == "chroma":
                vector_store = UniversalVectorStore(provider="chroma", persist_directory=os.path.join(temp_dir, "chroma_db"))
            elif store_type == "qdrant":
                vector_store = UniversalVectorStore(provider="qdrant")
            else:
                raise ValueError(f"Unsupported vector store: {store_type}")
            
            # Parse and chunk documents
            parser = UniversalParser()
            from nexusrag.chunking.universal import UniversalChunker
            chunker = UniversalChunker(strategy="character", chunk_size=500, chunk_overlap=100)
            
            all_documents = []
            for doc_path in test_docs:
                documents = parser.parse(doc_path)
                for doc in documents:
                    chunked_docs = chunker.chunk_document(doc)
                    all_documents.extend(chunked_docs)
            
            # Benchmark adding documents
            start_time = time.time()
            vector_store.add(all_documents)
            add_time = time.time() - start_time
            
            # Benchmark querying
            start_time = time.time()
            results = vector_store.query("test document", top_k=5)
            query_time = time.time() - start_time
            
            return {
                "add_time": add_time,
                "query_time": query_time,
                "total_time": add_time + query_time
            }
    
    def _benchmark_hybrid_search(self, test_docs: List[str]) -> Dict[str, Any]:
        """Benchmark hybrid search.
        
        Args:
            test_docs (List[str]): List of document file paths
            
        Returns:
            Dict[str, Any]: Benchmark results
        """
        # This would require a more complex setup with a hybrid retriever
        # For now, we'll return placeholder results
        return {
            "time": 0.0,
            "placeholder": True
        }
    
    def _benchmark_llm(self, llm_type: str) -> Dict[str, Any]:
        """Benchmark an LLM.
        
        Args:
            llm_type (str): Type of LLM
            
        Returns:
            Dict[str, Any]: Benchmark results
        """
        # Initialize LLM
        llm = UniversalLLM(provider=llm_type)
        
        # Benchmark generation
        prompt = "Explain what artificial intelligence is in one sentence."
        start_time = time.time()
        response = llm.generate(prompt)
        gen_time = time.time() - start_time
        
        return {
            "generation_time": gen_time,
            "response_length": len(response)
        }
    
    def _benchmark_reasoning(self) -> Dict[str, Any]:
        """Benchmark reasoning capabilities.
        
        Returns:
            Dict[str, Any]: Benchmark results
        """
        # This would require a more complex setup with a reasoning engine
        # For now, we'll return placeholder results
        return {
            "time": 0.0,
            "placeholder": True
        }
    
    def _benchmark_image_processing(self) -> Dict[str, Any]:
        """Benchmark image processing.
        
        Returns:
            Dict[str, Any]: Benchmark results
        """
        # This would require actual image files
        # For now, we'll return placeholder results
        return {
            "time": 0.0,
            "placeholder": True
        }
    
    def _benchmark_audio_processing(self) -> Dict[str, Any]:
        """Benchmark audio processing.
        
        Returns:
            Dict[str, Any]: Benchmark results
        """
        # This would require actual audio files
        # For now, we'll return placeholder results
        return {
            "time": 0.0,
            "placeholder": True
        }
    
    def _benchmark_table_processing(self) -> Dict[str, Any]:
        """Benchmark table processing.
        
        Returns:
            Dict[str, Any]: Benchmark results
        """
        # This would require actual table data
        # For now, we'll return placeholder results
        return {
            "time": 0.0,
            "placeholder": True
        }


def main():
    """Run the benchmark suite."""
    benchmark = BenchmarkSuite()
    results = benchmark.run_full_benchmark()
    benchmark.print_report()


if __name__ == "__main__":
    main()
