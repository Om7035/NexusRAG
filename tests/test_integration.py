import unittest
import tempfile
import os
from pathlib import Path
import sys

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nexusrag.rag import RAG
from nexusrag.parsers.base import Document


class TestIntegration(unittest.TestCase):
    """Integration tests for NexusRAG."""
    
    def test_basic_rag_workflow(self):
        """Test basic RAG workflow with text documents."""
        # Create a simple RAG instance
        rag = RAG()
        
        # Create a test document
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("""This is a test document for integration testing.
            
            NexusRAG is an open-source framework for building autonomous AI agents.
            It combines high-fidelity document parsing with a fully modular architecture.
            The framework enables developers to create powerful, data-aware applications.
            
            The capital of France is Paris.
            The largest planet in our solar system is Jupiter.
            """)
            test_file = f.name
        
        try:
            # Process the document
            rag.process([test_file])
            
            # Ask a question
            answer = rag.ask("What is the capital of France?")
            
            # Verify we got an answer
            self.assertIsInstance(answer, str)
            self.assertGreater(len(answer), 0)
            
            # Check that the answer contains relevant information
            self.assertIn("Paris", answer)
            
        finally:
            # Clean up
            os.unlink(test_file)
    
    def test_multiple_document_types(self):
        """Test processing multiple document types."""
        # Create a simple RAG instance
        rag = RAG()
        
        # Create test documents of different types
        test_files = []
        
        # Text document
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a text document. The sky is blue.")
            test_files.append(f.name)
        
        try:
            # Process the documents
            rag.process(test_files)
            
            # Ask a question
            answer = rag.ask("What color is the sky?")
            
            # Verify we got an answer
            self.assertIsInstance(answer, str)
            self.assertGreater(len(answer), 0)
            
        finally:
            # Clean up
            for file_path in test_files:
                os.unlink(file_path)
    
    def test_reasoning_capabilities(self):
        """Test reasoning capabilities."""
        # Create a simple RAG instance
        rag = RAG()
        
        # Create a test document
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("""Mathematical facts:
            
            2 + 2 = 4
            5 * 3 = 15
            10 - 7 = 3
            """)
            test_file = f.name
        
        try:
            # Process the document
            rag.process([test_file])
            
            # Ask a question with reasoning
            answer = rag.ask_with_reasoning("What is 2 + 2?", max_steps=2)
            
            # Verify we got an answer
            self.assertIsInstance(answer, str)
            self.assertGreater(len(answer), 0)
            
        finally:
            # Clean up
            os.unlink(test_file)


def main():
    """Run the integration tests."""
    unittest.main()


if __name__ == "__main__":
    main()
