from flask import Flask, request, jsonify
from typing import List, Dict, Any
import os
import tempfile
from nexusrag.rag import RAG


def create_app():
    """Create Flask app for NexusRAG API."""
    app = Flask(__name__)
    
    # Initialize RAG instance
    rag = RAG()
    
    @app.route('/health', methods=['GET'])
    def health():
        """Health check endpoint."""
        return jsonify({"status": "healthy", "version": "0.1.0"})
    
    @app.route('/process', methods=['POST'])
    def process_documents():
        """Process uploaded documents."""
        try:
            # Get uploaded files
            files = request.files.getlist('files')
            
            if not files or len(files) == 0:
                return jsonify({"error": "No files provided"}), 400
            
            # Save files temporarily
            file_paths = []
            for file in files:
                # Create temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False)
                file.save(temp_file.name)
                file_paths.append(temp_file.name)
            
            # Process documents
            rag.process(file_paths)
            
            # Clean up temporary files
            for path in file_paths:
                os.unlink(path)
            
            return jsonify({"status": "success", "message": f"Processed {len(files)} document(s)"})
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/ask', methods=['POST'])
    def ask_question():
        """Ask a question about processed documents."""
        try:
            # Get question from request
            data = request.get_json()
            question = data.get('question', '')
            
            if not question:
                return jsonify({"error": "No question provided"}), 400
            
            # Get optional parameters
            filter_metadata = data.get('filter_metadata', None)
            top_k = data.get('top_k', 5)
            
            # Ask question
            answer = rag.ask(question, filter_metadata, top_k)
            
            return jsonify({
                "question": question,
                "answer": answer
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/ask_with_reasoning', methods=['POST'])
    def ask_with_reasoning():
        """Ask a question with multi-step reasoning."""
        try:
            # Get question from request
            data = request.get_json()
            question = data.get('question', '')
            
            if not question:
                return jsonify({"error": "No question provided"}), 400
            
            # Get optional parameters
            max_steps = data.get('max_steps', 3)
            
            # Ask question with reasoning
            answer = rag.ask_with_reasoning(question, max_steps)
            
            return jsonify({
                "question": question,
                "answer": answer
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/config', methods=['GET'])
    def get_config():
        """Get current configuration."""
        # This would require implementing a config getter in RAG
        # For now, we'll return a placeholder
        return jsonify({
            "embedder": "sentence-transformers",
            "vector_store": "chroma",
            "llm": "huggingface"
        })
    
    return app


def main():
    """Main entry point for the API server."""
    app = create_app()
    app.run(host='0.0.0.0', port=8000, debug=True)


if __name__ == '__main__':
    main()
