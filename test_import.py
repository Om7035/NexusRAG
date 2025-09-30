import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(__file__))

try:
    from nexusrag import RAG
    print("Successfully imported RAG")
    rag = RAG()
    print("Successfully created RAG instance")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
