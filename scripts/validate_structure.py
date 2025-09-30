#!/usr/bin/env python3
"""
Script to validate the NexusRAG project structure.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_file_exists(filepath):
    """Check if a file exists and print status."""
    if os.path.exists(filepath):
        print(f"✓ {filepath} exists")
        return True
    else:
        print(f"✗ {filepath} missing")
        return False


def check_directory_exists(dirpath):
    """Check if a directory exists and print status."""
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        print(f"✓ {dirpath} exists")
        return True
    else:
        print(f"✗ {dirpath} missing")
        return False


def check_import(module_name):
    """Check if a module can be imported and print status."""
    try:
        __import__(module_name)
        print(f"✓ {module_name} can be imported")
        return True
    except ImportError as e:
        print(f"✗ {module_name} cannot be imported: {e}")
        return False


def main():
    """Validate the NexusRAG project structure."""
    print("NexusRAG Project Structure Validation")
    print("=" * 35)
    
    # Check required files
    print("\nChecking required files...")
    required_files = [
        "LICENSE",
        "README.md",
        "pyproject.toml",
        "setup.cfg",
        "requirements.txt",
        "Dockerfile",
        "docker-compose.yml",
        ".gitignore",
        ".env.example",
        "Makefile"
    ]
    
    file_checks = []
    for file in required_files:
        filepath = os.path.join(project_root, file)
        file_checks.append(check_file_exists(filepath))
    
    # Check required directories
    print("\nChecking required directories...")
    required_dirs = [
        "nexusrag",
        "nexusrag/parsers",
        "nexusrag/embedders",
        "nexusrag/vectorstores",
        "nexusrag/llms",
        "tests",
        "docs",
        "docs/api",
        "docs/api/parsers",
        "docs/api/embedders",
        "docs/api/vectorstores",
        "docs/api/llms",
        "scripts",
        "examples",
        ".github/workflows"
    ]
    
    dir_checks = []
    for dir in required_dirs:
        dirpath = os.path.join(project_root, dir)
        dir_checks.append(check_directory_exists(dirpath))
    
    # Check required Python modules
    print("\nChecking Python module imports...")
    required_modules = [
        "nexusrag",
        "nexusrag.pipeline",
        "nexusrag.enhanced_pipeline",
        "nexusrag.chunking",
        "nexusrag.metadata_filter",
        "nexusrag.multimodal",
        "nexusrag.knowledge_graph",
        "nexusrag.parsers.base",
        "nexusrag.parsers.pdf",
        "nexusrag.parsers.word",
        "nexusrag.parsers.html",
        "nexusrag.parsers.markdown",
        "nexusrag.parsers.text",
        "nexusrag.parsers.universal",
        "nexusrag.parsers.advanced_pdf",
        "nexusrag.processors.table_processor",
        "nexusrag.agents.basic_agent",
        "nexusrag.embedders.base",
        "nexusrag.embedders.sentence_transformers",
        "nexusrag.embedders.openai",
        "nexusrag.embedders.cohere",
        "nexusrag.embedders.gemini",
        "nexusrag.embedders.universal",
        "nexusrag.vectorstores.base",
        "nexusrag.vectorstores.chroma",
        "nexusrag.vectorstores.pinecone",
        "nexusrag.vectorstores.weaviate",
        "nexusrag.vectorstores.universal",
        "nexusrag.llms.base",
        "nexusrag.llms.huggingface",
        "nexusrag.llms.openai",
        "nexusrag.llms.anthropic",
        "nexusrag.llms.gemini",
        "nexusrag.llms.ollama",
        "nexusrag.llms.universal",
        "nexusrag.cli"
    ]
    
    import_checks = []
    for module in required_modules:
        import_checks.append(check_import(module))
    
    # Print summary
    print("\n" + "=" * 35)
    total_checks = len(file_checks) + len(dir_checks) + len(import_checks)
    passed_checks = sum(file_checks) + sum(dir_checks) + sum(import_checks)
    
    print(f"Validation Summary: {passed_checks}/{total_checks} checks passed")
    
    if passed_checks == total_checks:
        print("✓ All checks passed! Project structure is valid.")
        return True
    else:
        print("✗ Some checks failed. Please review the issues above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
