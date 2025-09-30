#!/usr/bin/env python3
"""
Script to run all tests for NexusRAG.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, cwd=None):
    """Run a command and return the result."""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            cwd=cwd,
            check=True, 
            capture_output=True, 
            text=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(f"Error output: {e.stderr}")
        return None


def main():
    """Run all tests."""
    print("Running NexusRAG Tests")
    print("=" * 30)
    
    # Get project root directory
    project_root = Path(__file__).parent.parent
    
    # Run component tests
    print("\n1. Running component tests...")
    result = run_command(f"{sys.executable} scripts/test_components.py", cwd=project_root)
    if result:
        print(result)
    
    # Run import tests with pytest
    print("\n2. Running import tests with pytest...")
    result = run_command(f"{sys.executable} -m pytest tests/test_imports.py -v", cwd=project_root)
    if result:
        print(result)
    
    # Run pipeline tests with pytest
    print("\n3. Running pipeline tests with pytest...")
    result = run_command(f"{sys.executable} -m pytest tests/test_pipeline.py -v", cwd=project_root)
    if result:
        print(result)
    
    print("\n" + "=" * 30)
    print("All tests completed!")


if __name__ == "__main__":
    main()
