#!/usr/bin/env python3
"""
Evaluation script for NexusRAG.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nexusrag.evaluation.benchmark import BenchmarkSuite


def main():
    """Run the evaluation suite."""
    print("NexusRAG Evaluation Suite")
    print("=" * 30)
    
    # Check if required dependencies are installed
    try:
        import sklearn
        import numpy as np
    except ImportError as e:
        print(f"Missing required dependencies: {e}")
        print("Please install them with: pip install scikit-learn numpy")
        sys.exit(1)
    
    # Run benchmarks
    try:
        benchmark = BenchmarkSuite()
        results = benchmark.run_full_benchmark()
        benchmark.print_report()
        
        print("\nEvaluation completed successfully!")
        print("For detailed results, check the benchmark results.")
        
    except Exception as e:
        print(f"Error running evaluation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
