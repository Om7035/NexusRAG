"""
Generation & Reasoning Example for NexusRAG.

This example demonstrates the enhanced generation and reasoning capabilities of NexusRAG.
"""

import os
import tempfile
from pathlib import Path
import sys

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nexusrag.parsers.base import Document
from nexusrag.embedders.universal import UniversalEmbedder
from nexusrag.vectorstores.universal import UniversalVectorStore
from nexusrag.llms.universal import UniversalLLM
from nexusrag.generation.universal import UniversalGenerator


def create_sample_documents():
    """Create sample documents for demonstration."""
    documents = [
        Document(
            content="Apple Inc. is a technology company founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976.",
            metadata={"source": "tech_companies.txt", "content_type": "text"}
        ),
        Document(
            content="Steve Jobs was the CEO of Apple Inc. until his death in 2011. He was born on February 24, 1955.",
            metadata={"source": "ceos.txt", "content_type": "text"}
        ),
        Document(
            content="Tim Cook is the current CEO of Apple Inc. He took over after Steve Jobs' death in 2011.",
            metadata={"source": "ceos.txt", "content_type": "text"}
        ),
        Document(
            content="Microsoft is a technology company founded by Bill Gates and Paul Allen in 1975.",
            metadata={"source": "tech_companies.txt", "content_type": "text"}
        ),
        Document(
            content="Bill Gates was the CEO of Microsoft until 2000. He was born on October 28, 1955.",
            metadata={"source": "ceos.txt", "content_type": "text"}
        ),
        Document(
            content="Satya Nadella is the current CEO of Microsoft. He took over in 2014.",
            metadata={"source": "ceos.txt", "content_type": "text"}
        ),
        Document(
            content="Google is a technology company founded by Larry Page and Sergey Brin in 1998.",
            metadata={"source": "tech_companies.txt", "content_type": "text"}
        ),
        Document(
            content="Larry Page and Sergey Brin were the CEOs of Google until 2015. They were both born in 1973.",
            metadata={"source": "ceos.txt", "content_type": "text"}
        ),
        Document(
            content="Sundar Pichai is the current CEO of Google. He took over in 2015.",
            metadata={"source": "ceos.txt", "content_type": "text"}
        ),
    ]
    
    return documents


def main():
    """Demonstrate generation and reasoning capabilities."""
    print("NexusRAG Generation & Reasoning Example")
    print("=" * 42)
    
    # Create sample documents
    documents = create_sample_documents()
    print(f"Created {len(documents)} sample documents")
    
    # Initialize components
    print("\nInitializing Generation & Reasoning components...")
    embedder = UniversalEmbedder(provider="sentence-transformers")
    vector_store = UniversalVectorStore(provider="chroma")
    llm = UniversalLLM(provider="huggingface", model_name="google/flan-t5-base")
    
    # Initialize universal generator
    generator = UniversalGenerator(llm, vector_store)
    print("✓ Components initialized successfully")
    
    # Add documents to vector store
    print("\nAdding documents to vector store...")
    vector_store.add(documents)
    print("✓ Documents added successfully")
    
    # Perform comprehensive reasoning
    print("\nPerforming COMPREHENSIVE REASONING:")
    query = "Compare the leadership changes at Apple and Microsoft."
    print(f"Query: {query}")
    
    reasoning_result = generator.generate_answer(query, max_reasoning_steps=3)
    print(f"Answer: {reasoning_result['answer'][:200]}...")
    print(f"Confidence: {reasoning_result['confidence']:.2f}")
    print(f"Context used: {reasoning_result['context_used']} documents")
    print(f"Citation rate: {reasoning_result['citation_report']['citation_rate']:.2f}")
    
    # Verify a claim
    print("\nVerifying a CLAIM:")
    claim = "Steve Jobs was the CEO of Apple Inc."
    print(f"Claim: {claim}")
    
    verification_result = generator.verify_claim(claim)
    print(f"Verified: {verification_result['verified']}")
    print(f"Confidence: {verification_result['confidence']:.2f}")
    if verification_result['supporting_evidence']:
        evidence = verification_result['supporting_evidence'][0]
        print(f"Evidence: {evidence['content'][:100]}...")
    
    # Generate JSON response
    print("\nGenerating STRUCTURED JSON response:")
    json_prompt = "List the CEOs of Apple, Microsoft, and Google with their tenure start years."
    print(f"Prompt: {json_prompt}")
    
    json_result = generator.generate_json(json_prompt, {
        "type": "object",
        "properties": {
            "companies": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "ceo": {"type": "string"},
                        "tenure_start": {"type": "number"}
                    },
                    "required": ["name", "ceo", "tenure_start"]
                }
            }
        },
        "required": ["companies"]
    })
    
    if json_result['valid_json']:
        print(f"Generated JSON: {json_result['response']}")
    else:
        print(f"Failed to generate valid JSON: {json_result.get('error', 'Unknown error')}")
    
    # Generate with constraints
    print("\nGenerating with CONSTRAINTS:")
    constraint_prompt = "Explain quantum computing in simple terms."
    print(f"Prompt: {constraint_prompt}")
    
    constraint_result = generator.generate_with_constraints(constraint_prompt, {
        "max_length": 200,
        "required_keywords": ["quantum", "computing"],
        "forbidden_keywords": ["complex", "advanced"]
    })
    
    if constraint_result['constraints_satisfied']:
        print(f"Constrained response: {constraint_result['response'][:150]}...")
    else:
        print(f"Constraints not satisfied: {constraint_result.get('constraint_check', {}).get('violations', [])}")
    
    # Generate with regex pattern
    print("\nGenerating with REGEX pattern:")
    regex_prompt = "Provide a phone number in the format (XXX) XXX-XXXX."
    print(f"Prompt: {regex_prompt}")
    
    regex_result = generator.generate_with_regex(regex_prompt, r"\(\d{3}\) \d{3}-\d{4}")
    
    if regex_result['pattern_matched']:
        print(f"Generated phone number: {regex_result['response']}")
    else:
        print(f"Pattern not matched: {regex_result.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 42)
    print("Generation & reasoning example completed successfully!")
    print("\nNexusRAG now supports:")
    print("- Local LLM orchestration")
    print("- Multi-step reasoning")
    print("- Citation & verification")
    print("- Constrained generation")


if __name__ == "__main__":
    main()
