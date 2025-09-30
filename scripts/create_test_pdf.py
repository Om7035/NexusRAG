#!/usr/bin/env python3
"""
Script to create a simple test PDF file for testing NexusRAG components.
"""

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import os

def create_test_pdf(filename="test_document.pdf"):
    """Create a simple test PDF document."""
    # Create document
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Add title
    title = Paragraph("Test Document for NexusRAG", styles["Title"])
    story.append(title)
    story.append(Spacer(1, 12))
    
    # Add some content
    content = [
        "This is a test document created for testing NexusRAG components.",
        "NexusRAG is an open-source framework for building autonomous AI agents that reason over complex, multimodal data.",
        "It combines high-fidelity document parsing with a fully modular architecture.",
        "The framework enables developers to create powerful, data-aware applications.",
        "Key features include multimodal parsing, modular design, and agent-ready capabilities.",
        "The capital of France is Paris.",
        "The largest planet in our solar system is Jupiter.",
        "Water boils at 100 degrees Celsius at sea level.",
        "The Python programming language was created by Guido van Rossum.",
        "Machine learning is a subset of artificial intelligence."
    ]
    
    for paragraph_text in content:
        paragraph = Paragraph(paragraph_text, styles["Normal"])
        story.append(paragraph)
        story.append(Spacer(1, 12))
    
    # Build PDF
    doc.build(story)
    print(f"Created test PDF: {filename}")

if __name__ == "__main__":
    try:
        import reportlab
        create_test_pdf()
    except ImportError:
        print("ReportLab library not found. Please install it with: pip install reportlab")
        print("Skipping test PDF creation.")
