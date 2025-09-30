"""
Document Processing Example for NexusRAG.

This example demonstrates the enhanced document processing capabilities of NexusRAG.
"""

import os
import tempfile
from pathlib import Path
import sys

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nexusrag.parsers.universal import UniversalParser
from nexusrag.parsers.advanced_pdf import AdvancedPDFParser
from nexusrag.metadata.extractor import MetadataExtractor

# Temporarily disable chunking imports due to circular import issues
# from nexusrag.chunking.universal import UniversalChunker


def create_sample_files(temp_dir):
    """Create sample files for demonstration."""
    files = {}
    
    # Create a sample text file
    txt_path = os.path.join(temp_dir, "sample.txt")
    with open(txt_path, 'w') as f:
        f.write("""This is a sample text document.

It contains multiple paragraphs.

This is the second paragraph with some content.

And this is the third paragraph.""")
    files['text'] = txt_path
    
    # Create a sample PDF file
    pdf_path = os.path.join(temp_dir, "sample.pdf")
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    
    c = canvas.Canvas(pdf_path, pagesize=letter)
    c.drawString(100, 750, "Sample PDF Document")
    c.drawString(100, 700, "This is page 1 of the sample PDF.")
    c.drawString(100, 650, "It contains some text content.")
    c.showPage()
    c.drawString(100, 750, "This is page 2 of the sample PDF.")
    c.drawString(100, 700, "It also contains some text content.")
    c.save()
    files['pdf'] = pdf_path
    
    # Create a sample image file
    img_path = os.path.join(temp_dir, "sample.png")
    from PIL import Image, ImageDraw
    
    img = Image.new('RGB', (200, 100), color=(73, 109, 137))
    d = ImageDraw.Draw(img)
    d.text((10, 10), "Sample Image", fill=(255, 255, 0))
    img.save(img_path)
    files['image'] = img_path
    
    # Create a sample audio file (just metadata)
    audio_path = os.path.join(temp_dir, "sample.mp3")
    with open(audio_path, 'w') as f:
        f.write("")  # Empty file for metadata demonstration
    files['audio'] = audio_path
    
    # Create a sample video file (just metadata)
    video_path = os.path.join(temp_dir, "sample.mp4")
    with open(video_path, 'w') as f:
        f.write("")  # Empty file for metadata demonstration
    files['video'] = video_path
    
    return files


def main():
    """Demonstrate document processing capabilities."""
    print("NexusRAG Document Processing Example")
    print("=" * 40)
    
    # Create a temporary directory for our files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create sample files
        files = create_sample_files(temp_dir)
        print(f"Created {len(files)} sample files")
        
        # Initialize universal parser
        print("\nInitializing Universal Parser...")
        parser = UniversalParser()
        print("✓ Parser initialized successfully")
        
        # Process different file types
        print("\nProcessing different file types...")
        for file_type, file_path in files.items():
            print(f"\nProcessing {file_type.upper()} file: {os.path.basename(file_path)}")
            
            try:
                # Parse the document
                documents = parser.parse(file_path)
                print(f"  Parsed into {len(documents)} document(s)")
                
                # Show metadata for first document
                if documents:
                    doc = documents[0]
                    print(f"  Content preview: {doc.content[:50]}...")
                    print(f"  Metadata keys: {list(doc.metadata.keys())}")
                    
                    # Show some specific metadata
                    if 'file_size' in doc.metadata:
                        print(f"  File size: {doc.metadata['file_size']} bytes")
                    if 'content_length' in doc.metadata:
                        print(f"  Content length: {doc.metadata['content_length']} characters")
                    if 'word_count' in doc.metadata:
                        print(f"  Word count: {doc.metadata['word_count']}")
                        
            except Exception as e:
                print(f"  Error processing {file_type} file: {e}")
        
        # Demonstrate advanced PDF parsing
        print("\nDemonstrating Advanced PDF Parsing...")
        pdf_parser = AdvancedPDFParser()
        pdf_documents = pdf_parser.parse(files['pdf'])
        print(f"  Parsed PDF into {len(pdf_documents)} document(s)")
        
        for i, doc in enumerate(pdf_documents):
            print(f"  Document {i+1}: {doc.content[:30]}... (Page {doc.metadata.get('page', 'N/A')})")
        
        # Demonstrate chunking strategies
        # Temporarily disabled due to circular import issues
        # print("\nDemonstrating Chunking Strategies...")
        # sample_doc = pdf_documents[0] if pdf_documents else documents[0]
        # 
        # # Character-based chunking
        # char_chunker = UniversalChunker(strategy="character", chunk_size=50, chunk_overlap=10)
        # char_chunks = char_chunker.chunk_document(sample_doc)
        # print(f"  Character-based chunking: {len(char_chunks)} chunks")
        # 
        # # Semantic chunking
        # semantic_chunker = UniversalChunker(strategy="semantic", max_chunk_size=100)
        # semantic_chunks = semantic_chunker.chunk_document(sample_doc)
        # print(f"  Semantic chunking: {len(semantic_chunks)} chunks")
        # 
        # # Sentence chunking
        # sentence_chunker = UniversalChunker(strategy="sentence", max_chunk_size=100)
        # sentence_chunks = sentence_chunker.chunk_document(sample_doc)
        # print(f"  Sentence chunking: {len(sentence_chunks)} chunks")
        
        # Demonstrate metadata extraction
        print("\nDemonstrating Metadata Extraction...")
        file_metadata = MetadataExtractor.extract_file_metadata(files['text'])
        print(f"  File metadata: {file_metadata}")
        
        # Use the first document for content metadata extraction
        first_doc = pdf_documents[0] if pdf_documents else documents[0]
        content_metadata = MetadataExtractor.extract_content_metadata(first_doc.content)
        print(f"  Content metadata: {content_metadata}")
        
        print("\n" + "=" * 40)
        print("Document processing example completed successfully!")
        print("\nNexusRAG now supports:")
        print("- Universal file parsing (text, PDF, images, audio, video)")
        print("- Advanced PDF parsing with layout analysis")
        print("- Smart chunking strategies (character, semantic, sentence) - temporarily disabled due to circular import issues")
        print("- Comprehensive metadata extraction")


if __name__ == "__main__":
    main()
