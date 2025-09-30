"""
Multimodal Processing Example for NexusRAG.

This example demonstrates the enhanced multimodal processing capabilities of NexusRAG.
"""

import os
import tempfile
from pathlib import Path
import sys

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import MultimodalProcessor directly from multimodal.py file
import importlib.util
import sys
import os

# Add the nexusrag directory to the path
nexusrag_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, nexusrag_path)

# Import MultimodalProcessor directly from the multimodal.py file
spec = importlib.util.spec_from_file_location("multimodal", os.path.join(nexusrag_path, "nexusrag", "multimodal.py"))
multimodal_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(multimodal_module)
MultimodalProcessor = multimodal_module.MultimodalProcessor

from nexusrag.parsers.universal import UniversalParser


def create_sample_files(temp_dir):
    """Create sample files for demonstration."""
    files = {}
    
    # Create a sample text file with a table
    txt_path = os.path.join(temp_dir, "sample_with_table.txt")
    with open(txt_path, 'w') as f:
        f.write("""This is a sample text document with a table.

It contains multiple paragraphs and a table.

| Name | Age | City |
|------|-----|------|
| John | 30  | NYC  |
| Jane | 25  | LA   |
| Bob  | 35  | Chicago |

This is the second paragraph with some content.

And this is the third paragraph.""")
    files['text_with_table'] = txt_path
    
    # Create a sample image file
    img_path = os.path.join(temp_dir, "sample.png")
    from PIL import Image, ImageDraw
    
    img = Image.new('RGB', (200, 100), color=(73, 109, 137))
    d = ImageDraw.Draw(img)
    d.text((10, 10), "Sample Image", fill=(255, 255, 0))
    img.save(img_path)
    files['image'] = img_path
    
    # Create a sample audio file (just metadata for now)
    audio_path = os.path.join(temp_dir, "sample.mp3")
    with open(audio_path, 'w') as f:
        f.write("")  # Empty file for metadata demonstration
    files['audio'] = audio_path
    
    # Create a sample video file (just metadata for now)
    video_path = os.path.join(temp_dir, "sample.mp4")
    with open(video_path, 'w') as f:
        f.write("")  # Empty file for metadata demonstration
    files['video'] = video_path
    
    return files


def main():
    """Demonstrate multimodal processing capabilities."""
    print("NexusRAG Multimodal Processing Example")
    print("=" * 45)
    
    # Create a temporary directory for our files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create sample files
        files = create_sample_files(temp_dir)
        print(f"Created {len(files)} sample files")
        
        # Initialize multimodal processor
        print("\nInitializing Multimodal Processor...")
        processor = MultimodalProcessor()
        print("✓ Processor initialized successfully")
        
        # Process different file types
        print("\nProcessing different file types...")
        
        # Process image file
        print("\nProcessing IMAGE file:")
        try:
            image_doc = processor.process_image(files['image'])
            print(f"  Content: {image_doc.content[:50]}...")
            print(f"  Metadata: {image_doc.metadata}")
        except Exception as e:
            print(f"  Error processing image: {e}")
        
        # Process text file with table
        print("\nProcessing TEXT file with table:")
        try:
            # First parse with universal parser
            parser = UniversalParser()
            text_docs = parser.parse(files['text_with_table'])
            
            if text_docs:
                text_doc = text_docs[0]
                print(f"  Content preview: {text_doc.content[:50]}...")
                
                # Extract tables from document
                table_docs = processor.extract_tables_from_document(text_doc.content)
                print(f"  Extracted {len(table_docs)} table(s)")
                
                if table_docs:
                    table_doc = table_docs[0]
                    print(f"  Table content preview: {table_doc.content[:50]}...")
                    print(f"  Table metadata: {table_doc.metadata}")
        except Exception as e:
            print(f"  Error processing text with table: {e}")
        
        # Process audio file
        print("\nProcessing AUDIO file:")
        try:
            audio_doc = processor.process_audio(files['audio'])
            print(f"  Content: {audio_doc.content[:50]}...")
            print(f"  Metadata: {audio_doc.metadata}")
        except Exception as e:
            print(f"  Error processing audio: {e}")
        
        # Process video file
        print("\nProcessing VIDEO file:")
        try:
            video_doc = processor.process_video(files['video'])
            print(f"  Content: {video_doc.content[:50]}...")
            print(f"  Metadata: {video_doc.metadata}")
        except Exception as e:
            print(f"  Error processing video: {e}")
        
        # Process multimodal document (universal processing)
        print("\nProcessing MULTIMODAL document (universal processing):")
        try:
            multimodal_docs = processor.process_multimodal_document(files['image'])
            print(f"  Processed into {len(multimodal_docs)} document(s)")
            
            if multimodal_docs:
                doc = multimodal_docs[0]
                print(f"  Content: {doc.content[:50]}...")
                print(f"  Metadata: {doc.metadata}")
        except Exception as e:
            print(f"  Error processing multimodal document: {e}")
        
        # Demonstrate table processing
        print("\nDemonstrating TABLE processing:")
        try:
            # Create sample table data
            table_data = [
                ["Product", "Price", "Quantity"],
                ["Apple", "$1.00", "10"],
                ["Banana", "$0.50", "20"],
                ["Orange", "$0.75", "15"]
            ]
            
            table_doc = processor.process_table(table_data)
            print(f"  Table content: {table_doc.content[:50]}...")
            print(f"  Table metadata: {table_doc.metadata}")
        except Exception as e:
            print(f"  Error processing table: {e}")
        
        # Demonstrate HTML table processing
        print("\nDemonstrating HTML TABLE processing:")
        try:
            html_content = """
            <table>
                <tr><th>Product</th><th>Price</th><th>Quantity</th></tr>
                <tr><td>Apple</td><td>$1.00</td><td>10</td></tr>
                <tr><td>Banana</td><td>$0.50</td><td>20</td></tr>
                <tr><td>Orange</td><td>$0.75</td><td>15</td></tr>
            </table>
            """
            
            html_table_doc = processor.process_html_table(html_content)
            print(f"  HTML table content: {html_table_doc.content[:50]}...")
            print(f"  HTML table metadata: {html_table_doc.metadata}")
        except Exception as e:
            print(f"  Error processing HTML table: {e}")
        
        print("\n" + "=" * 45)
        print("Multimodal processing example completed successfully!")
        print("\nNexusRAG now supports:")
        print("- Image understanding & captioning (OCR fallback)")
        print("- Audio/video transcription (with Whisper)")
        print("- Table & chart comprehension")
        print("- PDF with math/formula understanding (with Nougat)")
        print("- Universal multimodal processing")


if __name__ == "__main__":
    main()
