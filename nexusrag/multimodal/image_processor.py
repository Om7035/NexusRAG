from typing import List, Dict, Any
from ..parsers.base import Document
import os


class ImageProcessor:
    """Advanced image processor with captioning capabilities."""
    
    def __init__(self, model_type: str = "blip"):
        """Initialize the image processor.
        
        Args:
            model_type (str): Type of model to use ("blip", "llava", etc.)
        """
        self.model_type = model_type
        self.model = None
        
    def _load_model(self):
        """Load the image understanding model."""
        if self.model is not None:
            return
            
        try:
            if self.model_type == "blip":
                # Try to import BLIP-2
                from transformers import Blip2Processor, Blip2ForConditionalGeneration
                from PIL import Image
                import torch
                
                self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
                self.model = Blip2ForConditionalGeneration.from_pretrained(
                    "Salesforce/blip2-opt-2.7b", 
                    torch_dtype=torch.float16
                )
                self.model.to("cuda" if torch.cuda.is_available() else "cpu")
                self.image_module = Image
            elif self.model_type == "llava":
                # Placeholder for LLaVA integration
                raise NotImplementedError("LLaVA integration not yet implemented")
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
        except ImportError as e:
            raise ImportError(
                f"To use advanced image processing with {self.model_type}, you need to install additional dependencies. "
                f"Error: {e}"
            )
    
    def process_image(self, image_path: str) -> Document:
        """Process an image and generate a caption.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            Document: Document containing image caption and metadata
        """
        # For now, we'll use the existing OCR approach as fallback
        try:
            from PIL import Image
            import pytesseract
        except ImportError:
            raise ImportError(
                "To process images, you need to install PIL and pytesseract. "
                "Please run: pip install Pillow pytesseract"
            )
        
        # Try to use advanced model if available
        try:
            self._load_model()
            return self._process_with_model(image_path)
        except Exception as e:
            # Fallback to OCR
            print(f"Falling back to OCR due to error: {e}")
            return self._process_with_ocr(image_path)
    
    def _process_with_model(self, image_path: str) -> Document:
        """Process image using advanced model.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            Document: Document containing image caption
        """
        # Load image
        image = self.image_module.open(image_path)
        
        # Generate caption
        inputs = self.processor(image, return_tensors="pt").to(self.model.device)
        
        # Generate caption
        generated_ids = self.model.generate(**inputs, max_new_tokens=20)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        # Create document
        metadata = {
            "source": image_path,
            "content_type": "image",
            "media_type": "image",
            "processing_method": "model",
            "model_type": self.model_type
        }
        
        return Document(content=generated_text, metadata=metadata)
    
    def _process_with_ocr(self, image_path: str) -> Document:
        """Process image using OCR as fallback.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            Document: Document containing OCR text
        """
        try:
            from PIL import Image
            import pytesseract
        except ImportError:
            raise ImportError(
                "To process images with OCR, you need to install PIL and pytesseract. "
                "Please run: pip install Pillow pytesseract"
            )
        
        # Open and process image
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        
        # Create document
        metadata = {
            "source": image_path,
            "content_type": "image",
            "media_type": "image",
            "processing_method": "ocr"
        }
        
        return Document(content=text.strip(), metadata=metadata)
