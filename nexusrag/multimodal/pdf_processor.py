from typing import List, Dict, Any
from ..parsers.base import Document
import os


class PDFProcessor:
    """Advanced PDF processor with math/formula understanding using Nougat."""
    
    def __init__(self, model_type: str = "nougat"):
        """Initialize the PDF processor.
        
        Args:
            model_type (str): Type of model to use ("nougat")
        """
        self.model_type = model_type
        self.model = None
        self.processor = None
        
    def _load_model(self):
        """Load the Nougat model."""
        if self.model is not None:
            return
            
        try:
            # Try to import Nougat
            from nougat import NougatModel
            from nougat.utils.checkpoint import get_checkpoint
            from nougat.utils.dataset import ImageDataset
            import torch
            
            # Load checkpoint
            checkpoint = get_checkpoint("0.1.0-base")
            
            # Initialize model
            self.model = NougatModel.from_pretrained(checkpoint)
            self.model.eval()
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.model.to("cuda")
                
        except ImportError as e:
            raise ImportError(
                f"To use advanced PDF processing with {self.model_type}, you need to install Nougat. "
                f"Please visit https://github.com/facebookresearch/nougat for installation instructions. "
                f"Error: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Nougat model: {e}")
    
    def process_pdf(self, pdf_path: str) -> List[Document]:
        """Process a PDF file with math/formula understanding.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            List[Document]: List of documents containing PDF content with math/formulas
        """
        # Try to use advanced model if available
        try:
            self._load_model()
            return self._process_with_nougat(pdf_path)
        except Exception as e:
            # Fallback to existing PDF parser
            print(f"Falling back to basic PDF parser due to error: {e}")
            return self._process_with_basic_parser(pdf_path)
    
    def _process_with_nougat(self, pdf_path: str) -> List[Document]:
        """Process PDF using Nougat model.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            List[Document]: List of documents containing PDF content
        """
        try:
            from nougat.utils.dataset import ImageDataset
            from torch.utils.data import DataLoader
            import torch
        except ImportError:
            raise ImportError(
                "To use Nougat, you need to install the required dependencies. "
                "Please visit https://github.com/facebookresearch/nougat for installation instructions."
            )
        
        # Create dataset
        dataset = ImageDataset(pdf_path, self.model.config)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        documents = []
        
        # Process each page
        for idx, (image, _) in enumerate(dataloader):
            # Move to GPU if available
            if torch.cuda.is_available():
                image = image.to("cuda")
            
            # Generate prediction
            outputs = self.model.generate(
                image,
                temperature=self.model.config.temperature,
                max_length=self.model.config.max_length,
                min_length=self.model.config.min_length,
                early_stopping=self.model.config.early_stopping,
                num_beams=self.model.config.num_beams,
                pad_token_id=self.model.config.pad_token_id,
                eos_token_id=self.model.config.eos_token_id,
                use_cache=True,
                decoder_start_token_id=self.model.config.decoder_start_token_id,
            )
            
            # Decode output
            prediction = self.model.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Create document
            metadata = {
                "source": pdf_path,
                "content_type": "pdf",
                "media_type": "text",
                "page": idx + 1,
                "processing_method": "nougat",
                "model_type": self.model_type
            }
            
            documents.append(Document(content=prediction.strip(), metadata=metadata))
        
        return documents
    
    def _process_with_basic_parser(self, pdf_path: str) -> List[Document]:
        """Process PDF using basic parser as fallback.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            List[Document]: List of documents containing PDF content
        """
        from ..parsers.pdf import PDFParser
        
        parser = PDFParser()
        documents = parser.parse(pdf_path)
        
        # Enhance metadata
        for doc in documents:
            doc.metadata["processing_method"] = "basic"
            doc.metadata["fallback"] = True
        
        return documents
