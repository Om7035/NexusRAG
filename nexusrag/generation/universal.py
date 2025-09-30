from typing import List, Dict, Any, Optional
from ..llms.base import BaseLLM
from ..vectorstores.base import BaseVectorStore
from ..parsers.base import Document
from ..reasoning.multi_step import MultiStepReasoner
from ..reasoning.citation import CitationEngine
from .constrained import ConstrainedGenerator


class UniversalGenerator:
    """Universal generation and reasoning engine combining all capabilities."""
    
    def __init__(self, llm: BaseLLM, vector_store: BaseVectorStore):
        """Initialize the universal generator.
        
        Args:
            llm (BaseLLM): Language model for generation
            vector_store (BaseVectorStore): Vector store for retrieval
        """
        self.llm = llm
        self.vector_store = vector_store
        
        # Initialize components
        self.reasoner = MultiStepReasoner(llm, vector_store)
        self.citation_engine = CitationEngine()
        self.constrained_generator = ConstrainedGenerator(llm)
    
    def generate_answer(self, query: str, context: List[Dict[str, Any]] = None, 
                       max_reasoning_steps: int = 5) -> Dict[str, Any]:
        """Generate a comprehensive answer with reasoning and citations.
        
        Args:
            query (str): Query to answer
            context (List[Dict[str, Any]]): Optional context documents
            max_reasoning_steps (int): Maximum reasoning steps
            
        Returns:
            Dict[str, Any]: Comprehensive answer with reasoning and citations
        """
        # Retrieve context if not provided
        if context is None:
            context = self.vector_store.query(query, top_k=10)
        
        # Perform multi-step reasoning
        reasoning_result = self.reasoner.reason(query, max_reasoning_steps, context)
        
        # Extract final answer
        final_answer = reasoning_result.get("final_answer", "")
        
        # Generate citations for the answer
        citation_report = self.citation_engine.generate_citations(final_answer, context)
        
        # Combine results
        comprehensive_answer = {
            "query": query,
            "answer": final_answer,
            "confidence": reasoning_result.get("confidence", 0.0),
            "reasoning_steps": reasoning_result.get("steps", []),
            "citation_report": citation_report,
            "context_used": len(context),
            "timestamp": self._get_timestamp()
        }
        
        return comprehensive_answer
    
    def generate_with_constraints(self, prompt: str, constraints: Dict[str, Any], 
                                max_attempts: int = 3) -> Dict[str, Any]:
        """Generate text with specified constraints.
        
        Args:
            prompt (str): Generation prompt
            constraints (Dict[str, Any]): Constraints for generation
            max_attempts (int): Maximum number of generation attempts
            
        Returns:
            Dict[str, Any]: Generation results with constraint checking
        """
        return self.constrained_generator.generate_with_constraints(prompt, constraints, max_attempts)
    
    def generate_json(self, prompt: str, schema: Dict[str, Any] = None, 
                     max_attempts: int = 3) -> Dict[str, Any]:
        """Generate JSON-structured output.
        
        Args:
            prompt (str): Generation prompt
            schema (Dict[str, Any]): Expected JSON schema
            max_attempts (int): Maximum number of generation attempts
            
        Returns:
            Dict[str, Any]: JSON generation results
        """
        return self.constrained_generator.generate_json(prompt, schema, max_attempts)
    
    def generate_with_regex(self, prompt: str, pattern: str, 
                           max_attempts: int = 3) -> Dict[str, Any]:
        """Generate text that matches a regex pattern.
        
        Args:
            prompt (str): Generation prompt
            pattern (str): Regex pattern to match
            max_attempts (int): Maximum number of generation attempts
            
        Returns:
            Dict[str, Any]: Regex-constrained generation results
        """
        return self.constrained_generator.generate_with_regex(prompt, pattern, max_attempts)
    
    def verify_claim(self, claim: str, context: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Verify a claim against provided context.
        
        Args:
            claim (str): Claim to verify
            context (List[Dict[str, Any]]): Context documents
            
        Returns:
            Dict[str, Any]: Verification results
        """
        # Retrieve context if not provided
        if context is None:
            # For claim verification, we'll search broadly
            context = self.vector_store.query(claim, top_k=20)
        
        return self.citation_engine.verify_claim(claim, context)
    
    def interactive_reasoning(self, query: str, max_steps: int = 5) -> Dict[str, Any]:
        """Perform interactive reasoning with step-by-step feedback.
        
        Args:
            query (str): Query to reason about
            max_steps (int): Maximum reasoning steps
            
        Returns:
            Dict[str, Any]: Interactive reasoning session
        """
        # This would typically involve user interaction, but we'll simulate it
        # In a real implementation, this would pause for user feedback at each step
        
        reasoning_result = self.reasoner.reason(query, max_steps)
        
        # Add interactive elements to the result
        interactive_result = reasoning_result.copy()
        interactive_result["interactive"] = True
        interactive_result["user_feedback_requested"] = False  # In simulation, no feedback requested
        
        return interactive_result
    
    def get_reasoning_history(self) -> List[Dict[str, Any]]:
        """Get the reasoning history.
        
        Returns:
            List[Dict[str, Any]]: Reasoning history
        """
        return self.reasoner.get_reasoning_history()
    
    def get_citation_report(self) -> Dict[str, Any]:
        """Get a report of all citations.
        
        Returns:
            Dict[str, Any]: Citation report
        """
        return self.citation_engine.get_citation_report()
    
    def clear_history(self) -> None:
        """Clear all history and caches."""
        self.reasoner.clear_history()
        self.citation_engine.clear_citations()
    
    def _get_timestamp(self) -> str:
        """Get current timestamp.
        
        Returns:
            str: Timestamp
        """
        from datetime import datetime
        return datetime.now().isoformat()
