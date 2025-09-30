from typing import List, Dict, Any, Optional
from ..llms.base import BaseLLM
from ..vectorstores.base import BaseVectorStore
from ..parsers.base import Document
import json


class MultiStepReasoner:
    """Multi-step reasoning engine with iterative refinement."""
    
    def __init__(self, llm: BaseLLM, vector_store: BaseVectorStore):
        """Initialize the multi-step reasoner.
        
        Args:
            llm (BaseLLM): Language model for generation
            vector_store (BaseVectorStore): Vector store for retrieval
        """
        self.llm = llm
        self.vector_store = vector_store
        self.reasoning_history: List[Dict[str, Any]] = []
    
    def reason(self, query: str, max_steps: int = 5, context: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Perform multi-step reasoning to answer a query.
        
        Args:
            query (str): The query to answer
            max_steps (int): Maximum number of reasoning steps
            context (List[Dict[str, Any]]): Optional initial context
            
        Returns:
            Dict[str, Any]: Reasoning results with steps and final answer
        """
        # Initialize reasoning session
        session_id = len(self.reasoning_history)
        reasoning_session = {
            "session_id": session_id,
            "query": query,
            "steps": [],
            "final_answer": "",
            "confidence": 0.0
        }
        
        # Retrieve initial context if not provided
        if context is None:
            context = self.vector_store.query(query, top_k=10)
        
        # Step 1: Initial analysis
        step_result = self._initial_analysis(query, context)
        reasoning_session["steps"].append(step_result)
        
        # Subsequent refinement steps
        current_state = step_result
        for step in range(1, max_steps):
            step_result = self._refinement_step(query, context, current_state, step)
            reasoning_session["steps"].append(step_result)
            
            # Check if we should stop early
            if self._should_stop_early(step_result):
                break
            
            current_state = step_result
        
        # Final synthesis
        final_result = self._synthesize_answer(query, context, reasoning_session["steps"])
        reasoning_session["final_answer"] = final_result["answer"]
        reasoning_session["confidence"] = final_result["confidence"]
        reasoning_session["evidence"] = final_result["evidence"]
        
        # Store reasoning session
        self.reasoning_history.append(reasoning_session)
        
        return reasoning_session
    
    def _initial_analysis(self, query: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform initial analysis of the query.
        
        Args:
            query (str): The query to analyze
            context (List[Dict[str, Any]]): Context documents
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        prompt = f"""Analyze the following query and provide an initial assessment:

Query: {query}

Context:
{self._format_context(context)}

Please provide:
1. Key entities and concepts in the query
2. Relevant context documents
3. Initial approach to answer the query
4. Potential challenges or ambiguities

Format your response as JSON with keys: entities, relevant_context, approach, challenges"""
        
        response = self.llm.generate(prompt)
        
        try:
            analysis = json.loads(response)
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            analysis = {
                "entities": [],
                "relevant_context": [],
                "approach": response,
                "challenges": []
            }
        
        return {
            "step": 0,
            "type": "initial_analysis",
            "analysis": analysis,
            "prompt": prompt,
            "response": response
        }
    
    def _refinement_step(self, query: str, context: List[Dict[str, Any]], 
                        previous_state: Dict[str, Any], step: int) -> Dict[str, Any]:
        """Perform a refinement step.
        
        Args:
            query (str): The query to refine
            context (List[Dict[str, Any]]): Context documents
            previous_state (Dict[str, Any]): Previous reasoning state
            step (int): Current step number
            
        Returns:
            Dict[str, Any]: Refinement results
        """
        prompt = f"""Refine the analysis of the following query based on previous reasoning:

Query: {query}

Previous Analysis:
{json.dumps(previous_state.get('analysis', {}), indent=2)}

Context:
{self._format_context(context)}

Please provide:
1. Refined understanding of key entities and concepts
2. Additional relevant context documents
3. Improved approach to answer the query
4. Resolved challenges or new insights

Format your response as JSON with keys: entities, relevant_context, approach, insights"""
        
        response = self.llm.generate(prompt)
        
        try:
            refinement = json.loads(response)
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            refinement = {
                "entities": [],
                "relevant_context": [],
                "approach": response,
                "insights": []
            }
        
        return {
            "step": step,
            "type": "refinement",
            "refinement": refinement,
            "prompt": prompt,
            "response": response
        }
    
    def _synthesize_answer(self, query: str, context: List[Dict[str, Any]], 
                          steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesize a final answer from all reasoning steps.
        
        Args:
            query (str): The query to answer
            context (List[Dict[str, Any]]): Context documents
            steps (List[Dict[str, Any]]): Reasoning steps
            
        Returns:
            Dict[str, Any]: Final answer with confidence and evidence
        """
        prompt = f"""Synthesize a final answer to the following query based on all reasoning steps:

Query: {query}

Reasoning Steps:
{json.dumps(steps, indent=2)}

Context:
{self._format_context(context)}

Please provide:
1. A comprehensive answer to the query
2. Confidence score (0.0 to 1.0)
3. Evidence from context documents supporting your answer
4. Limitations or uncertainties in your answer

Format your response as JSON with keys: answer, confidence, evidence, limitations"""
        
        response = self.llm.generate(prompt)
        
        try:
            synthesis = json.loads(response)
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            synthesis = {
                "answer": response,
                "confidence": 0.5,
                "evidence": [],
                "limitations": "Unable to parse structured response"
            }
        
        return {
            "answer": synthesis.get("answer", response),
            "confidence": synthesis.get("confidence", 0.5),
            "evidence": synthesis.get("evidence", []),
            "limitations": synthesis.get("limitations", "")
        }
    
    def _should_stop_early(self, step_result: Dict[str, Any]) -> bool:
        """Determine if reasoning should stop early.
        
        Args:
            step_result (Dict[str, Any]): Current step result
            
        Returns:
            bool: True if reasoning should stop
        """
        # Simple heuristic: stop if confidence is high enough
        if "refinement" in step_result:
            # This is a bit tricky without structured data, so we'll use a simple approach
            response = step_result.get("response", "").lower()
            if "sufficient" in response or "adequate" in response or "complete" in response:
                return True
        
        return False
    
    def _format_context(self, context: List[Dict[str, Any]]) -> str:
        """Format context for inclusion in prompts.
        
        Args:
            context (List[Dict[str, Any]]): Context documents
            
        Returns:
            str: Formatted context string
        """
        formatted = ""
        for i, doc in enumerate(context[:5]):  # Limit to first 5 documents
            content = doc.get("content", "")[:200] + "..." if len(doc.get("content", "")) > 200 else doc.get("content", "")
            formatted += f"Document {i+1} (Score: {doc.get('score', 0.0):.2f}):\n{content}\n\n"
        return formatted.strip()
    
    def get_reasoning_history(self) -> List[Dict[str, Any]]:
        """Get the reasoning history.
        
        Returns:
            List[Dict[str, Any]]: Reasoning history
        """
        return self.reasoning_history.copy()
    
    def clear_history(self) -> None:
        """Clear the reasoning history."""
        self.reasoning_history = []
