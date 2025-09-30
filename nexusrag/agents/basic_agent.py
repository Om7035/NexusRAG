from typing import List, Dict, Any, Optional
from ..llms.base import BaseLLM
from ..vectorstores.base import BaseVectorStore
from ..parsers.base import Document


class BasicAgent:
    """Basic agent with reasoning and tool usage capabilities."""
    
    def __init__(self, llm: BaseLLM, vector_store: BaseVectorStore):
        """Initialize the basic agent.
        
        Args:
            llm (BaseLLM): Language model for generation
            vector_store (BaseVectorStore): Vector store for retrieval
        """
        self.llm = llm
        self.vector_store = vector_store
        self.memory: List[Dict[str, Any]] = []
        self.tools: Dict[str, Any] = {}
    
    def add_tool(self, name: str, tool_func: Any) -> None:
        """Add a tool to the agent's toolkit.
        
        Args:
            name (str): Name of the tool
            tool_func (Any): Function or callable that implements the tool
        """
        self.tools[name] = tool_func
    
    def think(self, query: str, max_steps: int = 3) -> str:
        """Perform multi-step reasoning to answer a query.
        
        Args:
            query (str): The query to answer
            max_steps (int): Maximum number of reasoning steps
            
        Returns:
            str: The final answer
        """
        # Retrieve relevant context
        context = self.vector_store.query(query, top_k=5)
        
        # Store in memory
        self.memory.append({
            "query": query,
            "context": context,
            "step": 0
        })
        
        # Initial response
        initial_prompt = f"""Answer the following question using the provided context:
        
Question: {query}

Context:
{self._format_context(context)}

Answer:"""
        
        response = self.llm.generate(initial_prompt)
        
        # Refinement steps
        current_answer = response
        for step in range(max_steps):
            refinement_prompt = f"""Review and improve the following answer to the question: "{query}"
            
Current answer: {current_answer}

Context:
{self._format_context(context)}

Please provide an improved answer or explain why the current answer is sufficient:"""
            
            refined_response = self.llm.generate(refinement_prompt)
            current_answer = refined_response
            
            # Store in memory
            self.memory.append({
                "step": step + 1,
                "prompt": refinement_prompt,
                "response": refined_response
            })
        
        return current_answer
    
    def _format_context(self, context: List[Dict[str, Any]]) -> str:
        """Format context for inclusion in prompts.
        
        Args:
            context (List[Dict[str, Any]]): Context documents
            
        Returns:
            str: Formatted context string
        """
        formatted = ""
        for i, doc in enumerate(context):
            formatted += f"Document {i+1}:\n{doc.get('content', '')}\n\n"
        return formatted.strip()
    
    def get_memory(self) -> List[Dict[str, Any]]:
        """Get the agent's memory/history.
        
        Returns:
            List[Dict[str, Any]]: Agent's memory
        """
        return self.memory
    
    def clear_memory(self) -> None:
        """Clear the agent's memory."""
        self.memory = []
    
    def use_tool(self, tool_name: str, *args, **kwargs) -> Any:
        """Use a tool from the agent's toolkit.
        
        Args:
            tool_name (str): Name of the tool to use
            *args: Positional arguments for the tool
            **kwargs: Keyword arguments for the tool
            
        Returns:
            Any: Result of tool execution
        """
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found in agent's toolkit")
        
        tool_func = self.tools[tool_name]
        return tool_func(*args, **kwargs)
