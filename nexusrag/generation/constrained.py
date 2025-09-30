from typing import List, Dict, Any, Optional, Callable
from ..llms.base import BaseLLM
import re
import json


class ConstrainedGenerator:
    """Constrained generation engine inspired by LMQL approaches."""
    
    def __init__(self, llm: BaseLLM):
        """Initialize the constrained generator.
        
        Args:
            llm (BaseLLM): Language model for generation
        """
        self.llm = llm
    
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
        for attempt in range(max_attempts):
            # Generate response
            response = self.llm.generate(prompt)
            
            # Check constraints
            constraint_check = self._check_constraints(response, constraints)
            
            if constraint_check["satisfied"]:
                return {
                    "response": response,
                    "constraints_satisfied": True,
                    "constraint_check": constraint_check,
                    "attempts": attempt + 1
                }
            
            # If constraints not satisfied, modify prompt for next attempt
            prompt = self._refine_prompt(prompt, response, constraint_check)
        
        # If all attempts failed, return last result with warning
        return {
            "response": response,
            "constraints_satisfied": False,
            "constraint_check": constraint_check,
            "attempts": max_attempts,
            "warning": "Constraints not fully satisfied after maximum attempts"
        }
    
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
        # Add JSON formatting instructions to prompt
        json_prompt = f"""{prompt}

Please provide your response in valid JSON format.
{'Schema: ' + json.dumps(schema, indent=2) if schema else 'Please use appropriate JSON structure.'}

Response:"""
        
        for attempt in range(max_attempts):
            # Generate response
            response = self.llm.generate(json_prompt)
            
            # Try to parse as JSON
            try:
                parsed_json = json.loads(response)
                
                # If schema provided, validate against it
                if schema:
                    schema_check = self._validate_json_schema(parsed_json, schema)
                    if schema_check["valid"]:
                        return {
                            "response": parsed_json,
                            "valid_json": True,
                            "schema_valid": True,
                            "attempts": attempt + 1
                        }
                    else:
                        # Refine prompt with schema feedback
                        json_prompt = self._refine_json_prompt(json_prompt, response, schema_check)
                        continue
                else:
                    return {
                        "response": parsed_json,
                        "valid_json": True,
                        "schema_valid": None,
                        "attempts": attempt + 1
                    }
            except json.JSONDecodeError as e:
                # Refine prompt with JSON formatting feedback
                json_prompt = self._refine_json_prompt(json_prompt, response, {"valid": False, "error": str(e)})
        
        # If all attempts failed, return error
        return {
            "response": response,
            "valid_json": False,
            "schema_valid": False,
            "attempts": max_attempts,
            "error": "Unable to generate valid JSON after maximum attempts"
        }
    
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
        # Add regex constraint to prompt
        regex_prompt = f"""{prompt}

Please ensure your response matches the following pattern: {pattern}"""
        
        for attempt in range(max_attempts):
            # Generate response
            response = self.llm.generate(regex_prompt)
            
            # Check regex match
            if re.match(pattern, response):
                return {
                    "response": response,
                    "pattern_matched": True,
                    "attempts": attempt + 1
                }
            
            # Refine prompt with regex feedback
            regex_prompt = self._refine_regex_prompt(regex_prompt, response, pattern)
        
        # If all attempts failed, return error
        return {
            "response": response,
            "pattern_matched": False,
            "attempts": max_attempts,
            "error": "Unable to generate response matching pattern after maximum attempts"
        }
    
    def _check_constraints(self, response: str, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Check if response satisfies constraints.
        
        Args:
            response (str): Generated response
            constraints (Dict[str, Any]): Constraints to check
            
        Returns:
            Dict[str, Any]: Constraint check results
        """
        results = {"satisfied": True, "violations": []}
        
        # Check length constraints
        if "max_length" in constraints:
            if len(response) > constraints["max_length"]:
                results["satisfied"] = False
                results["violations"].append(f"Response too long: {len(response)} > {constraints['max_length']}")
        
        if "min_length" in constraints:
            if len(response) < constraints["min_length"]:
                results["satisfied"] = False
                results["violations"].append(f"Response too short: {len(response)} < {constraints['min_length']}")
        
        # Check keyword constraints
        if "required_keywords" in constraints:
            missing_keywords = []
            for keyword in constraints["required_keywords"]:
                if keyword not in response:
                    missing_keywords.append(keyword)
            
            if missing_keywords:
                results["satisfied"] = False
                results["violations"].append(f"Missing required keywords: {missing_keywords}")
        
        if "forbidden_keywords" in constraints:
            forbidden_found = []
            for keyword in constraints["forbidden_keywords"]:
                if keyword in response:
                    forbidden_found.append(keyword)
            
            if forbidden_found:
                results["satisfied"] = False
                results["violations"].append(f"Forbidden keywords found: {forbidden_found}")
        
        # Check custom constraint functions
        if "custom_constraints" in constraints:
            for constraint_func in constraints["custom_constraints"]:
                if callable(constraint_func):
                    try:
                        satisfied, violation = constraint_func(response)
                        if not satisfied:
                            results["satisfied"] = False
                            results["violations"].append(violation)
                    except Exception as e:
                        results["satisfied"] = False
                        results["violations"].append(f"Constraint function error: {str(e)}")
        
        return results
    
    def _validate_json_schema(self, data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate JSON data against a schema.
        
        Args:
            data (Dict[str, Any]): JSON data
            schema (Dict[str, Any]): Schema to validate against
            
        Returns:
            Dict[str, Any]: Validation results
        """
        results = {"valid": True, "errors": []}
        
        # Check required fields
        if "required" in schema:
            for field in schema["required"]:
                if field not in data:
                    results["valid"] = False
                    results["errors"].append(f"Missing required field: {field}")
        
        # Check field types
        if "properties" in schema:
            for field, field_schema in schema["properties"].items():
                if field in data:
                    expected_type = field_schema.get("type")
                    if expected_type:
                        actual_type = type(data[field]).__name__
                        if expected_type == "string" and actual_type != "str":
                            results["valid"] = False
                            results["errors"].append(f"Field '{field}' should be string, got {actual_type}")
                        elif expected_type == "number" and actual_type not in ["int", "float"]:
                            results["valid"] = False
                            results["errors"].append(f"Field '{field}' should be number, got {actual_type}")
                        elif expected_type == "boolean" and actual_type != "bool":
                            results["valid"] = False
                            results["errors"].append(f"Field '{field}' should be boolean, got {actual_type}")
                        elif expected_type == "array" and actual_type != "list":
                            results["valid"] = False
                            results["errors"].append(f"Field '{field}' should be array, got {actual_type}")
                        elif expected_type == "object" and actual_type != "dict":
                            results["valid"] = False
                            results["errors"].append(f"Field '{field}' should be object, got {actual_type}")
        
        return results
    
    def _refine_prompt(self, prompt: str, response: str, constraint_check: Dict[str, Any]) -> str:
        """Refine prompt based on constraint violations.
        
        Args:
            prompt (str): Original prompt
            response (str): Generated response
            constraint_check (Dict[str, Any]): Constraint check results
            
        Returns:
            str: Refined prompt
        """
        violations = constraint_check.get("violations", [])
        if not violations:
            return prompt
        
        # Add constraint feedback to prompt
        feedback = "\n\nPlease correct the following issues in your response:\n"
        for violation in violations:
            feedback += f"- {violation}\n"
        
        return f"{prompt}{feedback}"
    
    def _refine_json_prompt(self, prompt: str, response: str, validation_result: Dict[str, Any]) -> str:
        """Refine JSON prompt based on validation results.
        
        Args:
            prompt (str): Original prompt
            response (str): Generated response
            validation_result (Dict[str, Any]): Validation results
            
        Returns:
            str: Refined prompt
        """
        error = validation_result.get("error", "")
        errors = validation_result.get("errors", [])
        
        feedback = "\n\nPlease provide a valid JSON response. "
        if error:
            feedback += f"Error: {error}. "
        if errors:
            feedback += f"Validation errors: {', '.join(errors)}. "
        
        feedback += "Ensure your response is properly formatted JSON."
        
        return f"{prompt}{feedback}"
    
    def _refine_regex_prompt(self, prompt: str, response: str, pattern: str) -> str:
        """Refine regex prompt based on pattern mismatch.
        
        Args:
            prompt (str): Original prompt
            response (str): Generated response
            pattern (str): Regex pattern
            
        Returns:
            str: Refined prompt
        """
        feedback = f"\n\nPlease ensure your response matches the required pattern: {pattern}"
        return f"{prompt}{feedback}"
