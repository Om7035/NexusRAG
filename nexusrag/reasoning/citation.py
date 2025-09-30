from typing import List, Dict, Any, Optional
from ..parsers.base import Document
import re
import hashlib


class CitationEngine:
    """Citation and verification engine for RAG responses."""
    
    def __init__(self):
        """Initialize the citation engine."""
        self.citations = []
        self.verification_cache = {}
    
    def add_citation(self, content: str, source: str, metadata: Dict[str, Any] = None) -> str:
        """Add a citation and return a citation ID.
        
        Args:
            content (str): Content being cited
            source (str): Source of the content
            metadata (Dict[str, Any]): Additional metadata
            
        Returns:
            str: Citation ID
        """
        # Generate citation ID based on content hash
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        citation_id = f"cite_{content_hash}"
        
        citation = {
            "id": citation_id,
            "content": content,
            "source": source,
            "metadata": metadata or {},
            "timestamp": self._get_timestamp()
        }
        
        self.citations.append(citation)
        return citation_id
    
    def verify_claim(self, claim: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Verify a claim against provided context.
        
        Args:
            claim (str): Claim to verify
            context (List[Dict[str, Any]]): Context documents
            
        Returns:
            Dict[str, Any]: Verification results
        """
        # Check cache first
        claim_hash = hashlib.md5(claim.encode()).hexdigest()
        if claim_hash in self.verification_cache:
            return self.verification_cache[claim_hash]
        
        # Extract key entities from claim
        entities = self._extract_entities(claim)
        
        # Search for supporting evidence
        supporting_evidence = []
        contradicting_evidence = []
        
        for doc in context:
            doc_content = doc.get("content", "").lower()
            claim_lower = claim.lower()
            
            # Check for direct matches
            if claim_lower in doc_content:
                supporting_evidence.append({
                    "document_id": self._get_doc_id(doc),
                    "content": doc.get("content", ""),
                    "score": 1.0,
                    "type": "direct_match"
                })
            
            # Check for entity matches
            entity_matches = 0
            for entity in entities:
                if entity.lower() in doc_content:
                    entity_matches += 1
            
            if entity_matches > 0:
                score = entity_matches / len(entities) if entities else 0
                if score > 0.5:  # More than half of entities match
                    supporting_evidence.append({
                        "document_id": self._get_doc_id(doc),
                        "content": doc.get("content", ""),
                        "score": score,
                        "type": "entity_match"
                    })
        
        # Determine verification result
        if supporting_evidence:
            # Sort by score
            supporting_evidence.sort(key=lambda x: x["score"], reverse=True)
            verification_result = {
                "claim": claim,
                "verified": True,
                "confidence": min(1.0, len(supporting_evidence) / 3.0),
                "supporting_evidence": supporting_evidence[:3],  # Top 3
                "contradicting_evidence": contradicting_evidence[:3],  # Top 3
                "timestamp": self._get_timestamp()
            }
        else:
            verification_result = {
                "claim": claim,
                "verified": False,
                "confidence": 0.0,
                "supporting_evidence": [],
                "contradicting_evidence": [],
                "timestamp": self._get_timestamp()
            }
        
        # Cache result
        self.verification_cache[claim_hash] = verification_result
        
        return verification_result
    
    def generate_citations(self, response: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate citations for a response based on context.
        
        Args:
            response (str): Response text
            context (List[Dict[str, Any]]): Context documents
            
        Returns:
            Dict[str, Any]: Response with citations
        """
        # Extract claims from response
        claims = self._extract_claims(response)
        
        # Verify each claim
        verified_claims = []
        for claim in claims:
            verification = self.verify_claim(claim, context)
            verified_claims.append({
                "claim": claim,
                "verification": verification
            })
        
        # Generate citation report
        citation_report = {
            "response": response,
            "claims": verified_claims,
            "total_claims": len(claims),
            "verified_claims": len([c for c in verified_claims if c["verification"]["verified"]]),
            "citation_rate": len([c for c in verified_claims if c["verification"]["verified"]]) / len(claims) if claims else 0.0,
            "timestamp": self._get_timestamp()
        }
        
        return citation_report
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text.
        
        Args:
            text (str): Text to extract entities from
            
        Returns:
            List[str]: Extracted entities
        """
        # Simple entity extraction using regex patterns
        # In a real implementation, you might use spaCy or similar
        entities = []
        
        # Extract quoted phrases
        quoted = re.findall(r'"([^"]+)"', text)
        entities.extend(quoted)
        
        # Extract capitalized words (potential proper nouns)
        capitalized = re.findall(r'\b[A-Z][a-z]+\b', text)
        entities.extend(capitalized)
        
        # Remove duplicates and return
        return list(set(entities))
    
    def _extract_claims(self, text: str) -> List[str]:
        """Extract factual claims from text.
        
        Args:
            text (str): Text to extract claims from
            
        Returns:
            List[str]: Extracted claims
        """
        # Simple claim extraction
        # In a real implementation, you might use more sophisticated NLP
        sentences = re.split(r'[.!?]+', text)
        claims = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                # Simple heuristic: sentences with verbs are likely claims
                if re.search(r'\b(is|are|was|were|has|have|had|will|would|could|should)\b', sentence, re.IGNORECASE):
                    claims.append(sentence)
        
        return claims
    
    def _get_doc_id(self, doc: Dict[str, Any]) -> str:
        """Generate a document ID.
        
        Args:
            doc (Dict[str, Any]): Document
            
        Returns:
            str: Document ID
        """
        content = doc.get("content", "")
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _get_timestamp(self) -> str:
        """Get current timestamp.
        
        Returns:
            str: Timestamp
        """
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_citation_report(self) -> Dict[str, Any]:
        """Get a report of all citations.
        
        Returns:
            Dict[str, Any]: Citation report
        """
        return {
            "total_citations": len(self.citations),
            "citations": self.citations,
            "verification_cache_size": len(self.verification_cache)
        }
    
    def clear_citations(self) -> None:
        """Clear all citations and verification cache."""
        self.citations = []
        self.verification_cache = {}
