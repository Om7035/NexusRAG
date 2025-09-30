from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class EvaluationMetrics:
    """Evaluation metrics for RAG systems."""
    
    @staticmethod
    def calculate_retrieval_metrics(relevant_docs: List[str], retrieved_docs: List[str], 
                                  k: int = None) -> Dict[str, float]:
        """Calculate retrieval metrics.
        
        Args:
            relevant_docs (List[str]): List of relevant document IDs
            retrieved_docs (List[str]): List of retrieved document IDs
            k (int): Cutoff for precision@k and recall@k
            
        Returns:
            Dict[str, float]: Retrieval metrics
        """
        if k is None:
            k = len(retrieved_docs)
        
        # Convert to sets for easier computation
        relevant_set = set(relevant_docs)
        retrieved_set = set(retrieved_docs[:k])
        
        # Calculate basic metrics
        intersection = relevant_set.intersection(retrieved_set)
        precision = len(intersection) / len(retrieved_set) if retrieved_set else 0.0
        recall = len(intersection) / len(relevant_set) if relevant_set else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Calculate precision@k and recall@k
        relevant_at_k = len([doc for doc in retrieved_docs[:k] if doc in relevant_set])
        precision_at_k = relevant_at_k / k if k > 0 else 0.0
        recall_at_k = relevant_at_k / len(relevant_set) if relevant_set else 0.0
        
        # Calculate mean reciprocal rank (MRR)
        mrr = 0.0
        for i, doc in enumerate(retrieved_docs):
            if doc in relevant_set:
                mrr = 1.0 / (i + 1)
                break
        
        # Calculate normalized discounted cumulative gain (NDCG)
        ndcg = EvaluationMetrics._calculate_ndcg(relevant_docs, retrieved_docs, k)
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "precision_at_k": precision_at_k,
            "recall_at_k": recall_at_k,
            "mrr": mrr,
            "ndcg": ndcg
        }
    
    @staticmethod
    def calculate_generation_metrics(generated_text: str, reference_text: str) -> Dict[str, float]:
        """Calculate generation metrics.
        
        Args:
            generated_text (str): Generated text
            reference_text (str): Reference text
            
        Returns:
            Dict[str, float]: Generation metrics
        """
        # Simple string similarity metrics
        from difflib import SequenceMatcher
        
        # Calculate similarity ratio
        similarity = SequenceMatcher(None, generated_text, reference_text).ratio()
        
        # Calculate exact match
        exact_match = 1.0 if generated_text.strip() == reference_text.strip() else 0.0
        
        # Calculate length ratio
        gen_len = len(generated_text)
        ref_len = len(reference_text)
        length_ratio = min(gen_len, ref_len) / max(gen_len, ref_len) if max(gen_len, ref_len) > 0 else 0.0
        
        return {
            "similarity": similarity,
            "exact_match": exact_match,
            "length_ratio": length_ratio
        }
    
    @staticmethod
    def calculate_reasoning_metrics(predicted_answers: List[str], 
                                  ground_truth_answers: List[str]) -> Dict[str, float]:
        """Calculate reasoning metrics.
        
        Args:
            predicted_answers (List[str]): Predicted answers
            ground_truth_answers (List[str]): Ground truth answers
            
        Returns:
            Dict[str, float]: Reasoning metrics
        """
        if len(predicted_answers) != len(ground_truth_answers):
            raise ValueError("Predicted and ground truth answers must have the same length")
        
        # Calculate accuracy
        correct = sum(1 for pred, truth in zip(predicted_answers, ground_truth_answers) 
                     if pred.strip().lower() == truth.strip().lower())
        accuracy = correct / len(predicted_answers) if predicted_answers else 0.0
        
        # Calculate average similarity
        similarities = []
        for pred, truth in zip(predicted_answers, ground_truth_answers):
            from difflib import SequenceMatcher
            similarity = SequenceMatcher(None, pred, truth).ratio()
            similarities.append(similarity)
        
        avg_similarity = np.mean(similarities) if similarities else 0.0
        
        return {
            "accuracy": accuracy,
            "avg_similarity": avg_similarity
        }
    
    @staticmethod
    def _calculate_ndcg(relevant_docs: List[str], retrieved_docs: List[str], k: int) -> float:
        """Calculate normalized discounted cumulative gain.
        
        Args:
            relevant_docs (List[str]): List of relevant document IDs
            retrieved_docs (List[str]): List of retrieved document IDs
            k (int): Cutoff for NDCG calculation
            
        Returns:
            float: NDCG score
        """
        if not relevant_docs or not retrieved_docs or k <= 0:
            return 0.0
        
        relevant_set = set(relevant_docs)
        
        # Calculate DCG
        dcg = 0.0
        for i, doc in enumerate(retrieved_docs[:k]):
            if doc in relevant_set:
                dcg += 1.0 / np.log2(i + 2)  # log2(1+position)
        
        # Calculate IDCG (ideal DCG)
        ideal_ranking = [1] * min(len(relevant_docs), k) + [0] * max(0, k - len(relevant_docs))
        idcg = 0.0
        for i, rel in enumerate(ideal_ranking):
            if rel > 0:
                idcg += 1.0 / np.log2(i + 2)
        
        # Calculate NDCG
        ndcg = dcg / idcg if idcg > 0 else 0.0
        return ndcg
    
    @staticmethod
    def calculate_multimodal_metrics(predictions: List[Dict[str, Any]], 
                                   ground_truth: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate multimodal metrics.
        
        Args:
            predictions (List[Dict[str, Any]]): Predictions with modality information
            ground_truth (List[Dict[str, Any]]): Ground truth with modality information
            
        Returns:
            Dict[str, float]: Multimodal metrics
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have the same length")
        
        # Calculate per-modality metrics
        modality_metrics = {}
        modalities = set()
        
        # Collect all modalities
        for pred in predictions:
            modalities.add(pred.get("modality", "unknown"))
        for truth in ground_truth:
            modalities.add(truth.get("modality", "unknown"))
        
        # Calculate metrics for each modality
        for modality in modalities:
            pred_for_modality = [p for p in predictions if p.get("modality", "unknown") == modality]
            truth_for_modality = [t for t in ground_truth if t.get("modality", "unknown") == modality]
            
            if pred_for_modality and truth_for_modality:
                # For simplicity, we'll just calculate accuracy for each modality
                correct = sum(1 for p, t in zip(pred_for_modality, truth_for_modality)
                             if p.get("content", "").strip().lower() == t.get("content", "").strip().lower())
                accuracy = correct / len(pred_for_modality) if pred_for_modality else 0.0
                
                modality_metrics[modality] = {
                    "accuracy": accuracy,
                    "count": len(pred_for_modality)
                }
        
        # Calculate overall metrics
        correct = sum(1 for p, t in zip(predictions, ground_truth)
                     if p.get("content", "").strip().lower() == t.get("content", "").strip().lower())
        overall_accuracy = correct / len(predictions) if predictions else 0.0
        
        return {
            "overall_accuracy": overall_accuracy,
            "per_modality": modality_metrics
        }
