"""
Evaluation metrics and utilities for recommendation systems.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import torch
from sklearn.metrics import precision_score, recall_score
from collections import defaultdict
import warnings


class RecommendationMetrics:
    """Class for computing recommendation metrics."""
    
    def __init__(self, k_values: List[int] = [5, 10, 20]):
        self.k_values = k_values
    
    def precision_at_k(self, y_true: List[int], y_pred: List[int], k: int) -> float:
        """Compute Precision@K."""
        if len(y_pred) == 0:
            return 0.0
        
        y_pred_k = y_pred[:k]
        hits = len(set(y_true) & set(y_pred_k))
        return hits / min(k, len(y_pred_k))
    
    def recall_at_k(self, y_true: List[int], y_pred: List[int], k: int) -> float:
        """Compute Recall@K."""
        if len(y_true) == 0:
            return 0.0
        
        y_pred_k = y_pred[:k]
        hits = len(set(y_true) & set(y_pred_k))
        return hits / len(y_true)
    
    def ndcg_at_k(self, y_true: List[int], y_pred: List[int], k: int) -> float:
        """Compute NDCG@K."""
        if len(y_pred) == 0:
            return 0.0
        
        y_pred_k = y_pred[:k]
        
        # Create relevance scores (1 for relevant items, 0 for others)
        relevance = [1 if item in y_true else 0 for item in y_pred_k]
        
        # Compute DCG
        dcg = sum(relevance[i] / np.log2(i + 2) for i in range(len(relevance)))
        
        # Compute IDCG (ideal DCG)
        ideal_relevance = [1] * min(len(y_true), k)
        idcg = sum(ideal_relevance[i] / np.log2(i + 2) for i in range(len(ideal_relevance)))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def map_at_k(self, y_true: List[int], y_pred: List[int], k: int) -> float:
        """Compute MAP@K."""
        if len(y_pred) == 0 or len(y_true) == 0:
            return 0.0
        
        y_pred_k = y_pred[:k]
        
        # Compute average precision
        precision_sum = 0.0
        hits = 0
        
        for i, item in enumerate(y_pred_k):
            if item in y_true:
                hits += 1
                precision_sum += hits / (i + 1)
        
        return precision_sum / len(y_true) if len(y_true) > 0 else 0.0
    
    def hit_rate_at_k(self, y_true: List[int], y_pred: List[int], k: int) -> float:
        """Compute Hit Rate@K."""
        if len(y_pred) == 0 or len(y_true) == 0:
            return 0.0
        
        y_pred_k = y_pred[:k]
        hits = len(set(y_true) & set(y_pred_k))
        return 1.0 if hits > 0 else 0.0
    
    def compute_all_metrics(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        """Compute all metrics for all k values."""
        metrics = {}
        
        for k in self.k_values:
            metrics[f'precision@{k}'] = self.precision_at_k(y_true, y_pred, k)
            metrics[f'recall@{k}'] = self.recall_at_k(y_true, y_pred, k)
            metrics[f'ndcg@{k}'] = self.ndcg_at_k(y_true, y_pred, k)
            metrics[f'map@{k}'] = self.map_at_k(y_true, y_pred, k)
            metrics[f'hit_rate@{k}'] = self.hit_rate_at_k(y_true, y_pred, k)
        
        return metrics


class ModelEvaluator:
    """Model evaluator for recommendation systems."""
    
    def __init__(self, model, device: str = 'cpu', k_values: List[int] = [5, 10, 20]):
        self.model = model
        self.device = device
        self.metrics = RecommendationMetrics(k_values)
        self.k_values = k_values
    
    def evaluate_model(self, test_loader, user_history: Dict, 
                      num_negatives: int = 100) -> Dict[str, float]:
        """Evaluate model on test set."""
        self.model.eval()
        
        all_metrics = defaultdict(list)
        
        with torch.no_grad():
            for batch in test_loader:
                user_ids = batch['user_idx'].to(self.device)
                item_ids = batch['item_idx'].to(self.device)
                ratings = batch['rating'].to(self.device)
                
                # Get predictions for positive items
                pos_predictions, _ = self.model(user_ids, item_ids)
                
                # For each user, get top-k recommendations
                for i, user_id in enumerate(user_ids):
                    user_id_val = user_id.item()
                    
                    # Get user's test items (positive items)
                    user_test_items = [item_ids[i].item()]
                    
                    # Generate negative samples
                    all_items = list(range(self.model.num_items))
                    user_train_items = user_history.get(user_id_val, [])
                    
                    # Remove items user has already interacted with
                    available_items = [item for item in all_items 
                                     if item not in user_train_items]
                    
                    if len(available_items) < num_negatives:
                        negative_items = available_items
                    else:
                        negative_items = np.random.choice(
                            available_items, num_negatives, replace=False
                        ).tolist()
                    
                    # Get predictions for all candidate items
                    candidate_items = user_test_items + negative_items
                    candidate_tensor = torch.tensor(candidate_items, device=self.device)
                    user_tensor = user_id.unsqueeze(0).repeat(len(candidate_items), 1).squeeze()
                    
                    candidate_predictions, _ = self.model(user_tensor, candidate_tensor)
                    
                    # Rank items by prediction score
                    item_scores = list(zip(candidate_items, candidate_predictions.cpu().numpy()))
                    item_scores.sort(key=lambda x: x[1], reverse=True)
                    ranked_items = [item for item, _ in item_scores]
                    
                    # Compute metrics
                    metrics = self.metrics.compute_all_metrics(user_test_items, ranked_items)
                    
                    for metric_name, value in metrics.items():
                        all_metrics[metric_name].append(value)
        
        # Average metrics across all users
        avg_metrics = {}
        for metric_name, values in all_metrics.items():
            avg_metrics[metric_name] = np.mean(values)
        
        return avg_metrics
    
    def evaluate_user(self, user_id: int, user_history: Dict, 
                     num_recommendations: int = 20) -> Tuple[List[int], Dict[str, float]]:
        """Evaluate recommendations for a specific user."""
        self.model.eval()
        
        with torch.no_grad():
            # Get all items user hasn't interacted with
            all_items = list(range(self.model.num_items))
            user_train_items = user_history.get(user_id, [])
            candidate_items = [item for item in all_items if item not in user_train_items]
            
            if len(candidate_items) == 0:
                return [], {}
            
            # Get predictions for all candidate items
            candidate_tensor = torch.tensor(candidate_items, device=self.device)
            user_tensor = torch.tensor([user_id], device=self.device).repeat(len(candidate_items))
            
            predictions, attention_weights = self.model(user_tensor, candidate_tensor)
            
            # Rank items by prediction score
            item_scores = list(zip(candidate_items, predictions.cpu().numpy()))
            item_scores.sort(key=lambda x: x[1], reverse=True)
            ranked_items = [item for item, _ in item_scores]
            
            # Get top recommendations
            recommendations = ranked_items[:num_recommendations]
            
            return recommendations, attention_weights.cpu().numpy()


class ModelComparison:
    """Class for comparing multiple models."""
    
    def __init__(self, k_values: List[int] = [5, 10, 20]):
        self.k_values = k_values
        self.results = {}
    
    def add_model_results(self, model_name: str, metrics: Dict[str, float]):
        """Add results for a model."""
        self.results[model_name] = metrics
    
    def get_leaderboard(self) -> pd.DataFrame:
        """Get model comparison leaderboard."""
        if not self.results:
            return pd.DataFrame()
        
        # Create DataFrame from results
        df = pd.DataFrame(self.results).T
        
        # Sort by NDCG@10 (or first available metric)
        sort_metric = None
        for k in self.k_values:
            if f'ndcg@{k}' in df.columns:
                sort_metric = f'ndcg@{k}'
                break
        
        if sort_metric:
            df = df.sort_values(sort_metric, ascending=False)
        
        return df
    
    def plot_comparison(self, metric: str = 'ndcg@10'):
        """Plot comparison of models for a specific metric."""
        import matplotlib.pyplot as plt
        
        if metric not in self.results:
            print(f"Metric {metric} not found in results")
            return
        
        models = list(self.results.keys())
        values = [self.results[model][metric] for model in models]
        
        plt.figure(figsize=(10, 6))
        plt.bar(models, values)
        plt.title(f'Model Comparison - {metric.upper()}')
        plt.xlabel('Models')
        plt.ylabel(metric.upper())
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


def compute_coverage_metrics(predictions: Dict[int, List[int]], 
                           all_items: List[int]) -> Dict[str, float]:
    """Compute coverage and diversity metrics."""
    
    # Item coverage
    recommended_items = set()
    for user_recs in predictions.values():
        recommended_items.update(user_recs)
    
    item_coverage = len(recommended_items) / len(all_items)
    
    # User coverage
    users_with_recs = len([user for user, recs in predictions.items() if len(recs) > 0])
    total_users = len(predictions)
    user_coverage = users_with_recs / total_users if total_users > 0 else 0.0
    
    # Diversity (intra-list diversity)
    diversity_scores = []
    for user_recs in predictions.values():
        if len(user_recs) > 1:
            # Simple diversity: number of unique items / total items
            diversity = len(set(user_recs)) / len(user_recs)
            diversity_scores.append(diversity)
    
    avg_diversity = np.mean(diversity_scores) if diversity_scores else 0.0
    
    return {
        'item_coverage': item_coverage,
        'user_coverage': user_coverage,
        'diversity': avg_diversity
    }


def compute_popularity_bias(predictions: Dict[int, List[int]], 
                          item_popularity: Dict[int, int]) -> Dict[str, float]:
    """Compute popularity bias metrics."""
    
    all_recs = []
    for user_recs in predictions.values():
        all_recs.extend(user_recs)
    
    if not all_recs:
        return {'popularity_bias': 0.0, 'gini_coefficient': 0.0}
    
    # Compute popularity of recommended items
    rec_popularities = [item_popularity.get(item, 0) for item in all_recs]
    
    # Popularity bias: average popularity of recommended items
    popularity_bias = np.mean(rec_popularities)
    
    # Gini coefficient for popularity distribution
    sorted_popularities = sorted(rec_popularities)
    n = len(sorted_popularities)
    cumsum = np.cumsum(sorted_popularities)
    gini_coefficient = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0.0
    
    return {
        'popularity_bias': popularity_bias,
        'gini_coefficient': gini_coefficient
    }
