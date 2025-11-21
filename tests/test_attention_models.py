"""
Unit tests for attention-based recommendation models.
"""

import pytest
import torch
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

# Import modules to test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.attention_models import (
    SelfAttention, MultiHeadAttention, TransformerAttention, AttentionRecommender
)
from data.data_loader import RecommendationDataset, DataProcessor
from evaluation.metrics import RecommendationMetrics, ModelEvaluator
from utils.trainer import Trainer, set_seed


class TestAttentionMechanisms:
    """Test attention mechanism implementations."""
    
    def test_self_attention(self):
        """Test self-attention mechanism."""
        batch_size, seq_len, input_dim = 2, 5, 8
        attention = SelfAttention(input_dim)
        
        # Create test tensors
        query = torch.randn(batch_size, seq_len, input_dim)
        key = torch.randn(batch_size, seq_len, input_dim)
        value = torch.randn(batch_size, seq_len, input_dim)
        
        # Forward pass
        output, attention_weights = attention(query, key, value)
        
        # Check output shape
        assert output.shape == (batch_size, seq_len, input_dim)
        assert attention_weights.shape == (batch_size, seq_len, seq_len)
        
        # Check attention weights sum to 1
        assert torch.allclose(attention_weights.sum(dim=-1), torch.ones(batch_size, seq_len), atol=1e-6)
    
    def test_multi_head_attention(self):
        """Test multi-head attention mechanism."""
        batch_size, seq_len, input_dim = 2, 5, 8
        num_heads = 4
        attention = MultiHeadAttention(input_dim, num_heads)
        
        # Create test tensors
        query = torch.randn(batch_size, seq_len, input_dim)
        key = torch.randn(batch_size, seq_len, input_dim)
        value = torch.randn(batch_size, seq_len, input_dim)
        
        # Forward pass
        output, attention_weights = attention(query, key, value)
        
        # Check output shape
        assert output.shape == (batch_size, seq_len, input_dim)
        assert attention_weights.shape == (batch_size, seq_len, seq_len)
        
        # Check attention weights sum to 1
        assert torch.allclose(attention_weights.sum(dim=-1), torch.ones(batch_size, seq_len), atol=1e-6)
    
    def test_transformer_attention(self):
        """Test transformer attention mechanism."""
        batch_size, seq_len, input_dim = 2, 5, 8
        num_heads = 4
        num_layers = 2
        attention = TransformerAttention(input_dim, num_heads, num_layers)
        
        # Create test tensors
        query = torch.randn(batch_size, seq_len, input_dim)
        key = torch.randn(batch_size, seq_len, input_dim)
        value = torch.randn(batch_size, seq_len, input_dim)
        
        # Forward pass
        output, attention_weights = attention(query, key, value)
        
        # Check output shape
        assert output.shape == (batch_size, seq_len, input_dim)
        assert attention_weights.shape == (batch_size, seq_len, seq_len)


class TestAttentionRecommender:
    """Test the main recommendation model."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        num_users, num_items = 100, 50
        embedding_dim = 32
        
        model = AttentionRecommender(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
            attention_type="self_attention"
        )
        
        assert model.num_users == num_users
        assert model.num_items == num_items
        assert model.embedding_dim == embedding_dim
    
    def test_model_forward(self):
        """Test model forward pass."""
        num_users, num_items = 100, 50
        embedding_dim = 32
        batch_size = 4
        
        model = AttentionRecommender(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
            attention_type="self_attention"
        )
        
        # Create test inputs
        user_ids = torch.randint(0, num_users, (batch_size,))
        item_ids = torch.randint(0, num_items, (batch_size,))
        user_history = torch.randint(0, num_items, (batch_size, 10))
        
        # Forward pass
        predictions, attention_weights = model(user_ids, item_ids, user_history)
        
        # Check output shapes
        assert predictions.shape == (batch_size,)
        assert attention_weights.shape == (batch_size, 1, 1)
    
    def test_different_attention_types(self):
        """Test different attention types."""
        num_users, num_items = 100, 50
        embedding_dim = 32
        
        attention_types = ["self_attention", "multi_head", "transformer"]
        
        for attention_type in attention_types:
            model = AttentionRecommender(
                num_users=num_users,
                num_items=num_items,
                embedding_dim=embedding_dim,
                attention_type=attention_type
            )
            
            # Test forward pass
            user_ids = torch.randint(0, num_users, (2,))
            item_ids = torch.randint(0, num_items, (2,))
            user_history = torch.randint(0, num_items, (2, 10))
            
            predictions, attention_weights = model(user_ids, item_ids, user_history)
            assert predictions.shape == (2,)


class TestRecommendationMetrics:
    """Test recommendation metrics."""
    
    def test_precision_at_k(self):
        """Test precision@k calculation."""
        metrics = RecommendationMetrics([5, 10])
        
        y_true = [1, 2, 3]
        y_pred = [1, 4, 5, 2, 6, 7, 8, 9, 10, 11]
        
        precision_5 = metrics.precision_at_k(y_true, y_pred, 5)
        precision_10 = metrics.precision_at_k(y_true, y_pred, 10)
        
        # Items 1 and 2 are in top-5, so precision@5 = 2/5 = 0.4
        assert abs(precision_5 - 0.4) < 1e-6
        
        # Items 1 and 2 are in top-10, so precision@10 = 2/10 = 0.2
        assert abs(precision_10 - 0.2) < 1e-6
    
    def test_recall_at_k(self):
        """Test recall@k calculation."""
        metrics = RecommendationMetrics([5, 10])
        
        y_true = [1, 2, 3]
        y_pred = [1, 4, 5, 2, 6, 7, 8, 9, 10, 11]
        
        recall_5 = metrics.recall_at_k(y_true, y_pred, 5)
        recall_10 = metrics.recall_at_k(y_true, y_pred, 10)
        
        # Items 1 and 2 are in top-5, so recall@5 = 2/3 ≈ 0.667
        assert abs(recall_5 - 2/3) < 1e-6
        
        # Items 1 and 2 are in top-10, so recall@10 = 2/3 ≈ 0.667
        assert abs(recall_10 - 2/3) < 1e-6
    
    def test_ndcg_at_k(self):
        """Test NDCG@k calculation."""
        metrics = RecommendationMetrics([5])
        
        y_true = [1, 2, 3]
        y_pred = [1, 4, 5, 2, 6]
        
        ndcg_5 = metrics.ndcg_at_k(y_true, y_pred, 5)
        
        # Should be positive since we have relevant items
        assert ndcg_5 > 0
        assert ndcg_5 <= 1.0
    
    def test_compute_all_metrics(self):
        """Test computing all metrics."""
        metrics = RecommendationMetrics([5])
        
        y_true = [1, 2, 3]
        y_pred = [1, 4, 5, 2, 6]
        
        all_metrics = metrics.compute_all_metrics(y_true, y_pred)
        
        expected_keys = ['precision@5', 'recall@5', 'ndcg@5', 'map@5', 'hit_rate@5']
        for key in expected_keys:
            assert key in all_metrics
            assert 0 <= all_metrics[key] <= 1


class TestDataLoader:
    """Test data loading utilities."""
    
    def test_recommendation_dataset(self):
        """Test RecommendationDataset class."""
        # Create sample data
        interactions = pd.DataFrame({
            'user_id': ['user1', 'user2', 'user1', 'user3'],
            'item_id': ['item1', 'item1', 'item2', 'item1'],
            'rating': [4.5, 3.0, 5.0, 2.5],
            'timestamp': [1600000000, 1600000100, 1600000200, 1600000300]
        })
        
        user_history = {
            'user1': ['item1', 'item2'],
            'user2': ['item1'],
            'user3': ['item1']
        }
        
        dataset = RecommendationDataset(interactions, user_history=user_history)
        
        # Test dataset properties
        assert len(dataset) == 4
        assert dataset.num_users == 3
        assert dataset.num_items == 2
        
        # Test getting an item
        item = dataset[0]
        assert 'user_idx' in item
        assert 'item_idx' in item
        assert 'rating' in item
        assert 'user_history' in item
        
        # Test user history
        history = dataset.get_user_history('user1')
        assert len(history) == 2
    
    def test_data_processor(self):
        """Test DataProcessor class."""
        # Mock configuration
        config = {
            'data': {
                'min_interactions_per_user': 1,
                'min_interactions_per_item': 1,
                'test_size': 0.2,
                'val_size': 0.1,
                'random_state': 42,
                'interactions_file': 'test_interactions.csv',
                'items_file': 'test_items.csv',
                'users_file': 'test_users.csv'
            }
        }
        
        processor = DataProcessor(config)
        
        # Test synthetic data generation
        interactions = processor._generate_synthetic_data()
        
        assert len(interactions) > 0
        assert 'user_id' in interactions.columns
        assert 'item_id' in interactions.columns
        assert 'rating' in interactions.columns
        assert 'timestamp' in interactions.columns


class TestTrainer:
    """Test training utilities."""
    
    def test_set_seed(self):
        """Test random seed setting."""
        set_seed(42)
        
        # Generate some random numbers
        np_rand1 = np.random.random()
        torch_rand1 = torch.rand(1).item()
        
        # Reset seed and generate again
        set_seed(42)
        np_rand2 = np.random.random()
        torch_rand2 = torch.rand(1).item()
        
        # Should be the same
        assert abs(np_rand1 - np_rand2) < 1e-6
        assert abs(torch_rand1 - torch_rand2) < 1e-6


if __name__ == '__main__':
    pytest.main([__file__])
