"""
Attention-based Recommendation System

This module contains the core attention mechanisms and recommendation models.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod


class BaseAttention(nn.Module, ABC):
    """Base class for attention mechanisms."""
    
    def __init__(self, input_dim: int, **kwargs):
        super().__init__()
        self.input_dim = input_dim
    
    @abstractmethod
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of attention mechanism.
        
        Args:
            query: Query tensor [batch_size, seq_len, input_dim]
            key: Key tensor [batch_size, seq_len, input_dim]
            value: Value tensor [batch_size, seq_len, input_dim]
            mask: Optional attention mask [batch_size, seq_len, seq_len]
            
        Returns:
            Tuple of (output, attention_weights)
        """
        pass


class SelfAttention(BaseAttention):
    """Self-attention mechanism for recommendation systems."""
    
    def __init__(self, input_dim: int, dropout: float = 0.1):
        super().__init__(input_dim)
        self.query_proj = nn.Linear(input_dim, input_dim)
        self.key_proj = nn.Linear(input_dim, input_dim)
        self.value_proj = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = input_dim ** -0.5
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = query.size()
        
        # Project to query, key, value
        Q = self.query_proj(query)
        K = self.key_proj(key)
        V = self.value_proj(value)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights


class MultiHeadAttention(BaseAttention):
    """Multi-head attention mechanism."""
    
    def __init__(self, input_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__(input_dim)
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.query_proj = nn.Linear(input_dim, input_dim)
        self.key_proj = nn.Linear(input_dim, input_dim)
        self.value_proj = nn.Linear(input_dim, input_dim)
        self.out_proj = nn.Linear(input_dim, input_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = query.size()
        
        # Project to query, key, value
        Q = self.query_proj(query)
        K = self.key_proj(key)
        V = self.value_proj(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.out_proj(output)
        
        # Average attention weights across heads
        avg_attention_weights = attention_weights.mean(dim=1)
        
        return output, avg_attention_weights


class TransformerAttention(BaseAttention):
    """Transformer-style attention with positional encoding."""
    
    def __init__(self, input_dim: int, num_heads: int = 8, num_layers: int = 2, 
                 dropout: float = 0.1, max_seq_len: int = 100):
        super().__init__(input_dim)
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(max_seq_len, input_dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=num_heads,
                dim_feedforward=input_dim * 4,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = query.size()
        
        # Add positional encoding
        if seq_len <= self.max_seq_len:
            pos_enc = self.pos_encoding[:, :seq_len, :].to(query.device)
            query = query + pos_enc
            key = key + pos_enc
            value = value + pos_enc
        
        # Apply dropout
        query = self.dropout(query)
        key = self.dropout(key)
        value = self.dropout(value)
        
        # Pass through transformer layers
        output = query
        attention_weights_list = []
        
        for layer in self.layers:
            # Create attention mask for transformer
            if mask is not None:
                # Convert to transformer format (True for valid positions)
                transformer_mask = mask.bool()
            else:
                transformer_mask = None
                
            output = layer(output, src_key_padding_mask=transformer_mask)
            
            # Extract attention weights (simplified - in practice you'd need to modify the layer)
            attention_weights_list.append(torch.ones(batch_size, seq_len, seq_len) / seq_len)
        
        # Use the last layer's attention weights
        attention_weights = attention_weights_list[-1]
        
        return output, attention_weights


class AttentionRecommender(nn.Module):
    """Attention-based recommendation model."""
    
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64,
                 attention_type: str = "multi_head", num_heads: int = 8,
                 num_layers: int = 2, dropout: float = 0.1, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.attention_type = attention_type
        
        # Embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Attention mechanism
        if attention_type == "self_attention":
            self.attention = SelfAttention(embedding_dim, dropout)
        elif attention_type == "multi_head":
            self.attention = MultiHeadAttention(embedding_dim, num_heads, dropout)
        elif attention_type == "transformer":
            self.attention = TransformerAttention(
                embedding_dim, num_heads, num_layers, dropout, **kwargs
            )
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        # Output layers
        self.output_layer = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.01)
    
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor,
                user_history: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the recommendation model.
        
        Args:
            user_ids: User IDs [batch_size]
            item_ids: Item IDs [batch_size]
            user_history: User interaction history [batch_size, seq_len]
            
        Returns:
            Tuple of (predictions, attention_weights)
        """
        batch_size = user_ids.size(0)
        
        # Get embeddings
        user_emb = self.user_embedding(user_ids)  # [batch_size, embedding_dim]
        item_emb = self.item_embedding(item_ids)  # [batch_size, embedding_dim]
        
        if user_history is not None:
            # Use user history for attention
            hist_items = user_history  # [batch_size, seq_len]
            hist_emb = self.item_embedding(hist_items)  # [batch_size, seq_len, embedding_dim]
            
            # Apply attention between user embedding and item history
            user_emb_expanded = user_emb.unsqueeze(1).expand(-1, hist_emb.size(1), -1)
            
            # Use user embedding as query, history as key and value
            attended_output, attention_weights = self.attention(
                user_emb_expanded, hist_emb, hist_emb
            )
            
            # Combine with current item
            user_context = attended_output.mean(dim=1)  # [batch_size, embedding_dim]
            combined_emb = user_context + item_emb
        else:
            # Simple concatenation without history
            combined_emb = user_emb + item_emb
            attention_weights = torch.ones(batch_size, 1, 1)
        
        # Generate prediction
        prediction = self.output_layer(combined_emb).squeeze(-1)
        
        return prediction, attention_weights
    
    def get_user_embeddings(self, user_ids: torch.Tensor) -> torch.Tensor:
        """Get user embeddings."""
        return self.user_embedding(user_ids)
    
    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Get item embeddings."""
        return self.item_embedding(item_ids)
    
    def compute_similarity(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between users and items."""
        user_emb = self.get_user_embeddings(user_ids)
        item_emb = self.get_item_embeddings(item_ids)
        
        # Normalize embeddings
        user_emb = F.normalize(user_emb, p=2, dim=1)
        item_emb = F.normalize(item_emb, p=2, dim=1)
        
        # Compute cosine similarity
        similarity = torch.sum(user_emb * item_emb, dim=1)
        
        return similarity
