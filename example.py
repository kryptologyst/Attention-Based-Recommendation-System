"""
Simple example script demonstrating attention-based recommendations.
This is a simplified version of the original 0338.py file.
"""

import sys
import os
sys.path.append('src')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from src.data.data_loader import DataProcessor, load_config
from src.models.attention_models import AttentionRecommender
from src.utils.trainer import set_seed


def main():
    """Simple example of attention-based recommendation system."""
    print("Attention-Based Recommendation System - Simple Example")
    print("=" * 60)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Load configuration
    config = load_config('configs/config.yaml')
    
    # Process data
    print("Loading and processing data...")
    processor = DataProcessor(config)
    interactions, items, users = processor.load_data()
    interactions, user_history = processor.preprocess_data(interactions, items, users)
    
    # Update config with actual data dimensions
    config['data']['num_users'] = interactions['user_id'].nunique()
    config['data']['num_items'] = interactions['item_id'].nunique()
    
    print(f"Dataset: {config['data']['num_users']} users, {config['data']['num_items']} items")
    print(f"Interactions: {len(interactions)}")
    
    # Split data
    train_data, val_data, test_data = processor.split_data(interactions)
    
    # Create a simple model
    print("\nCreating attention-based recommendation model...")
    model = AttentionRecommender(
        num_users=config['data']['num_users'],
        num_items=config['data']['num_items'],
        embedding_dim=32,  # Smaller for quick demo
        attention_type="self_attention",
        num_heads=4,
        num_layers=1,
        dropout=0.1
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Simple training loop (simplified)
    print("\nTraining model...")
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Convert data to tensors
    train_users = torch.tensor([processor.user_to_idx.get(uid, 0) for uid in train_data['user_id']])
    train_items = torch.tensor([processor.item_to_idx.get(iid, 0) for iid in train_data['item_id']])
    train_ratings = torch.tensor(train_data['rating'].values, dtype=torch.float)
    
    # Train for a few epochs
    num_epochs = 5
    batch_size = 64
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        for i in range(0, len(train_users), batch_size):
            batch_users = train_users[i:i+batch_size]
            batch_items = train_items[i:i+batch_size]
            batch_ratings = train_ratings[i:i+batch_size]
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions, attention_weights = model(batch_users, batch_items)
            loss = criterion(predictions, batch_ratings)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # Get recommendations for a user
    print("\nGenerating recommendations...")
    model.eval()
    
    # Select first user
    user_id = interactions['user_id'].iloc[0]
    user_idx = processor.user_to_idx[user_id]
    
    with torch.no_grad():
        # Get recommendations for this user
        all_items = torch.arange(config['data']['num_items'])
        user_tensor = torch.tensor([user_idx] * config['data']['num_items'])
        
        predictions, attention_weights = model(user_tensor, all_items)
        
        # Get top 5 recommendations
        top_items = torch.topk(predictions, 5).indices
    
    print(f"\nTop 5 recommendations for user {user_id}:")
    for i, item_idx in enumerate(top_items):
        item_id = processor.idx_to_item[item_idx.item()]
        score = predictions[item_idx].item()
        print(f"{i+1}. {item_id} (score: {score:.3f})")
    
    # Show user's history
    user_interactions = interactions[interactions['user_id'] == user_id]
    print(f"\nUser's interaction history:")
    for _, row in user_interactions.head(5).iterrows():
        print(f"  {row['item_id']}: {row['rating']}")
    
    print("\nExample completed successfully!")
    print("\nTo explore more features, run:")
    print("  python scripts/train.py  # Train all attention models")
    print("  streamlit run demo.py    # Launch interactive demo")


if __name__ == '__main__':
    main()
