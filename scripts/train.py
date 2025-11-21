"""
Main training script for attention-based recommendation system.
"""

import os
import sys
import argparse
from typing import Dict, List
import yaml
import torch
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.data_loader import DataProcessor, create_data_loaders, load_config
from models.attention_models import AttentionRecommender
from utils.trainer import Trainer, ModelManager, set_seed


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train attention-based recommendation models')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--attention-types', nargs='+', 
                       default=['self_attention', 'multi_head_attention', 'transformer_attention'],
                       help='Types of attention mechanisms to train')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use for training (auto, cpu, cuda)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    print(f"Loaded configuration from {args.config}")
    
    # Update config with command line arguments
    config['attention_types'] = args.attention_types
    config['training']['device'] = args.device
    
    # Process data
    print("Processing data...")
    processor = DataProcessor(config)
    interactions, items, users = processor.load_data()
    interactions, user_history = processor.preprocess_data(interactions, items, users)
    
    # Update config with actual data dimensions
    config['data']['num_users'] = interactions['user_id'].nunique()
    config['data']['num_items'] = interactions['item_id'].nunique()
    
    print(f"Dataset statistics:")
    print(f"  Users: {config['data']['num_users']}")
    print(f"  Items: {config['data']['num_items']}")
    print(f"  Interactions: {len(interactions)}")
    
    # Split data
    train_data, val_data, test_data = processor.split_data(interactions)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data, val_data, test_data, user_history, 
        config['training']['batch_size']
    )
    
    # Train multiple models
    print(f"\nTraining {len(args.attention_types)} attention models...")
    model_manager = ModelManager(config)
    
    results = model_manager.train_multiple_models(
        train_loader, val_loader, test_loader, user_history, args.attention_types
    )
    
    # Compare models
    leaderboard = model_manager.compare_models()
    
    # Save results
    model_manager.save_results('results/model_comparison.pkl')
    
    # Save leaderboard
    os.makedirs('results', exist_ok=True)
    leaderboard.to_csv('results/leaderboard.csv')
    print(f"\nLeaderboard saved to results/leaderboard.csv")
    
    print("\nTraining completed successfully!")


if __name__ == '__main__':
    main()
