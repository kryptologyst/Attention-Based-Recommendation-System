"""
Data loading and preprocessing utilities for the attention-based recommendation system.
"""

import os
import pickle
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import yaml


class RecommendationDataset(Dataset):
    """Dataset class for recommendation data."""
    
    def __init__(self, interactions: pd.DataFrame, users: Optional[pd.DataFrame] = None,
                 items: Optional[pd.DataFrame] = None, user_history: Optional[Dict] = None):
        """
        Initialize the dataset.
        
        Args:
            interactions: DataFrame with columns [user_id, item_id, rating, timestamp]
            users: Optional DataFrame with user features
            items: Optional DataFrame with item features
            user_history: Optional dict mapping user_id to list of item_ids
        """
        self.interactions = interactions
        self.users = users
        self.items = items
        self.user_history = user_history or {}
        
        # Create mappings
        self.user_to_idx = {user_id: idx for idx, user_id in enumerate(interactions['user_id'].unique())}
        self.item_to_idx = {item_id: idx for idx, item_id in enumerate(interactions['item_id'].unique())}
        self.idx_to_user = {idx: user_id for user_id, idx in self.user_to_idx.items()}
        self.idx_to_item = {idx: item_id for item_id, idx in self.item_to_idx.items()}
        
        self.num_users = len(self.user_to_idx)
        self.num_items = len(self.item_to_idx)
        
        # Convert to indices
        self.interactions['user_idx'] = self.interactions['user_id'].map(self.user_to_idx)
        self.interactions['item_idx'] = self.interactions['item_id'].map(self.item_to_idx)
        
    def __len__(self) -> int:
        return len(self.interactions)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single interaction."""
        row = self.interactions.iloc[idx]
        
        user_idx = torch.tensor(row['user_idx'], dtype=torch.long)
        item_idx = torch.tensor(row['item_idx'], dtype=torch.long)
        rating = torch.tensor(row['rating'], dtype=torch.float)
        
        # Get user history if available
        user_id = self.idx_to_user[row['user_idx']]
        history = self.user_history.get(user_id, [])
        
        if history:
            # Convert history to indices and pad/truncate
            hist_indices = [self.item_to_idx.get(item_id, 0) for item_id in history]
            hist_tensor = torch.tensor(hist_indices[:50], dtype=torch.long)  # Max 50 items
            if len(hist_indices) < 50:
                hist_tensor = torch.cat([hist_tensor, torch.zeros(50 - len(hist_indices), dtype=torch.long)])
        else:
            hist_tensor = torch.zeros(50, dtype=torch.long)
        
        return {
            'user_idx': user_idx,
            'item_idx': item_idx,
            'rating': rating,
            'user_history': hist_tensor
        }
    
    def get_user_history(self, user_id: int) -> List[int]:
        """Get user interaction history."""
        return self.user_history.get(user_id, [])
    
    def get_item_popularity(self) -> Dict[int, int]:
        """Get item popularity counts."""
        return self.interactions['item_idx'].value_counts().to_dict()


class DataProcessor:
    """Data processing utilities."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.min_user_interactions = config['data']['min_interactions_per_user']
        self.min_item_interactions = config['data']['min_interactions_per_item']
        self.test_size = config['data']['test_size']
        self.val_size = config['data']['val_size']
        self.random_state = config['data']['random_state']
    
    def load_data(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Load data from files."""
        interactions_file = self.config['data']['interactions_file']
        items_file = self.config['data']['items_file']
        users_file = self.config['data']['users_file']
        
        # Load interactions
        if os.path.exists(interactions_file):
            interactions = pd.read_csv(interactions_file)
        else:
            interactions = self._generate_synthetic_data()
        
        # Load items if available
        items = None
        if os.path.exists(items_file):
            items = pd.read_csv(items_file)
        
        # Load users if available
        users = None
        if os.path.exists(users_file):
            users = pd.read_csv(users_file)
        
        return interactions, items, users
    
    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic movie recommendation data."""
        np.random.seed(self.random_state)
        
        # Parameters
        num_users = 1000
        num_items = 500
        num_interactions = 10000
        
        # Generate user-item interactions with some patterns
        interactions = []
        
        # Create some user clusters with different preferences
        user_clusters = np.random.choice(5, num_users)
        item_clusters = np.random.choice(5, num_items)
        
        for _ in range(num_interactions):
            user_id = np.random.randint(0, num_users)
            item_id = np.random.randint(0, num_items)
            
            # Higher rating if user and item are in same cluster
            base_rating = 3.0
            if user_clusters[user_id] == item_clusters[item_id]:
                base_rating += 1.5
            
            # Add some noise
            rating = np.clip(base_rating + np.random.normal(0, 0.5), 1, 5)
            
            # Generate timestamp (last 2 years)
            timestamp = np.random.randint(1600000000, 1700000000)
            
            interactions.append({
                'user_id': f'user_{user_id}',
                'item_id': f'item_{item_id}',
                'rating': rating,
                'timestamp': timestamp
            })
        
        interactions_df = pd.DataFrame(interactions)
        
        # Save synthetic data
        os.makedirs(os.path.dirname(self.config['data']['interactions_file']), exist_ok=True)
        interactions_df.to_csv(self.config['data']['interactions_file'], index=False)
        
        # Generate synthetic items
        items_data = []
        genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 'Thriller']
        
        for i in range(num_items):
            items_data.append({
                'item_id': f'item_{i}',
                'title': f'Movie {i}',
                'genre': np.random.choice(genres),
                'year': np.random.randint(1990, 2024),
                'rating': np.random.uniform(2.0, 5.0)
            })
        
        items_df = pd.DataFrame(items_data)
        items_df.to_csv(self.config['data']['items_file'], index=False)
        
        # Generate synthetic users
        users_data = []
        age_groups = ['18-25', '26-35', '36-45', '46-55', '55+']
        
        for i in range(num_users):
            users_data.append({
                'user_id': f'user_{i}',
                'age_group': np.random.choice(age_groups),
                'gender': np.random.choice(['M', 'F']),
                'location': f'City_{np.random.randint(1, 20)}'
            })
        
        users_df = pd.DataFrame(users_data)
        users_df.to_csv(self.config['data']['users_file'], index=False)
        
        return interactions_df
    
    def preprocess_data(self, interactions: pd.DataFrame, items: Optional[pd.DataFrame] = None,
                       users: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Dict]:
        """Preprocess the data."""
        print(f"Original interactions: {len(interactions)}")
        
        # Filter users with minimum interactions
        user_counts = interactions['user_id'].value_counts()
        valid_users = user_counts[user_counts >= self.min_user_interactions].index
        interactions = interactions[interactions['user_id'].isin(valid_users)]
        
        # Filter items with minimum interactions
        item_counts = interactions['item_id'].value_counts()
        valid_items = item_counts[item_counts >= self.min_item_interactions].index
        interactions = interactions[interactions['item_id'].isin(valid_items)]
        
        print(f"After filtering: {len(interactions)} interactions")
        print(f"Users: {interactions['user_id'].nunique()}")
        print(f"Items: {interactions['item_id'].nunique()}")
        
        # Create user history
        user_history = {}
        for user_id in interactions['user_id'].unique():
            user_interactions = interactions[interactions['user_id'] == user_id]
            # Sort by timestamp and get item sequence
            user_items = user_interactions.sort_values('timestamp')['item_id'].tolist()
            user_history[user_id] = user_items
        
        return interactions, user_history
    
    def split_data(self, interactions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train/validation/test sets."""
        # Sort by timestamp for temporal split
        interactions = interactions.sort_values('timestamp')
        
        # Split by user to ensure each user appears in only one set
        users = interactions['user_id'].unique()
        
        # First split: train vs (val + test)
        train_users, temp_users = train_test_split(
            users, test_size=self.val_size + self.test_size, random_state=self.random_state
        )
        
        # Second split: val vs test
        val_users, test_users = train_test_split(
            temp_users, test_size=self.test_size / (self.val_size + self.test_size),
            random_state=self.random_state
        )
        
        # Create splits
        train_data = interactions[interactions['user_id'].isin(train_users)]
        val_data = interactions[interactions['user_id'].isin(val_users)]
        test_data = interactions[interactions['user_id'].isin(test_users)]
        
        print(f"Train: {len(train_data)} interactions")
        print(f"Validation: {len(val_data)} interactions")
        print(f"Test: {len(test_data)} interactions")
        
        return train_data, val_data, test_data


def create_data_loaders(train_data: pd.DataFrame, val_data: pd.DataFrame, test_data: pd.DataFrame,
                       user_history: Dict, batch_size: int = 256) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for training."""
    
    train_dataset = RecommendationDataset(train_data, user_history=user_history)
    val_dataset = RecommendationDataset(val_data, user_history=user_history)
    test_dataset = RecommendationDataset(test_data, user_history=user_history)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
