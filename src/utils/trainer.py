"""
Training utilities for attention-based recommendation models.
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import pickle

from src.models.attention_models import AttentionRecommender
from src.data.data_loader import RecommendationDataset
from src.evaluation.metrics import ModelEvaluator, ModelComparison


class Trainer:
    """Trainer class for attention-based recommendation models."""
    
    def __init__(self, model: nn.Module, config: Dict, device: str = 'auto'):
        self.model = model
        self.config = config
        self.device = self._setup_device(device)
        self.model.to(self.device)
        
        # Training parameters
        self.batch_size = config['training']['batch_size']
        self.learning_rate = config['training']['learning_rate']
        self.num_epochs = config['training']['num_epochs']
        self.weight_decay = config['training']['weight_decay']
        self.gradient_clip_norm = config['training']['gradient_clip_norm']
        self.early_stopping_patience = config['training']['early_stopping_patience']
        
        # Initialize optimizer and loss
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        self.criterion = nn.MSELoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # Model comparison
        self.model_comparison = ModelComparison(config['evaluation']['k_values'])
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup device for training."""
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            else:
                return torch.device('cpu')
        else:
            return torch.device(device)
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            user_ids = batch['user_idx'].to(self.device)
            item_ids = batch['item_idx'].to(self.device)
            ratings = batch['rating'].to(self.device)
            user_history = batch['user_history'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions, attention_weights = self.model(user_ids, item_ids, user_history)
            
            # Compute loss
            loss = self.criterion(predictions, ratings)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / num_batches
    
    def validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                user_ids = batch['user_idx'].to(self.device)
                item_ids = batch['item_idx'].to(self.device)
                ratings = batch['rating'].to(self.device)
                user_history = batch['user_history'].to(self.device)
                
                # Forward pass
                predictions, attention_weights = self.model(user_ids, item_ids, user_history)
                
                # Compute loss
                loss = self.criterion(predictions, ratings)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              test_loader: Optional[DataLoader] = None, user_history: Optional[Dict] = None) -> Dict:
        """Train the model."""
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            # Training
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss = self.validate_epoch(val_loader)
            self.val_losses.append(val_loss)
            
            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                # Save best model
                self.save_checkpoint(epoch, is_best=True)
            else:
                self.epochs_without_improvement += 1
            
            # Print progress
            print(f"Epoch {epoch+1}/{self.num_epochs}: "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Load best model for evaluation
        self.load_checkpoint(is_best=True)
        
        # Evaluate on test set if provided
        results = {}
        if test_loader is not None and user_history is not None:
            evaluator = ModelEvaluator(self.model, self.device, self.config['evaluation']['k_values'])
            test_metrics = evaluator.evaluate_model(test_loader, user_history, 
                                                  self.config['evaluation']['num_negatives'])
            results['test_metrics'] = test_metrics
            
            print("\nTest Results:")
            for metric, value in test_metrics.items():
                print(f"{metric}: {value:.4f}")
        
        results['training_time'] = training_time
        results['best_epoch'] = len(self.train_losses) - self.epochs_without_improvement
        results['train_losses'] = self.train_losses
        results['val_losses'] = self.val_losses
        
        return results
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Create models directory if it doesn't exist
        os.makedirs('models/checkpoints', exist_ok=True)
        
        # Save regular checkpoint
        checkpoint_path = f'models/checkpoints/checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = 'models/checkpoints/best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"Best model saved at epoch {epoch}")
    
    def load_checkpoint(self, checkpoint_path: Optional[str] = None, is_best: bool = False):
        """Load model checkpoint."""
        if checkpoint_path is None:
            if is_best:
                checkpoint_path = 'models/checkpoints/best_model.pth'
            else:
                # Load latest checkpoint
                checkpoint_files = [f for f in os.listdir('models/checkpoints') 
                                  if f.startswith('checkpoint_epoch_')]
                if not checkpoint_files:
                    print("No checkpoint found")
                    return
                latest_file = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
                checkpoint_path = f'models/checkpoints/{latest_file}'
        
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Checkpoint loaded from epoch {checkpoint['epoch']}")
    
    def plot_training_history(self):
        """Plot training history."""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History (Log Scale)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()


class ModelManager:
    """Manager for training and comparing multiple models."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.results = {}
        self.models = {}
    
    def train_multiple_models(self, train_loader: DataLoader, val_loader: DataLoader,
                            test_loader: DataLoader, user_history: Dict,
                            attention_types: List[str] = None) -> Dict:
        """Train multiple models with different attention mechanisms."""
        
        if attention_types is None:
            attention_types = self.config['attention_types']
        
        print(f"Training {len(attention_types)} different attention models...")
        
        for attention_type in attention_types:
            print(f"\n{'='*50}")
            print(f"Training {attention_type} model")
            print(f"{'='*50}")
            
            # Create model
            model = AttentionRecommender(
                num_users=self.config['data']['num_users'],
                num_items=self.config['data']['num_items'],
                embedding_dim=self.config['model']['embedding_dim'],
                attention_type=attention_type,
                num_heads=self.config['model']['num_heads'],
                num_layers=self.config['model']['num_layers'],
                dropout=self.config['model']['dropout']
            )
            
            # Create trainer
            trainer = Trainer(model, self.config)
            
            # Train model
            results = trainer.train(train_loader, val_loader, test_loader, user_history)
            
            # Store results
            self.results[attention_type] = results
            self.models[attention_type] = model
            
            print(f"\n{attention_type} training completed!")
            print(f"Best validation loss: {trainer.best_val_loss:.4f}")
            print(f"Training time: {results['training_time']:.2f} seconds")
        
        return self.results
    
    def compare_models(self) -> pd.DataFrame:
        """Compare all trained models."""
        comparison = ModelComparison(self.config['evaluation']['k_values'])
        
        for model_name, results in self.results.items():
            if 'test_metrics' in results:
                comparison.add_model_results(model_name, results['test_metrics'])
        
        leaderboard = comparison.get_leaderboard()
        
        print("\nModel Comparison Leaderboard:")
        print("="*80)
        print(leaderboard.round(4))
        
        return leaderboard
    
    def save_results(self, filepath: str = 'results/model_comparison.pkl'):
        """Save all results."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'results': self.results,
                'config': self.config
            }, f)
        
        print(f"Results saved to {filepath}")
    
    def load_results(self, filepath: str = 'results/model_comparison.pkl'):
        """Load results from file."""
        if not os.path.exists(filepath):
            print(f"Results file not found: {filepath}")
            return
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.results = data['results']
        self.config = data['config']
        
        print(f"Results loaded from {filepath}")


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
