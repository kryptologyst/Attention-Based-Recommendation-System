# Project 338. Attention mechanisms for recommendations
# 
# This is the original simple implementation that has been refactored into a modern,
# production-ready recommendation system. See the updated project structure:
#
# - src/models/attention_models.py: Modern attention mechanisms (Self-Attention, Multi-Head, Transformer)
# - src/data/data_loader.py: Data processing and loading utilities
# - src/evaluation/metrics.py: Comprehensive evaluation metrics
# - src/utils/trainer.py: Training utilities and model management
# - scripts/train.py: Main training script
# - demo.py: Interactive Streamlit demo
# - example.py: Simple example script
#
# To run the modernized version:
# 1. python example.py                    # Simple example
# 2. python scripts/train.py              # Train all models
# 3. streamlit run demo.py               # Interactive demo
#
# Original simple implementation below for reference:

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Simulate user-item ratings matrix
users = ['User1', 'User2', 'User3', 'User4', 'User5']
items = ['Item1', 'Item2', 'Item3', 'Item4', 'Item5']
ratings = np.array([
    [5, 4, 0, 0, 3],
    [4, 0, 0, 2, 1],
    [1, 1, 0, 5, 4],
    [0, 0, 5, 4, 4],
    [2, 3, 0, 1, 0]
])

df = pd.DataFrame(ratings, index=users, columns=items)

# 2. Simple Attention Mechanism
class SimpleAttention(nn.Module):
    def __init__(self, input_size):
        super(SimpleAttention, self).__init__()
        self.attn_weights = nn.Parameter(torch.randn(input_size))

    def forward(self, x):
        attn_scores = torch.matmul(x, self.attn_weights)
        attn_weights = torch.softmax(attn_scores, dim=0)
        weighted_input = x * attn_weights
        return weighted_input.sum(dim=0)

# 3. Simple Recommendation Model with Attention
class SimpleRecommendationModel(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=5):
        super(SimpleRecommendationModel, self).__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.attention = SimpleAttention(embedding_dim)

    def forward(self, user_idx, item_idx):
        user_emb = self.user_embedding(user_idx)
        item_emb = self.item_embedding(item_idx)
        combined_emb = torch.stack([user_emb, item_emb], dim=0)
        attention_output = self.attention(combined_emb)
        return attention_output

# 4. Train the simple model
if __name__ == "__main__":
    print("Original Simple Attention Implementation")
    print("For the modernized version, run: python example.py")
    print("=" * 50)
    
    n_users, n_items = df.shape
    model = SimpleRecommendationModel(n_users, n_items)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train for a few epochs
    num_epochs = 5
    for epoch in range(num_epochs):
        epoch_loss = 0
        for user_idx in range(len(users)):
            for item_idx in range(len(items)):
                if df.iloc[user_idx, item_idx] > 0:
                    optimizer.zero_grad()
                    user_tensor = torch.tensor([user_idx])
                    item_tensor = torch.tensor([item_idx])
                    predicted_rating = model(user_tensor, item_tensor)
                    actual_rating = torch.tensor(df.iloc[user_idx, item_idx], dtype=torch.float)
                    loss = loss_fn(predicted_rating, actual_rating)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(users)/len(items):.4f}")

    # Make predictions for User1
    user_idx = 0
    predicted_ratings = []
    for item_idx in range(len(items)):
        user_tensor = torch.tensor([user_idx])
        item_tensor = torch.tensor([item_idx])
        predicted_ratings.append(model(user_tensor, item_tensor).item())

    print(f"\nPredicted Ratings for User1:")
    for item, pred in zip(items, predicted_ratings):
        print(f"{item}: Predicted Rating = {pred:.2f}")
    
    print("\nThis demonstrates the basic concept.")
    print("For production-ready implementation with multiple attention mechanisms,")
    print("comprehensive evaluation, and interactive demo, see the modernized version!")