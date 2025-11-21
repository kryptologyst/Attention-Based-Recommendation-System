"""
Streamlit demo for attention-based recommendation system.
"""

import os
import sys
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.data_loader import DataProcessor, load_config
from models.attention_models import AttentionRecommender
from utils.trainer import set_seed


@st.cache_data
def load_data_and_models():
    """Load data and trained models."""
    # Load configuration
    config = load_config('configs/config.yaml')
    
    # Load data
    processor = DataProcessor(config)
    interactions, items, users = processor.load_data()
    interactions, user_history = processor.preprocess_data(interactions, items, users)
    
    # Load results
    results_file = 'results/model_comparison.pkl'
    if os.path.exists(results_file):
        with open(results_file, 'rb') as f:
            data = pickle.load(f)
        results = data['results']
    else:
        results = {}
    
    # Load models
    models = {}
    for model_name in ['self_attention', 'multi_head_attention', 'transformer_attention']:
        model_path = f'models/checkpoints/{model_name}_best.pth'
        if os.path.exists(model_path):
            model = AttentionRecommender(
                num_users=config['data']['num_users'],
                num_items=config['data']['num_items'],
                embedding_dim=config['model']['embedding_dim'],
                attention_type=model_name,
                num_heads=config['model']['num_heads'],
                num_layers=config['model']['num_layers'],
                dropout=config['model']['dropout']
            )
            
            checkpoint = torch.load(model_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            models[model_name] = model
    
    return config, interactions, items, users, user_history, results, models


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Attention-Based Recommendation System",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 Attention-Based Recommendation System")
    st.markdown("Explore how different attention mechanisms improve recommendation quality")
    
    # Load data and models
    with st.spinner("Loading data and models..."):
        config, interactions, items, users, user_history, results, models = load_data_and_models()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Overview", "Model Comparison", "User Recommendations", "Item Similarity", "Attention Analysis"]
    )
    
    if page == "Overview":
        show_overview(interactions, items, users, results)
    elif page == "Model Comparison":
        show_model_comparison(results)
    elif page == "User Recommendations":
        show_user_recommendations(models, interactions, items, user_history)
    elif page == "Item Similarity":
        show_item_similarity(models, interactions, items)
    elif page == "Attention Analysis":
        show_attention_analysis(models, interactions, items, user_history)


def show_overview(interactions, items, users, results):
    """Show dataset overview."""
    st.header("Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Users", f"{interactions['user_id'].nunique():,}")
    
    with col2:
        st.metric("Total Items", f"{interactions['item_id'].nunique():,}")
    
    with col3:
        st.metric("Total Interactions", f"{len(interactions):,}")
    
    with col4:
        avg_rating = interactions['rating'].mean()
        st.metric("Average Rating", f"{avg_rating:.2f}")
    
    # Rating distribution
    st.subheader("Rating Distribution")
    rating_counts = interactions['rating'].value_counts().sort_index()
    
    fig = px.bar(
        x=rating_counts.index,
        y=rating_counts.values,
        title="Distribution of Ratings",
        labels={'x': 'Rating', 'y': 'Count'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # User activity
    st.subheader("User Activity")
    user_interactions = interactions['user_id'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            user_interactions,
            title="Distribution of User Interactions",
            labels={'value': 'Number of Interactions', 'count': 'Number of Users'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(
            y=user_interactions.values,
            title="User Interaction Statistics"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Item popularity
    st.subheader("Item Popularity")
    item_interactions = interactions['item_id'].value_counts()
    
    fig = px.histogram(
        item_interactions,
        title="Distribution of Item Interactions",
        labels={'value': 'Number of Interactions', 'count': 'Number of Items'}
    )
    st.plotly_chart(fig, use_container_width=True)


def show_model_comparison(results):
    """Show model comparison results."""
    st.header("Model Comparison")
    
    if not results:
        st.warning("No model results found. Please train models first.")
        return
    
    # Create comparison dataframe
    comparison_data = []
    for model_name, model_results in results.items():
        if 'test_metrics' in model_results:
            metrics = model_results['test_metrics']
            metrics['model'] = model_name
            metrics['training_time'] = model_results.get('training_time', 0)
            comparison_data.append(metrics)
    
    if not comparison_data:
        st.warning("No test metrics found in results.")
        return
    
    df = pd.DataFrame(comparison_data)
    
    # Select metrics to display
    metric_cols = [col for col in df.columns if '@' in col]
    selected_metrics = st.multiselect(
        "Select metrics to compare",
        metric_cols,
        default=['ndcg@10', 'precision@10', 'recall@10']
    )
    
    if selected_metrics:
        # Display comparison table
        st.subheader("Performance Comparison")
        display_cols = ['model'] + selected_metrics + ['training_time']
        comparison_df = df[display_cols].round(4)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Create comparison chart
        st.subheader("Performance Visualization")
        
        fig = make_subplots(
            rows=1, cols=len(selected_metrics),
            subplot_titles=selected_metrics
        )
        
        for i, metric in enumerate(selected_metrics):
            fig.add_trace(
                go.Bar(
                    x=df['model'],
                    y=df[metric],
                    name=metric,
                    showlegend=False
                ),
                row=1, col=i+1
            )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Training time comparison
        st.subheader("Training Time Comparison")
        fig = px.bar(
            df,
            x='model',
            y='training_time',
            title="Training Time by Model"
        )
        st.plotly_chart(fig, use_container_width=True)


def show_user_recommendations(models, interactions, items, user_history):
    """Show user recommendation interface."""
    st.header("User Recommendations")
    
    if not models:
        st.warning("No trained models found. Please train models first.")
        return
    
    # User selection
    available_users = interactions['user_id'].unique()
    selected_user = st.selectbox("Select a user", available_users)
    
    # Model selection
    selected_model = st.selectbox("Select attention model", list(models.keys()))
    
    # Number of recommendations
    num_recs = st.slider("Number of recommendations", 5, 50, 20)
    
    if st.button("Get Recommendations"):
        model = models[selected_model]
        
        # Get user index
        user_to_idx = {user_id: idx for idx, user_id in enumerate(available_users)}
        user_idx = user_to_idx[selected_user]
        
        # Get recommendations
        with torch.no_grad():
            recommendations, attention_weights = model.evaluate_user(
                user_idx, user_history, num_recs
            )
        
        # Display recommendations
        st.subheader(f"Top {num_recs} Recommendations for {selected_user}")
        
        # Create recommendations dataframe
        rec_data = []
        for i, item_idx in enumerate(recommendations):
            item_id = f"item_{item_idx}"
            rec_data.append({
                'Rank': i + 1,
                'Item ID': item_id,
                'Item Index': item_idx
            })
        
        rec_df = pd.DataFrame(rec_data)
        st.dataframe(rec_df, use_container_width=True)
        
        # Show user history
        st.subheader("User Interaction History")
        user_interactions = interactions[interactions['user_id'] == selected_user]
        st.dataframe(user_interactions[['item_id', 'rating', 'timestamp']], use_container_width=True)
        
        # Attention weights visualization
        if attention_weights is not None:
            st.subheader("Attention Weights")
            st.write("Attention weights show which items in the user's history are most important for the recommendation.")
            
            # This would need to be implemented based on the specific attention mechanism
            st.info("Attention weight visualization would be implemented here based on the specific model architecture.")


def show_item_similarity(models, interactions, items):
    """Show item similarity interface."""
    st.header("Item Similarity")
    
    if not models:
        st.warning("No trained models found. Please train models first.")
        return
    
    # Model selection
    selected_model = st.selectbox("Select attention model", list(models.keys()))
    
    # Item selection
    available_items = interactions['item_id'].unique()
    selected_item = st.selectbox("Select an item", available_items)
    
    # Number of similar items
    num_similar = st.slider("Number of similar items", 5, 20, 10)
    
    if st.button("Find Similar Items"):
        model = models[selected_model]
        
        # Get item index
        item_to_idx = {item_id: idx for idx, item_id in enumerate(available_items)}
        item_idx = item_to_idx[selected_item]
        
        # Get item embedding
        with torch.no_grad():
            item_embedding = model.get_item_embeddings(torch.tensor([item_idx]))
            
            # Compute similarities with all items
            all_items = torch.arange(model.num_items)
            all_embeddings = model.get_item_embeddings(all_items)
            
            # Compute cosine similarity
            similarities = torch.cosine_similarity(
                item_embedding, all_embeddings, dim=1
            )
            
            # Get top similar items
            top_indices = torch.topk(similarities, num_similar + 1).indices[1:]  # Exclude self
            top_similarities = similarities[top_indices]
        
        # Display similar items
        st.subheader(f"Items Similar to {selected_item}")
        
        similar_data = []
        for i, (item_idx_sim, similarity) in enumerate(zip(top_indices, top_similarities)):
            item_id = f"item_{item_idx_sim.item()}"
            similar_data.append({
                'Rank': i + 1,
                'Item ID': item_id,
                'Similarity': f"{similarity.item():.4f}"
            })
        
        similar_df = pd.DataFrame(similar_data)
        st.dataframe(similar_df, use_container_width=True)
        
        # Show item details if available
        if items is not None:
            st.subheader("Item Details")
            item_details = items[items['item_id'] == selected_item]
            if not item_details.empty:
                st.dataframe(item_details, use_container_width=True)


def show_attention_analysis(models, interactions, items, user_history):
    """Show attention analysis."""
    st.header("Attention Analysis")
    
    if not models:
        st.warning("No trained models found. Please train models first.")
        return
    
    st.info("This section would show detailed attention weight analysis, including:")
    st.markdown("""
    - Attention patterns across different attention mechanisms
    - Visualization of attention weights for specific user-item interactions
    - Analysis of which features/items receive the most attention
    - Comparison of attention patterns between different models
    """)
    
    # Placeholder for attention analysis
    st.subheader("Attention Mechanism Comparison")
    
    attention_types = list(models.keys())
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Self-Attention**")
        st.write("- Focuses on relationships between items in user history")
        st.write("- Computes attention scores based on item similarities")
        
    with col2:
        st.write("**Multi-Head Attention**")
        st.write("- Captures different types of relationships")
        st.write("- Allows model to attend to different aspects simultaneously")
    
    st.write("**Transformer Attention**")
    st.write("- Combines self-attention with positional encoding")
    st.write("- Captures sequential patterns in user behavior")


if __name__ == '__main__':
    main()
