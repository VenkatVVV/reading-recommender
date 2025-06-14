import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity

from src.models.tensorflow.recommender import TensorFlowRecommender
from src.models.pytorch.recommender import PyTorchRecommender

def load_model(model_type: str, model_path: str, config_path: str):
    """
    Load a trained model.
    
    Args:
        model_type: Either 'tensorflow' or 'pytorch'
        model_path: Path to the saved model
        config_path: Path to the model configuration file
        
    Returns:
        Loaded model instance
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create model instance
    if model_type.lower() == 'tensorflow':
        model = TensorFlowRecommender(
            num_users=config['num_users'],
            num_items=config['num_items'],
            embedding_dim=config['embedding_dim'],
            hidden_layers=config['hidden_layers'],
            dropout_rate=config['dropout_rate']
        )
    elif model_type.lower() == 'pytorch':
        model = PyTorchRecommender(
            num_users=config['num_users'],
            num_items=config['num_items'],
            embedding_dim=config['embedding_dim'],
            hidden_layers=config['hidden_layers'],
            dropout_rate=config['dropout_rate']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load model weights
    model.load_model(model_path)
    return model

def get_item_embeddings(model) -> np.ndarray:
    """
    Get item embeddings from the model.
    
    Args:
        model: Loaded model instance
        
    Returns:
        Array of item embeddings
    """
    if isinstance(model, TensorFlowRecommender):
        # For TensorFlow model
        item_embeddings = model.model.get_layer('item_embedding').get_weights()[0]
    else:
        # For PyTorch model
        item_embeddings = model.model.item_embedding.weight.detach().cpu().numpy()
    
    return item_embeddings

def get_similar_items(
    model,
    item_id: int,
    n_recommendations: int = 10,
    exclude_items: Optional[List[int]] = None
) -> List[Tuple[int, float]]:
    """
    Get similar items based on a reference item using embedding similarity.
    
    Args:
        model: Loaded model instance
        item_id: ID of the reference item
        n_recommendations: Number of similar items to return
        exclude_items: List of item IDs to exclude from recommendations
        
    Returns:
        List of (item_id, similarity_score) tuples
    """
    # Get item embeddings
    item_embeddings = get_item_embeddings(model)
    
    # Get reference item embedding
    ref_embedding = item_embeddings[item_id].reshape(1, -1)
    
    # Calculate cosine similarity with all items
    similarities = cosine_similarity(ref_embedding, item_embeddings).flatten()
    
    # Create list of (item_id, similarity) tuples
    item_similarities = list(enumerate(similarities))
    
    # Remove the reference item
    item_similarities = [(i, s) for i, s in item_similarities if i != item_id]
    
    # Remove excluded items if provided
    if exclude_items:
        item_similarities = [(i, s) for i, s in item_similarities if i not in exclude_items]
    
    # Sort by similarity score in descending order
    item_similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return top-N similar items
    return item_similarities[:n_recommendations]

def get_recommendations(
    model,
    user_id: int,
    n_recommendations: int = 10,
    exclude_rated: bool = True,
    rated_items: List[int] = None
) -> List[Tuple[int, float]]:
    """
    Get recommendations for a user.
    
    Args:
        model: Loaded model instance
        user_id: ID of the user
        n_recommendations: Number of recommendations to return
        exclude_rated: Whether to exclude items the user has already rated
        rated_items: List of item IDs that the user has already rated
        
    Returns:
        List of (item_id, predicted_rating) tuples
    """
    # Get recommendations
    recommendations = model.get_recommendations(
        user_id=user_id,
        n_recommendations=n_recommendations,
        exclude_rated=exclude_rated
    )
    
    # Filter out rated items if requested
    if exclude_rated and rated_items:
        recommendations = [
            (item_id, score) for item_id, score in recommendations
            if item_id not in rated_items
        ][:n_recommendations]
    
    return recommendations

def main():
    parser = argparse.ArgumentParser(description="Make recommendations using a trained model")
    parser.add_argument(
        '--model',
        type=str,
        choices=['tensorflow', 'pytorch'],
        required=True,
        help="Model type to use"
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default='models',
        help="Directory containing the saved model"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.json',
        help="Path to the model configuration file"
    )
    
    # Create subparsers for different recommendation types
    subparsers = parser.add_subparsers(dest='command', help='Recommendation type')
    
    # User-based recommendations
    user_parser = subparsers.add_parser('user', help='Get user-based recommendations')
    user_parser.add_argument(
        '--user-id',
        type=int,
        required=True,
        help="ID of the user to make recommendations for"
    )
    user_parser.add_argument(
        '--n-recommendations',
        type=int,
        default=10,
        help="Number of recommendations to return"
    )
    user_parser.add_argument(
        '--rated-items',
        type=str,
        help="Path to a JSON file containing rated items for the user"
    )
    
    # Item-based recommendations
    item_parser = subparsers.add_parser('item', help='Get item-based recommendations')
    item_parser.add_argument(
        '--item-id',
        type=int,
        required=True,
        help="ID of the reference item"
    )
    item_parser.add_argument(
        '--n-recommendations',
        type=int,
        default=10,
        help="Number of similar items to return"
    )
    item_parser.add_argument(
        '--exclude-items',
        type=str,
        help="Path to a JSON file containing item IDs to exclude"
    )
    
    args = parser.parse_args()
    
    # Load model
    model_path = Path(args.model_dir) / args.model.lower() / 'model.pt'
    model = load_model(args.model, str(model_path), args.config)
    
    if args.command == 'user':
        # Load rated items if provided
        rated_items = None
        if args.rated_items:
            with open(args.rated_items, 'r') as f:
                rated_items = json.load(f)
        
        # Get user-based recommendations
        recommendations = get_recommendations(
            model=model,
            user_id=args.user_id,
            n_recommendations=args.n_recommendations,
            exclude_rated=True,
            rated_items=rated_items
        )
        
        # Print recommendations
        print(f"\nTop {args.n_recommendations} recommendations for user {args.user_id}:")
        print("-" * 40)
        for i, (item_id, score) in enumerate(recommendations, 1):
            print(f"{i}. Item {item_id}: {score:.4f}")
    
    elif args.command == 'item':
        # Load excluded items if provided
        exclude_items = None
        if args.exclude_items:
            with open(args.exclude_items, 'r') as f:
                exclude_items = json.load(f)
        
        # Get item-based recommendations
        similar_items = get_similar_items(
            model=model,
            item_id=args.item_id,
            n_recommendations=args.n_recommendations,
            exclude_items=exclude_items
        )
        
        # Print similar items
        print(f"\nTop {args.n_recommendations} similar items to item {args.item_id}:")
        print("-" * 40)
        for i, (item_id, similarity) in enumerate(similar_items, 1):
            print(f"{i}. Item {item_id}: {similarity:.4f}")

if __name__ == "__main__":
    main() 