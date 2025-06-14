import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple

from src.models.tensorflow.recommender import TensorFlowRecommender
from src.models.pytorch.recommender import PyTorchRecommender

def load_data(data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess the dataset.
    
    Args:
        data_path: Path to the dataset file
        
    Returns:
        Tuple of (user_ids, item_ids, ratings) as numpy arrays
    """
    # Read the dataset
    df = pd.read_csv(data_path)
    
    # Assuming the dataset has columns: user_id, item_id, rating
    # If your dataset has different column names, modify accordingly
    user_ids = df['user_id'].values
    item_ids = df['item_id'].values
    ratings = df['rating'].values
    
    # Normalize ratings to [0, 1] range
    ratings = (ratings - ratings.min()) / (ratings.max() - ratings.min())
    
    return user_ids, item_ids, ratings

def prepare_data(
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    ratings: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray],
          Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Split data into training and validation sets.
    
    Args:
        user_ids: Array of user IDs
        item_ids: Array of item IDs
        ratings: Array of ratings
        test_size: Proportion of data to use for validation
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_data, val_data) where each is a tuple of (user_ids, item_ids, ratings)
    """
    # Split the data
    (train_user_ids, val_user_ids,
     train_item_ids, val_item_ids,
     train_ratings, val_ratings) = train_test_split(
        user_ids, item_ids, ratings,
        test_size=test_size,
        random_state=random_state
    )
    
    return (train_user_ids, train_item_ids, train_ratings), \
           (val_user_ids, val_item_ids, val_ratings)

def train_model(
    model_type: str,
    train_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    val_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    num_users: int,
    num_items: int,
    config: Dict
) -> None:
    """
    Train the recommendation model.
    
    Args:
        model_type: Either 'tensorflow' or 'pytorch'
        train_data: Training data tuple
        val_data: Validation data tuple
        num_users: Number of unique users
        num_items: Number of unique items
        config: Model configuration dictionary
    """
    # Update config with dataset info
    config['num_users'] = num_users
    config['num_items'] = num_items
    
    # Create model
    if model_type.lower() == 'tensorflow':
        model = TensorFlowRecommender(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=config['embedding_dim'],
            hidden_layers=config['hidden_layers'],
            dropout_rate=config['dropout_rate']
        )
    elif model_type.lower() == 'pytorch':
        model = PyTorchRecommender(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=config['embedding_dim'],
            hidden_layers=config['hidden_layers'],
            dropout_rate=config['dropout_rate']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train the model
    history = model.train(
        train_data=train_data,
        validation_data=val_data,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate']
    )
    
    # Save the model
    model_dir = Path('models') / model_type.lower()
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_dir / 'model.pt'))
    
    # Save training history
    history_path = model_dir / 'history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f)
    
    # Save updated config
    config_path = model_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Train a recommendation model")
    parser.add_argument(
        '--model',
        type=str,
        choices=['tensorflow', 'pytorch'],
        required=True,
        help="Model type to train"
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help="Path to the dataset file"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.json',
        help="Path to the model configuration file"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Load and prepare data
    user_ids, item_ids, ratings = load_data(args.data)
    train_data, val_data = prepare_data(
        user_ids, item_ids, ratings,
        test_size=config['test_size'],
        random_state=config['random_state']
    )
    
    # Get number of unique users and items
    num_users = len(np.unique(user_ids))
    num_items = len(np.unique(item_ids))
    
    # Train the model
    train_model(
        model_type=args.model,
        train_data=train_data,
        val_data=val_data,
        num_users=num_users,
        num_items=num_items,
        config=config
    )

if __name__ == "__main__":
    main() 