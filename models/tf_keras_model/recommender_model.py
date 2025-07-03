import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import os

class RecommenderModel:
    def __init__(self, data: pd.DataFrame, embedding_dim: int = 32, model_dir : str = 'models/tf-keras-model'):
        if not all(col in data.columns for col in ['user_id', 'isbn', 'rating']):
            raise ValueError("Data must contain 'user_id', 'isbn', and 'rating' columns")
        
        self.data = data
        self.embedding_dim = embedding_dim
        self.model_dir = model_dir

        self.user_id_mapping = None
        self.item_id_mapping = None
        self.reverse_user_id_mapping = None
        self.reverse_item_id_mapping = None
        self.unique_user_ids = None
        self.unique_item_ids = None

        self.user_model = None
        self.item_model = None
        self.combined_training_model=None

        os.makedirs(self.model_dir, exist_ok=True)

        self._prepare_data()
        self._define_models()
        self._build_and_compile_combined_model()

    def _prepare_data(self):
        print("Preparing data...")
        self.unique_user_ids = self.data['user_id'].unique()
        self.unique_item_ids = self.data['isbn'].unique()

        #mappings from original ids to integers
        self.user_id_mapping = {idx: i for i, idx in enumerate(self.unique_user_ids)}
        self.item_id_mapping = {idx: i for i, idx in enumerate(self.unique_item_ids)}

        #reverse mappings for inference
        self.reverse_user_id_mapping = {v: k for k, v in self.user_id_mapping.items()}
        self.reverse_item_id_mapping = {v: k for k, v in self.item_id_mapping.items()}

        mapped_user_ids = self.data['user_id'].map(self.user_id_mapping).values
        mapped_item_ids = self.data['isbn'].map(self.item_id_mapping).values
        ratings = self.data['rating'].values

        #Create TF Dataset for training
        self.train_dataset = tf.data.Dataset.from_tensor_slices(
            ({'user_id':mapped_user_ids,'isbn':mapped_item_ids}, ratings)
        ).shuffle(len(self.data))
        
        print(f"total unique users: {len(self.unique_user_ids)}")
        print(f"total unique isbns: {len(self.unique_item_ids)}")
        print(f"total training examples: {len(self.train_dataset)}")

        print("Data preparation complete!")

    def _define_models(self):
        print("Defining User and Item embeddingmodels...")

        #User model: learns embeddings for each user
        user_id_input = keras.Input(shape=(1,), name='user_id', dtype=tf.int32)
        user_embedding = layers.Embedding(
            input_dim=len(self.unique_user_ids),    #number of unique users
            output_dim=self.embedding_dim,           #embedding dimension
            name='user_embedding'
        )(user_id_input)
        user_vec = layers.Flatten(name='user_flatten')(user_embedding) #Flatten to 1D vector
        self.user_model = keras.Model(inputs=user_id_input, outputs=user_vec, name="user_model")

        #Item Model: learns embeddings for each ISBN
        item_id_input = keras.Input(shape=(1,), name='isbn', dtype=tf.int32)
        item_embedding = layers.Embedding(
            input_dim=len(self.unique_item_ids),
            output_dim=self.embedding_dim,
            name='item_embedding'
        )(item_id_input)
        item_vec = layers.Flatten(name='item_flatten')(item_embedding)
        self.item_model = keras.Model(inputs=item_id_input, outputs=item_vec, name="item_model")

        print("User and Item embedding models defined!")

    def _build_and_compile_combined_model(self):

        print("Building and compiling combined model...")

        # Get outputs from individual user and item models

        user_embedding_output = self.user_model.output
        item_embedding_output = self.item_model.output

        # Define inputs for the combined model
        combined_input = [self.user_model.input, self.item_model.input]

        # Combine embeddings
        dot_product = layers.Dot(axes=1)([user_embedding_output, item_embedding_output])
        output = layers.Dense(1, activation='relu')(dot_product)

        self.combined_training_model = keras.Model(inputs=combined_input, outputs=output, name="Recommender_Training_Model")

        #Compile the combined model for training
        self.combined_training_model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )

        print("Combined model built and compiled!")
        self.combined_training_model.summary()

    def train(self, epochs:int=5, batch_size:int=256, verbose:int=1):
        print("Training the model...")

        print(f"Training for {epochs} epochs with batch size {batch_size}")

        # Apply batching and prefetching
        train_dataset_batched = self.train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        history = self.combined_training_model.fit(
            train_dataset_batched,
            epochs=epochs,
            verbose=verbose,
        )

        print("Training complete!")
        return history
    
    def save_for_inference(self):

        print("Saving model for inference...")
        
        all_mapped_user_ids_tensor = tf.constant(list(self.user_id_mapping.values()), dtype=tf.int32)
        user_embeddings = self.user_model.predict(all_mapped_user_ids_tensor, verbose=0)

        all_mapped_item_ids_tensor = tf.constant(list(self.item_id_mapping.values()), dtype=tf.int32)
        item_embeddings = self.item_model.predict(all_mapped_item_ids_tensor, verbose=0)

        print(f"User embeddings shape: {user_embeddings.shape}")
        print(f"Item embeddings shape: {item_embeddings.shape}")

        # save the user model used to get user embeddings during inference
        user_model_path = os.path.join(self.model_dir, 'user_model.keras')
        self.user_model.save(user_model_path)
        print(f"User model saved to {user_model_path}")

        #save item embeddings ( Should go into an Online Feature Store)
        np.save(os.path.join(self.model_dir, 'item_embeddings.npy'), item_embeddings)
        print(f"Item embeddings saved to {os.path.join(self.model_dir, 'item_embeddings.npy')}")

        # Save ID Mappings for lookup during inference
        pd.DataFrame.from_dict(self.user_id_mapping, orient='index', columns=['mapped_id']).to_csv(
            os.path.join(self.model_dir, 'user_id_mapping.csv')
        )
        print(f"User ID mapping saved to {os.path.join(self.model_dir, 'user_id_mapping.csv')}")

        pd.DataFrame.from_dict(self.reverse_item_id_mapping, orient='index', columns=['original_id']).to_csv(
            os.path.join(self.model_dir, 'reverse_item_id_mapping.csv')
        )

        print(f"Reverse item ID mapping saved to {os.path.join(self.model_dir, 'reverse_item_id_mapping.csv')}")

        print("Inference model saved successfully!")
        
        
        
        
        
