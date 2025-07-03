import pandas as pd
import numpy as np
from tf_keras_model.recommender_model import RecommenderModel


#Load data
data = pd.read_csv('/Users/venkatvive/Documents/projects/reading-recommender/data/processed/amazon_ratings.csv')

#Initialize the model
model = RecommenderModel(data, embedding_dim=32, model_dir='models/tf_keras_model')

#Train the model
model.train(epochs=5, batch_size=256, verbose=1)

#Save the model
model.save_for_inference()