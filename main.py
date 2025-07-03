import tensorflow as tf
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

# Configurations 
MODEL_DIR = 'models/tf_keras_model'
BOOKS_DATA_PATH = 'data/processed/amazon_books.csv'
TOP_N_RECOMMENDATIONS = 10

# Load the model and Data
app = FastAPI(
    title="Reading Recommender API",
    description = "API for recommending books based on user preferences",
    version="1.0.0"
)

user_model = None
item_embeddings = None
user_id_mapping = None
reverse_item_id_mapping = None
isbn_to_title_mapping = None

# Pydantic Models for Request and Response
class RecommendationRequest(BaseModel):
    user_id: int

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: list[dict]

def load_book_titles():
    """Load book titles from the CSV file and create ISBN to title mapping"""
    try:
        books_df = pd.read_csv(BOOKS_DATA_PATH)
        # Create mapping from ISBN to book title
        isbn_to_title = dict(zip(books_df['isbn'], books_df['title']))
        print(f"Loaded {len(isbn_to_title)} book titles")
        return isbn_to_title
    except Exception as e:
        print(f"Error loading book titles: {e}")
        return {}

@app.on_event("startup")
async def load_resources():
    global user_model, item_embeddings, user_id_mapping, reverse_item_id_mapping, isbn_to_title_mapping

    print(f"Loading resources from {MODEL_DIR}...")

    try: 
        user_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'user_model.keras'))
        print("User model loaded successfully")

        item_embeddings = np.load(os.path.join(MODEL_DIR, 'item_embeddings.npy'))
        print("Item embeddings loaded successfully")

        user_id_mapping_df = pd.read_csv(os.path.join(MODEL_DIR, 'user_id_mapping.csv'), index_col=0)
        user_id_mapping = {int(k): int(v['mapped_id']) for k, v in user_id_mapping_df.iterrows()}
        print("User ID mapping loaded successfully")

        reverse_item_id_mapping_df = pd.read_csv(os.path.join(MODEL_DIR, 'reverse_item_id_mapping.csv'), index_col=0)
        reverse_item_id_mapping = {(k): (v['original_id']) for k, v in reverse_item_id_mapping_df.iterrows()}
        print("Reverse item ID mapping loaded successfully")

        # Load book titles
        isbn_to_title_mapping = load_book_titles()
        print("Book titles loaded successfully")

    except Exception as e:
        print(f"Error loading resources: {e}")
        raise RuntimeError("Failed to load resources")
    
# Recommendation Endpoint
@app.post("/recomend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    if user_model is None or item_embeddings is None or user_id_mapping is None or reverse_item_id_mapping is None:
        raise HTTPException(status_code=503, detail="Resources not loaded")
    
    original_user_id = request.user_id

    # Get mapped user ID
    mapped_user_id = user_id_mapping.get(original_user_id)
    if mapped_user_id is None: 
        # Cold start
        return RecommendationResponse(
            user_id=original_user_id,
            recommendations=[]
        )
    
    
    # Get user embedding
    user_embedding_tensor = tf.constant([mapped_user_id], dtype=tf.int32)
    user_vec = user_model.predict(user_embedding_tensor, verbose=0)[0]


    # Compute similarity scores
    scores = np.dot(item_embeddings, user_vec)

    #Get top N recommendations
    top_n_indices = np.argsort(scores)[::-1][:TOP_N_RECOMMENDATIONS]

    # Convert mapped item IDs back to original item IDs and get book titles
    recommendations = []
    for i, idx in enumerate(top_n_indices):
        isbn = reverse_item_id_mapping[idx]
        book_title = isbn_to_title_mapping.get(isbn, f"Unknown Book (ISBN: {isbn})")
        recommendations.append({
            "rank": i + 1,
            "isbn": isbn,
            "title": book_title,
            "score": float(scores[idx])
        })

    print(f"Recommended {TOP_N_RECOMMENDATIONS} books for user {original_user_id}")
    for rec in recommendations:
        print(f"  {rec['rank']}. {rec['title']} (ISBN: {rec['isbn']}, Score: {rec['score']:.3f})")

    return RecommendationResponse(
        user_id=original_user_id,
        recommendations=recommendations
    )

# Health Check Endpoint
@app.get("/health")
async def health_check():
    if user_model is not None and item_embeddings is not None:
        return {"status": "ok", "message": "Model is ready"}
    else:
        return {"status": "degraded", "message": "Recommender API is running but model is not loaded"}









