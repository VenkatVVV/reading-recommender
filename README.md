# 📚 Reading Recommender System

## 🏗️ Architecture Overview

This project implements a multi-faceted recommendation system with the following components:

- **Collaborative Filtering**: TensorFlow-based neural network for user-item recommendations
- **Semantic Search**: LangChain + Gemini Vector Embeddings for content-based recommendations
- **Sentiment Analysis**: Hugging Face classification for emotional content analysis
- **API Layer**: FastAPI for inference serving
- **Containerization**: Docker for consistent deployment
- **Orchestration**: Google Kubernetes Engine (GKE) for scalable deployment

## 🚀 Features

### 1. TensorFlow Collaborative Filtering
- Neural network-based recommendation model
- User and item embedding learning
- Cold-start handling for new users
- Top-N recommendation generation

### 2. Semantic Search with LangChain & Gemini
- Content-based recommendations using book descriptions
- Google Generative AI embeddings (Gemini)
- ChromaDB vector database for similarity search
- Natural language query processing

### 3. Sentiment Analysis
- Hugging Face emotion classification model
- Multi-label emotion detection (anger, disgust, fear, joy, neutral, sadness, surprise)
- Emotion-aware recommendation filtering

### 4. Production-Ready API
- FastAPI-based REST API
- Health check endpoints
- Structured request/response models
- Error handling and validation

## 📁 Project Structure

```
reading-recommender/
├── main.py                          # FastAPI application
├── dashboard.py                     # Dashboard interface
├── Dockerfile                       # Container configuration
├── k8s_deployment.yaml             # Kubernetes deployment
├── requirements.txt                 # Python dependencies
├── data/
│   ├── processed/                   # Processed datasets
│   ├── raw/                        # Raw data files
│   └── ingest.py                   # Data ingestion scripts
├── models/
│   ├── tf_keras_model/             # TensorFlow model artifacts
│   ├── pytorch-model/              # PyTorch models
│   ├── train_and_save.py           # Model training script
│   ├── sentiment-analysis.ipynb    # Sentiment analysis pipeline
│   └── text-classification.ipynb   # Text classification
└── vector-embedding/
    ├── chroma_db/                  # Vector database
    └── semantic-search.ipynb       # Semantic search implementation
```

## 🛠️ Technology Stack

### Core ML/AI
- **TensorFlow**: Neural collaborative filtering model
- **Hugging Face Transformers**: Sentiment analysis pipeline
- **LangChain**: Semantic search orchestration
- **Google Generative AI**: Vector embeddings (Gemini)

### Backend & API
- **FastAPI**: High-performance API framework
- **Uvicorn**: ASGI server
- **Pydantic**: Data validation and serialization

### Data & Storage
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **ChromaDB**: Vector database for embeddings

### Infrastructure
- **Docker**: Containerization
- **Kubernetes**: Container orchestration
- **Google Kubernetes Engine (GKE)**: Cloud deployment

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker
- Google Cloud Platform account (for GKE deployment)
- Google API key for Gemini embeddings

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd reading-recommender
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   export GOOGLE_API_KEY="your-gemini-api-key"
   ```

4. **Run the FastAPI application**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

### Docker Deployment

1. **Build the Docker image**
   ```bash
   docker build -t reading-recommender .
   ```

2. **Run the container**
   ```bash
   docker run -p 8000:8000 reading-recommender
   ```

### Kubernetes Deployment

1. **Deploy to GKE**
   ```bash
   kubectl apply -f k8s_deployment.yaml
   ```

2. **Check deployment status**
   ```bash
   kubectl get pods
   kubectl get services
   ```

## 📊 API Endpoints

### Get Recommendations
```http
POST /recomend
Content-Type: application/json

{
  "user_id": 123
}
```

**Response:**
```json
{
  "user_id": 123,
  "recommendations": [
    {
      "rank": 1,
      "isbn": "9781234567890",
      "title": "Book Title",
      "score": 0.95
    }
  ]
}
```

### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "ok",
  "message": "Model is ready"
}
```

## 🔧 Model Training

### Collaborative Filtering Model
```bash
python models/train_and_save.py
```

### Semantic Search Setup
1. Process book descriptions in `vector-embedding/semantic-search.ipynb`
2. Generate embeddings using Gemini
3. Store in ChromaDB for similarity search

### Sentiment Analysis
1. Run emotion classification in `models/sentiment-analysis.ipynb`
2. Process book descriptions for emotional content
3. Generate emotion scores for recommendation filtering

## 📈 Performance & Scaling

### Resource Requirements
- **CPU**: 200m-500m per pod
- **Memory**: 512Mi-1Gi per pod
- **Storage**: Model artifacts and vector database

### Scaling Strategy
- Horizontal pod autoscaling based on CPU/memory usage
- Load balancer for traffic distribution
- Multiple replicas for high availability

## 🔒 Security Considerations

- API key management for external services
- Container image security scanning
- Network policies for pod communication
- Secrets management for sensitive data

## 📝 Development Workflow

1. **Data Processing**: Use notebooks in `data/` for EDA and preprocessing
2. **Model Development**: Train models using scripts in `models/`
3. **API Development**: Extend FastAPI endpoints in `main.py`
4. **Testing**: Add unit tests for API endpoints
5. **Deployment**: Use Docker and Kubernetes for production deployment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google Generative AI for embedding capabilities
- Hugging Face for pre-trained sentiment models
- LangChain for semantic search framework
- TensorFlow for collaborative filtering implementation

---

**Note**: This project requires appropriate API keys and cloud credentials for full functionality. Please ensure you have the necessary permissions and quotas set up in your Google Cloud Platform account. 
