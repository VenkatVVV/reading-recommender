# Reading Recommender

A hybrid recommendation engine built using TensorFlow and PyTorch, capable of processing Kaggle datasets to provide personalized reading recommendations.

## Project Structure

```
reading-recommender/
├── data/                   # Data directory for Kaggle datasets
├── src/                    # Source code
│   ├── data/              # Data processing utilities
│   ├── models/            # Model implementations
│   │   ├── tensorflow/    # TensorFlow-based models
│   │   └── pytorch/       # PyTorch-based models
│   ├── training/          # Training scripts
│   └── utils/             # Utility functions
├── notebooks/             # Jupyter notebooks for exploration
├── requirements.txt       # Project dependencies
├── setup.py              # Package installation file
└── README.md             # This file
```

## Setup Instructions

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the package in development mode:
   ```bash
   pip install -e .
   ```
   This will install all required dependencies and make the package importable.

3. Set up Kaggle API:
   - Go to your Kaggle account settings
   - Create a new API token
   - Place the kaggle.json file in ~/.kaggle/
   - Set appropriate permissions: `chmod 600 ~/.kaggle/kaggle.json`

## Usage

1. Download your desired Kaggle dataset:
   ```bash
   python src/data/download_dataset.py --dataset <dataset-name>
   ```

2. Train the model (choose one of these methods):
   ```bash
   # Method 1: Run as a module (recommended)
   python -m src.training.train --model [tensorflow|pytorch] --data data/ratings.csv
   
   # Method 2: Run script directly (after installing package)
   python src/training/train.py --model [tensorflow|pytorch] --data data/ratings.csv
   ```

3. Make predictions:
   ```bash
   python -m src.predict --model [tensorflow|pytorch] --input <user-data>
   ```

## Features

- Hybrid recommendation system using both TensorFlow and PyTorch
- Support for various Kaggle datasets
- Data preprocessing and feature engineering
- Model training and evaluation
- Easy-to-use prediction interface

## Contributing

Feel free to submit issues and enhancement requests!
