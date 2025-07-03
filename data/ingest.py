import pandas as pd
import os
import kaggle
import argparse

def download_kaggle_dataset(dataset_name, path='data/raw'):
    """
    Downloads a dataset from Kaggle and saves it to the specified path.
    """
    # Create directory if it doesn't exist
    os.makedirs(path, exist_ok=True)
    
    try:
        # Download the dataset
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            dataset_name,
            path=path,
            unzip=True
        )
        print(f"Successfully downloaded dataset '{dataset_name}' to {path}")
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")

def list_downloaded_files(path='data/raw'):
    """
    Lists all files in the specified directory.
    """
    if os.path.exists(path):
        files = os.listdir(path)
        print(f"\nFiles in {path}:")
        for file in files:
            print(f"- {file}")
    else:
        print(f"Directory {path} does not exist")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download a Kaggle dataset')
    parser.add_argument('dataset_name', help='Name of the Kaggle dataset (e.g. "username/dataset-name")')
    args = parser.parse_args()
    dataset_name = args.dataset_name
    download_kaggle_dataset(dataset_name)
    list_downloaded_files()
