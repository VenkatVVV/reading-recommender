from setuptools import setup, find_packages

setup(
    name="reading-recommender",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "tensorflow>=2.8.0",
        "torch>=1.10.0",
        "scikit-learn>=0.24.2",
        "matplotlib>=3.4.3",
        "seaborn>=0.11.2",
        "jupyter>=1.0.0",
        "kaggle>=1.5.12",
        "tqdm>=4.62.3",
        "python-dotenv>=0.19.0"
    ],
    python_requires=">=3.7",
) 