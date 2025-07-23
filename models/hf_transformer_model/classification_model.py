import os
import pandas as pd
import numpy as np
from transformers import pipeline
from tqdm import tqdm

class ZeroShotClassificationModel:
    def __init__(self, data: pd.DataFrame, name = 'zero-shot-classification', model='facebook/bart-large-mnli', device='mps'):

        self.data = data
        self.name = name
        self.model = model
        self.device = device

        self.classifier = None
        self.categories = None

        self.predicted_categories_missing = None
        self.missing_categories = None
        self.isbns = None

        self._prepare_data()
        self._get_classifer()


    
    def _prepare_data(self):
        category_mapping = {
            "Fiction": "Fiction",
            "Juvenile Fiction": "Children's Fiction",
            "Biography & Autobiography": "Nonfiction",
            "History": "Nonfiction",
            "Literary Criticism": "Nonfiction",
            "Philosophy": "Nonfiction",
            "Religion": "Nonfiction",
            "Comics & Graphic Novels": "Fiction",
            "Drama": "Fiction",
            "Juvenile Nonfiction": "Children's Nonfiction",
            "Science": "Nonfiction",
            "Poetry": "Fiction"
        }

        self.categories = ['Fiction', 'Nonfiction']
        self.data["simple_categories"] = self.data["categories"].map(category_mapping)

        self.isbns = []
        self.predicted_categories_missing = []
        self.missing_categories = self.data[self.datap['simple_categories'].isna(), 
        ["isbn13", "description"]].reset_index(drop=True)

    def _get_classifer(self):
        self.classifer = pipeline(self.name,
                 model=self.model,
                 device=self.device)
    
    def classify(self):
        def binary_classification():
            predictions = self.classifer(self.sequence, self.categories)
            max_index = np.argmax(predictions['scores'])
            max_label = predictions['labels'][max_index]
            return max_label
        for i in tqdm(range(len(self.missing_categories))):
            sequence = self.missing_categories['description'][i]
            self.predicted_categories_missing.append(binary_classification(sequence, self.fiction_categories))
            self.isbns.append(self.missing_categories['isbn13'][i])
        
        self.data = pd.merge(
            self.data,
            pd.DataFrame({"isbn13":self.isbns, "predicted_categories":self.predicted_categories_missing}),
            on='isbn13',
            how='left'
        )
        self.data['simple_categories'] = np.where(self.data['simple_categories'].isna(), self.data['predicted_categories'], self.data['simple_categories'])
        self.data = self.data.drop(columns=['predicted_categories','simple_cateogires', 'description_y'])
    


class TextClassificationModel:
    def __init__(self, data: pd.DataFrame, name = 'text-classification', model='j-hartmann/emotion-english-distilroberta-base',device='mps'):

        self.data = data
        self.name = name
        self.model = model
        self.top_k = None
        self.device = device

        self.classifer = None
        self.labels = None
        self.isbns = None
        self.emotion_scores = None

        self._prepare_data()
        self._get_classifer()

    def _prepare_data(self):
        self.labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
        self.isbns = []
        self.emotion_scores = {label: [] for label in self.labels}


    def _get_classifer(self):
        self.classifer = pipeline(self.name,
                 model=self.model,
                 top_k = self.top_k,
                 device=self.device)
    
    def classify(self):

        def calculate_max_emotion(description, emotion_labels):
            per_emotion_scores = {label: [] for label in emotion_labels}
            for d in description:
                sorted_predictions = sorted(d, key=lambda x: x['label'])
                for index, label in enumerate(emotion_labels):
                    per_emotion_scores[label].append(sorted_predictions[index]['score'])

            return {label: np.max(scores) for label, scores in per_emotion_scores.items()}
        
        for i in tqdm(range(len(self.data))):
            self.isbns.append(self.data['isbn13'][i])
            sentences = self.data['description'][i].split(".")
            predictions = self.classifer(sentences)
            max_scores = calculate_max_emotion(predictions, self.labels)
            
            for label in self.labels:
                self.emotion_scores[label].append(max_scores[label])
            
        emotion_scores_df = pd.DataFrame(self.emotion_scores)
        emotion_scores_df['isbn13'] = self.isbns

        self.data = pd.merge(
            self.data,
            emotion_scores_df,
            on='isbn13')

        




    
