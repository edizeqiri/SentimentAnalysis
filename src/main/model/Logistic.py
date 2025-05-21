import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

from string import punctuation 
from nltk.tokenize import word_tokenize
from nltk.stem import LancasterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import re
import warnings
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import stopwords
from icecream import ic
ic.configureOutput(includeContext=True)

class Logistic:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.model = None

    def pre_process(self, X):
        X['clean'] = X['sentence'].apply(ic(self.remove_unnecessary_characters)).apply(ic(self.normalize_text)).apply(ic(self.remove_stopwords)).astype(str)
        return X

    def train(self, X, y):
        ic("Start training")
        ic(X)
        X = self.pre_process(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.vectorizer = TfidfVectorizer()
        XV_train = self.vectorizer.fit_transform(X_train['clean'])
        XV_test = self.vectorizer.transform(X_test['clean'])
        ic("Training model")
        lr = LogisticRegression(n_jobs=-1, max_iter=self.num_iterations, C=1/self.learning_rate)
        self.model = lr.fit(XV_train, y_train)
        ic("Model trained")
        y_pred = self.model.predict(XV_test)
        ic("Predicting")
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        ic("Accuracy: ", accuracy)
        ic("Precision: ", precision)
        ic("Recall: ", recall)
        ic("F1: ", f1)
        """n = len(y_test)
        mae = np.mean(np.abs(y_test - y_pred))
        custom_score = 0.5 * (2 - mae)
        
        self.X_test = X_test
        self.y_test = y_test
        self.XV_test = XV_test
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'custom_sentiment_score': custom_score
        }"""
    
    def calculate_sentiment_score(self, y_true, y_pred):
        n = len(y_true)
        mae = np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
        return 0.5 * (2 - mae)
        
    def predict(self, X):
        if self.model == None:
            raise ValueError("Model has not been trained yet. Please call the train method before predicting.")
        X['label'] = self.model.predict(X)
        return X
   
    def remove_unnecessary_characters(self, text):
        text = re.sub(r'<.*?>', '', str(text))
        text = re.sub(r'[^a-zA-Z0-9\s]', '', str(text))
        text = re.sub(r'\s+', ' ', str(text)).strip()
        return text
    
    def normalize_text(self, text):
        if isinstance(text, str):
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
        else:
            text = str(text)
        return text
    
    def remove_stopwords(self, text):
        if isinstance(text, str):
            words = text.split()        
            filtered_words = [word for word in words if word.lower() not in stopwords.words('english')]
            filtered_text = ' '.join(filtered_words)
        else:
            filtered_text = ''
        return filtered_text


