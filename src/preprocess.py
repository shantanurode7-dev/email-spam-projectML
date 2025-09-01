import pandas as pd
import re
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(r'C:\Users\shant\OneDrive\Documents\Email spam\data\rawdata\raw.csv')

df = df[['text','label_num']]

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    filtered_words = [stemmer.stem(word) for word in words if word not in stop_words and len(word) > 2]
    processed_text = ' '.join(filtered_words)
    
    return processed_text

df['cleaned_text'] = df['text'].apply(preprocess_text)

import os 
os.makedirs(r'C:\Users\shant\OneDrive\Documents\Email spam\data\processeddata',exist_ok=True)
df.to_csv(r'C:\Users\shant\OneDrive\Documents\Email spam\data\processeddata\processeddd.csv',index=False)
            