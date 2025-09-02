import streamlit as st
import pandas as pd 
import numpy as np 
import pickle 
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


st.header('spam checker!!')

email = st.text_input('Enter your email to check')

df = pd.DataFrame(data=[email],columns=['text'])


stemmer = PorterStemmer()
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    filtered_words = [stemmer.stem(word) for word in words if word not in stop_words and len(word) > 2]
    processed_text = ' '.join(filtered_words)
    
    return processed_text

df['cleaned_text'] = df['text'].apply(preprocess_text)


file_path = 'models/model.pkl'
with open(file_path, 'rb') as file:
    loaded_model = pickle.load(file)

file_path = 'models/vectorizer.pkl'
with open(file_path, 'rb') as file:
    tfidf = pickle.load(file)

x = tfidf.transform(df['cleaned_text']).toarray()

button = st.button('predict')

if button:
    prediction = loaded_model.predict(x)
    if prediction==0:
        st.success('Not Spam')
    else:
        st.warning('spam')




    

    