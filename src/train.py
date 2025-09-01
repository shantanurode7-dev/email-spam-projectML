import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


df = pd.read_csv(r'C:\Users\shant\OneDrive\Documents\Email spam\data\processeddata\processeddd.csv')

tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
x = tfidf_vectorizer.fit_transform(df['cleaned_text']).toarray()
y = df['label_num']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

nb_model = MultinomialNB()
nb_model.fit(x_train, y_train)
pred = nb_model.predict(x_test)

print(classification_report(y_test,pred))

import pickle 
import os 
os.makedirs(r'C:\Users\shant\OneDrive\Documents\Email spam\models',exist_ok=True)


with open(r'C:\Users\shant\OneDrive\Documents\Email spam\models\model.pkl','wb') as file:
    pickle.dump(nb_model,file)

with open(r'C:\Users\shant\OneDrive\Documents\Email spam\models\vectorizer.pkl','wb') as file:
    pickle.dump(tfidf_vectorizer,file)

