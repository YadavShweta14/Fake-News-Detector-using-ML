import streamlit as st
import numpy as np
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# Load data
news_df = pd.read_csv('train.csv')
news_df = news_df.fillna(' ')
news_df['content'] = news_df['author'] + ' ' + news_df['title']

# Stemming function
ps = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower().split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    return ' '.join(stemmed_content)

# Preprocess content
news_df['content'] = news_df['content'].apply(stemming)

# Vectorize
vector = TfidfVectorizer()
X = vector.fit_transform(news_df['content'].values)
y = news_df['label'].values

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Streamlit app
st.title('📰 Fake News Detector')
input_text = st.text_area('Enter news article text below:')

if st.button("Check"):
    if input_text.strip() != "":
        input_text_processed = stemming(input_text)
        input_vector = vector.transform([input_text_processed])
        prediction = model.predict(input_vector)

        if prediction[0] == 1:
            st.error('⚠️ The News is Fake')
        else:
            st.success('✅ The News is Real')
    else:
        st.warning("Please enter some text to analyze.")
