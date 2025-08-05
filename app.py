import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import joblib

# Load saved model and vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Preprocessing function (must match training exactly)
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [ps.stem(word) for word in text if word not in stop_words]
    return ' '.join(text)

# Streamlit UI
st.title("📰 Fake News Detector")
#st.write("Enter a news headline or article and check whether it's real or fake.")

input_text = st.text_area("📝 Enter news article text:")

if st.button("Check News"):
    if input_text.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        processed_text = preprocess_text(input_text)
        vect_text = vectorizer.transform([processed_text])
        prediction = model.predict(vect_text)

        if prediction[0] == 1:
            st.success("This news is REAL!")
        else:
            st.error("This news is FAKE.")
