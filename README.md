# 📰 Fake News Detector using Machine Learning

This project is a simple and effective web-based application that classifies news articles as **Fake** or **Real** using natural language processing and machine learning. Built with **Streamlit** for interactivity, it leverages **Logistic Regression**, **text stemming**, and **TF-IDF vectorization** to make predictions on user-submitted content.

## 🚀 Features

- ✅ Interactive web interface with Streamlit
- 🔍 Text preprocessing: cleaning, stopwords removal, and stemming with NLTK
- 📊 Feature extraction using TF-IDF
- 🧠 Model training using Logistic Regression (scikit-learn)
- 📈 Real-time predictions for entered news articles

## 📚 Dataset

The model is trained on a dataset of news articles (`train.csv`).

## 🧠 Machine Learning Workflow

1. Combine `author` and `title` for richer content.
2. Clean and preprocess text using regular expressions and stemming.
3. Convert text to numerical features using TF-IDF.
4. Split data into training/testing sets.
5. Train a logistic regression model.
6. Classify incoming news snippets via Streamlit UI.

## 🛠️ Tech Stack

| Component      | Tool/Library           |
|----------------|------------------------|
| Interface      | Streamlit              |
| ML Model       | scikit-learn (Logistic Regression) |
| Text Processing| NLTK (Stopwords, Stemmer) |
| Vectorization  | TfidfVectorizer        |
| Data Handling  | Pandas                 |
| Programming    | Python                 |

## 📦 Installation

```bash
pip install streamlit scikit-learn nltk pandas
