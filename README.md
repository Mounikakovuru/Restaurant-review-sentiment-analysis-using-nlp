# 🍽️ Restaurant Review Sentiment Analysis using NLP

This project performs sentiment analysis on restaurant reviews using Natural Language Processing (NLP) techniques and machine learning models. It helps identify whether a review is **positive** or **negative**, aiding restaurant owners and analysts in understanding customer satisfaction.

---

## 🚀 Features

- Preprocesses text using NLP techniques (tokenization, stopword removal, etc.)
- Converts reviews into numerical vectors using TF-IDF
- Trains ML models (Logistic Regression, Naive Bayes, etc.) to classify sentiment
- Evaluates model performance using accuracy, precision, recall, and F1-score
- Optional: Flask-based web app for demo/deployment

---

## 🧠 Technologies Used

- **Python 3.9+**
- **Pandas, NumPy** – Data handling
- **NLTK / SpaCy** – Text preprocessing
- **Scikit-learn** – Machine learning
- **Matplotlib / Seaborn** – Visualization
- **Flask / Streamlit** – Web interface (optional)

---

## 🧪 Model Training Steps

1. **Data Preprocessing**
   - Lowercasing, punctuation removal
   - Stopword removal, tokenization
   - Lemmatization/Stemming

2. **Feature Extraction**
   - TF-IDF Vectorizer to convert text to numeric form

3. **Model Training**
   - Logistic Regression / Naive Bayes
   - Train/test split (e.g., 80/20)

4. **Evaluation**
   - Accuracy, Precision, Recall, F1-score
   - Confusion Matrix

---

## 📊 Example Output

| Review Text                                  | Sentiment Prediction |
|----------------------------------------------|-----------------------|
| "The food was delicious and service amazing" | ✅ Positive            |
| "Terrible experience. Not going back again." | ❌ Negative            |

---

## 🧩 Future Improvements

- Deploy as a web app (Flask/Streamlit)
- Handle neutral sentiments
- Add deep learning models (LSTM)
- Expand dataset to other domains

---

## 🙌 Acknowledgments

- [Kaggle Datasets](https://www.kaggle.com/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [NLTK Library](https://www.nltk.org/)
