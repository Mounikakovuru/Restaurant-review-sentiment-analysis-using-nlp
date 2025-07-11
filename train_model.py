import pandas as pd
import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression # Using Logistic Regression
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings('ignore')

nltk.download('stopwords')

# Load dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

# --- Added: Check Class Distribution ---
print("--- Dataset Class Distribution ---")
print(dataset['Liked'].value_counts())
print("-" * 30)
# If the distribution is highly skewed (e.g., 900 positive, 100 negative),
# you might need to consider techniques like oversampling/undersampling.

# Clean and preprocess review
def clean_review(review):
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    words = review.split()

    # Negation handling
    negation_words = {"not", "no", "never", "n't"}
    processed = []
    skip = False
    for i in range(len(words)):
        if skip:
            skip = False
            continue
        if words[i] in negation_words and i + 1 < len(words):
            processed.append(words[i] + '_' + words[i + 1])
            skip = True
        else:
            processed.append(words[i])

    ps = PorterStemmer()
    processed = [ps.stem(word) for word in processed if word not in stopwords.words('english')]
    return ' '.join(processed)

# Create corpus
corpus = [clean_review(review) for review in dataset['Review']]

# TF-IDF Vectorization (Adjusted parameters)
# Using ngram_range=(1,2) to capture single words and two-word phrases (like "not_good")
# Increased max_features to allow more words/phrases
tfidf = TfidfVectorizer(max_features=2500, ngram_range=(1, 2)) # Experiment with max_features
X = tfidf.fit_transform(corpus).toarray()
y = dataset['Liked'].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train model (Changed to Logistic Regression)
# Increased max_iter for convergence with larger datasets/features
classifier = LogisticRegression(random_state=0, solver='liblinear', max_iter=500) # Added solver and max_iter
classifier.fit(X_train, y_train)

# Evaluate
y_pred = classifier.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# --- Added: Detailed Classification Report ---
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['Negative (0)', 'Positive (1)']))
print("-" * 30)
# Look at precision, recall, and F1-score for 'Negative (0)' class.
# If they are very low, the model is still struggling with negative predictions.

# Save model/vectorizer/corpus
pickle.dump(classifier, open('Restaurant-reviews-model.pkl', 'wb'))
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
pickle.dump(corpus, open('corpus.pkl', 'wb'))

print("\nModel, Vectorizer, and Corpus saved successfully!")