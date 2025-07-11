from flask import Flask, render_template, request
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')

# Load model and vectorizer
classifier = pickle.load(open('Restaurant-reviews-model.pkl', 'rb'))
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))

def clean_review(review):
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    words = review.split()

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

def predict_review(message):
    cleaned = clean_review(message)
    temp = tfidf.transform([cleaned]).toarray()
    return classifier.predict(temp)[0]

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        if message:
            result = predict_review(message)
            return render_template('index.html', result=result, message=message)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
