from flask import Flask, render_template, request
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Load models and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Initialize the Flask app
app = Flask(__name__)

# Initialize the PorterStemmer
ps = PorterStemmer()

# Preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = [i for i in text if i.isalnum()]

    text = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]

    text = [ps.stem(i) for i in text]
    return " ".join(text)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_sms = request.form.get('message')

    # 1. Preprocess
    transform_sms = transform_text(input_sms)

    # 2. Vectorize
    vector_input = tfidf.transform([transform_sms])

    # 3. Predict
    result = model.predict(vector_input)[0]

    # 4. Return Result
    if result == 1:
        prediction = "Spam"
    else:
        prediction = "Not Spam"

    return render_template('index.html', prediction=prediction, message=input_sms)

if __name__ == '__main__':
    app.run(debug=True)

