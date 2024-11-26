import re
import string

import pandas as pd
from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Load and preprocess the data
def load_data():
    data_fake = pd.read_csv('Fake.csv')
    data_true = pd.read_csv('True.csv')
    
    data_fake['class'] = 0
    data_true['class'] = 1
    
    # Combine datasets
    data = pd.concat([data_fake, data_true], axis=0).drop(['title', 'subject', 'date'], axis=1)
    data['text'] = data['text'].apply(preprocess_text)
    
    x = data['text']
    y = data['class']
    
    return train_test_split(x, y, test_size=0.25, random_state=42)

# Split data and vectorize text
x_train, x_test, y_train, y_test = load_data()
vectorizer = TfidfVectorizer()
xv_train = vectorizer.fit_transform(x_train)
xv_test = vectorizer.transform(x_test)

# Initialize models
LR = LogisticRegression()
LR.fit(xv_train, y_train)

DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)

RF = RandomForestClassifier()
RF.fit(xv_train, y_train)

# Prediction function
def classify_news(text):
    transformed_text = vectorizer.transform([preprocess_text(text)])
    pred_LR = LR.predict(transformed_text)[0]
    pred_DT = DT.predict(transformed_text)[0]
    pred_RF = RF.predict(transformed_text)[0]
    
    return {
        "Logistic Regression": "Fake News" if pred_LR == 0 else "Real News",
        "Decision Tree": "Fake News" if pred_DT == 0 else "Real News",
        "Random Forest": "Fake News" if pred_RF == 0 else "Real News"
    }

# Define the route for prediction
@app.route('/predict_news', methods=['GET', 'POST'])
def predict_news():
    if request.method == 'POST':
        news_text = request.form['news_text']
        predictions = classify_news(news_text)
        return render_template('predicttext.html', news_text=news_text, predictions=predictions)
    return render_template('predicttext.html')

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
