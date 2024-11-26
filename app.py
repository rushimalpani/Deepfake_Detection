import os
import re
import sqlite3
import string
import numpy as np
import pandas as pd
import librosa
import cv2
import time

from flask import Flask, g, redirect, render_template, request, session, url_for, send_file
from functools import wraps
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from fpdf import FPDF

# Flask app initialization
app = Flask(__name__, static_folder='static')
app.secret_key = "KjhLJF54f6ds234H"

DATABASE = "mydb.sqlite3"
UPLOAD_FOLDER = "uploads"
MODEL_PATH = 'Models/WICServer/WCE_DLI_model.h5'
loaded_model = load_model(MODEL_PATH)

# === Fake News Detection Models ===
dataset_fake = pd.read_csv('Fake.csv')
dataset_true = pd.read_csv('True.csv')
dataset_fake['class'] = 0
dataset_true['class'] = 1

data = pd.concat([dataset_fake, dataset_true], axis=0).drop(['title', 'subject', 'date'], axis=1)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

data['text'] = data['text'].apply(preprocess_text)
x = data['text']
y = data['class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

vectorizer = TfidfVectorizer()
xv_train = vectorizer.fit_transform(x_train)
xv_test = vectorizer.transform(x_test)

LR = LogisticRegression()
LR.fit(xv_train, y_train)

DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)

RF = RandomForestClassifier()
RF.fit(xv_train, y_train)

def classify_news(text):
    transformed_text = vectorizer.transform([preprocess_text(text)])
    predictions = {
        "Logistic Regression": "Fake News" if LR.predict(transformed_text)[0] == 0 else "Real News",
        "Decision Tree": "Fake News" if DT.predict(transformed_text)[0] == 0 else "Real News",
        "Random Forest": "Fake News" if RF.predict(transformed_text)[0] == 0 else "Real News"
    }
    return predictions

# === Audio Classification Model ===
dataset_audio = pd.read_csv('dataset.csv')
num_mfcc = 100
num_mels = 128
num_chroma = 50

# Database connection
def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()

# === Helper Functions ===
def save_file(file):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    return file_path

def load_image(image_path):
    img = image.load_img(image_path, target_size=(256, 256))
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def classify_and_compare(image_path, threshold=0.5):
    img_array = load_image(image_path)
    predictions = loaded_model.predict(img_array)
    if predictions[0][0] >= threshold:
        deepfake_image = cv2.imread(image_path)
        heatmap = generate_forgery_heatmap(deepfake_image)
        heatmap_gray = generate_forgery_heatmap_grayscale(deepfake_image)
        overlay = cv2.addWeighted(deepfake_image, 0.3, heatmap, 1, 0)
        overlay_gray = cv2.addWeighted(deepfake_image, 0.3, heatmap_gray, 1, 0)
        create_pdf(image_path, overlay, overlay_gray, None, None)
        return "The uploaded image is a deepfake."
    return "The uploaded image is not a deepfake."

def generate_forgery_heatmap(image):
    reference_image = np.ones_like(image) * 255
    diff = cv2.absdiff(image, reference_image)
    _, mask = cv2.threshold(cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY), 60, 255, cv2.THRESH_BINARY)
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    return cv2.GaussianBlur(heatmap, (0, 0), sigmaX=3)

def generate_forgery_heatmap_grayscale(image):
    reference_image = np.ones_like(image) * 255
    diff = cv2.absdiff(image, reference_image)
    _, mask = cv2.threshold(cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY), 60, 255, cv2.THRESH_BINARY)
    heatmap_gray = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return cv2.GaussianBlur(heatmap_gray, (0, 0), sigmaX=3)

def create_pdf(original_image_path, image_with_heatmap, image_with_gray_heatmap, matching_image_path, psnr):
    pdf = FPDF()
    pdf.add_page()

    # Add a title
    pdf.set_font("Arial", "B", 18)
    pdf.cell(200, 10, "Deepfake Image Detected", ln=True, align="C")

    # Calculate the scale factor for each image
    max_width = 60
    max_height = 60
    scale_factor = min(max_width / pdf.w, max_height / pdf.h)

    # Add original image section
    pdf.ln(10)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(50, 10, "Impacted Image:", ln=False)
    pdf.ln(5)
    pdf.image(original_image_path, x=120, y=None, w=pdf.w * scale_factor)

    # Add image with color heatmap section
    pdf.ln(5)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(50, 10, "Image with Color Heatmap:", ln=False)
    pdf.ln(5)
    cv2.imwrite("heatmap_temp.png", cv2.cvtColor(image_with_heatmap, cv2.COLOR_BGR2RGB))
    pdf.image("heatmap_temp.png", x=120, y=None, w=pdf.w * scale_factor)
    os.remove("heatmap_temp.png")

    # Add image with grayscale heatmap section
    pdf.ln(5)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(50, 10, "Image with Grayscale Heatmap:", ln=False)
    pdf.ln(5)
    cv2.imwrite("gray_heatmap_temp.png", image_with_gray_heatmap)
    pdf.image("gray_heatmap_temp.png", x=120, y=None, w=pdf.w * scale_factor)
    os.remove("gray_heatmap_temp.png")

    # Add matching reference image section
    pdf.ln(5)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(50, 10, "Matching Reference Image:", ln=False)
    pdf.ln(5)
    if matching_image_path:
        pdf.image(matching_image_path, x=120, y=None, w=pdf.w * scale_factor)
    else:
        pdf.cell(200, 10, "No matching reference image found", ln=True, align="C")

    if psnr:
        psnr_trimmed = "{:.2f}".format(psnr)
        pdf.ln(5)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(50, 10, "PSNR Value: " + str(psnr_trimmed) + " dB", ln=False) 

    # Save the PDF
    pdf_output_path = "static/result.pdf"
    pdf.output(pdf_output_path)
    return pdf_output_path  # Return the path to the saved PDF


# === Routes ===
@app.route('/')
def home():
    return render_template('index.html', background_image="/static/image1.jpg")

@app.route('/about')
def about():
    return render_template('about.html', background_image="/static/image5.jpg")

@app.route('/login', methods=['GET', 'POST'])
def login():
    msg = ""
    if request.method == 'POST':
        email = request.form["email"]
        password = request.form["password"]
        cursor = get_db().cursor()
        cursor.execute("SELECT * FROM REGISTER WHERE EMAIL = ? AND PASSWORD = ?", (email, password))
        account = cursor.fetchone()
        if account:
            session['Loggedin'] = True
            session['id'] = account[0]
            session['email'] = account[1]
            return redirect(url_for('home'))
        else:
            msg = "Incorrect Email/password"
    return render_template('login.html', msg=msg)

@app.route('/register', methods=['GET', 'POST'])
def signup():
    msg = ''
    if request.method == 'POST':
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]
        confirm_password = request.form["confirm-password"]
        cursor = get_db().cursor()
        cursor.execute("SELECT * FROM REGISTER WHERE username = ?", (username,))
        account_username = cursor.fetchone()
        cursor.execute("SELECT * FROM REGISTER WHERE email = ?", (email,))
        account_email = cursor.fetchone()
        if account_username:
            msg = "Username already exists"
        elif account_email:
            msg = "Email already exists"
        elif password != confirm_password:
            msg = "Passwords do not match!"
        else:
            cursor.execute("INSERT INTO REGISTER (username, email, password) VALUES (?,?,?)", (username, email, password))
            get_db().commit()
            return redirect(url_for('login'))
    return render_template('register.html', msg=msg)

@app.route('/model.html', methods=['GET', 'POST'])
def model():
    background_image = "/static/image5.jpg"
    loader_visible = False
    if request.method == 'POST':
        selected_file = request.files['audio_file']
        file_name = selected_file.filename

        file_stream = selected_file.stream
        file_stream.seek(0)
        try:
            X, sample_rate = librosa.load(selected_file, res_type='kaiser_fast')
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=num_mfcc).T, axis=0)
            mel_spectrogram = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=num_mels).T, axis=0)
            chroma_features = np.mean(librosa.feature.chroma_stft(y=X, sr=sample_rate, n_chroma=num_chroma).T, axis=0)
            zcr = np.mean(librosa.feature.zero_crossing_rate(y=X).T, axis=0)
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=X, sr=sample_rate).T, axis=0)
            flatness = np.mean(librosa.feature.spectral_flatness(y=X).T, axis=0)
            
            features = np.concatenate((mfccs, mel_spectrogram, chroma_features, zcr, spectral_centroid, flatness))
            distances = np.linalg.norm(dataset_audio.iloc[:, :-1] - features, axis=1)
            closest_match_idx = np.argmin(distances)
            closest_match_label = dataset_audio.iloc[closest_match_idx, -1]
            total_distance = np.sum(distances)
            closest_match_prob = 1 - (distances[closest_match_idx] / total_distance)
            closest_match_prob_percentage = "{:.3f}".format(closest_match_prob * 100)

            file_label = f"File: {file_name}"
            result_label = f"Result: {'Fake' if closest_match_label == 'deepfake' else 'Real'} with {closest_match_prob_percentage}%"
            return render_template('model.html', file_label=file_label, result_label=result_label, background_image=background_image, loader_visible=loader_visible)
        except Exception as e:
            return render_template('model.html', file_label="Error processing audio", result_label=str(e), background_image=background_image)
    return render_template('model.html', background_image=background_image, loader_visible=loader_visible)

@app.route('/predict_news', methods=['GET', 'POST'])
def predict_news():
    if request.method == 'POST':
        news_text = request.form.get('news_text', '')
        predictions = classify_news(news_text)
        return render_template('predicttext.html', news_text=news_text, predictions=predictions)
    return render_template('predicttext.html')

@app.route('/image_upload', methods=['GET', 'POST'])
def image_upload():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            return render_template("image.html", error="No file selected.")
        file_path = save_file(file)
        result = classify_and_compare(file_path)
        return render_template("image.html", result=result)
    return render_template("image.html")

@app.route('/video_upload', methods=['GET', 'POST'])
def video_upload():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            return render_template("video.html", error="No file selected.")
        file_path = save_file(file)
        result = "Video processing logic not implemented yet."
        return render_template("video.html", result=result)
    return render_template("video.html")

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

@app.route("/download-pdf")
def download_pdf():
    pdf_output_path = "static/result.pdf"
    return send_file(pdf_output_path, as_attachment=True)

@app.route('/routes')
def list_routes():
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            "endpoint": rule.endpoint,
            "methods": list(rule.methods),
            "url": rule.rule
        })
    return {"routes": routes}

if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)