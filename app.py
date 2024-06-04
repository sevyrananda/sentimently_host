from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import mysql.connector
import os, re, csv, string
import pandas as pd
import numpy as np
import joblib
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import time
from sklearn.metrics import confusion_matrix
from flask import session
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Bidirectional
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
from sklearn.metrics import accuracy_score, classification_report
from textblob import TextBlob
from googletrans import Translator
from datetime import datetime
import shutil
import matplotlib.pyplot as plt
import seaborn as sns

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)
app.secret_key = 'sevyrav3921034'  # Ganti 'your_secret_key' dengan string acak yang aman
login_manager = LoginManager(app)
login_manager.login_view = 'login'  # Tentukan halaman login

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Konfigurasi koneksi database
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'login-flask'
}

# Membuat koneksi
mysql = mysql.connector.connect(**db_config)

# Kelas User
class User(UserMixin):
    def __init__(self, user_id):
        self.id = user_id

# Load user callback
@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

# Fungsi autentikasi
def authenticate(username, password):
    cur = mysql.cursor(dictionary=True)
    cur.execute("SELECT id, username, password FROM users WHERE username = %s AND password = %s", (username, password))
    user = cur.fetchone()
    cur.close()
    if user:
        return User(user['id'])
    return None

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = authenticate(username, password)
        if user:
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            error = 'Invalid username or password. Please try again.'
            return render_template('login.html', error=error)
    return render_template('login.html')

# GUEST
@app.route('/guest')
def guest():
    cur = mysql.cursor(dictionary=True)
    cur.execute("SELECT * FROM users")
    users = cur.fetchall()
    cur.close()
    return render_template('guest.html', users=users)

# ADMIN
@app.route('/dashboard')
@login_required
def dashboard():
    cur = mysql.cursor(dictionary=True)
    cur.execute("SELECT * FROM users")
    users = cur.fetchall()

    cur.execute("SELECT * FROM processed_datas")
    processed_data = cur.fetchall()

    cur.execute("SELECT * FROM hasil_train ORDER BY id DESC LIMIT 1")
    hasil_train = cur.fetchone()


    cur.close()
    return render_template('dashboard.html', users=users, processed_data=processed_data, hasil_train=hasil_train)


@app.route('/scraping')
@login_required
def scraping():
    cur = mysql.cursor(dictionary=True)
    cur.execute("SELECT * FROM users")
    users = cur.fetchall()

    cur.execute("SELECT * FROM processed_datas")
    processed_data = cur.fetchall()

    cur.close()
    return render_template('scraping.html', users=users, processed_data=processed_data)

@app.route('/save_data', methods=['POST'])
@login_required
def save_data():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        uploaded_file.save(file_path)
        try:
            hasilCSV(file_path)
            flash('Data berhasil ditambahkan!', 'success')
            return jsonify(success=True), 200  
        except Exception as e:
            flash(f'Gagal menyimpan data: {str(e)}', 'error')
            return jsonify(success=False, error=str(e)), 500  
    else:
        flash('Tidak ada file yang diupload', 'error')
        return jsonify(success=False, error='Tidak ada file yang diupload'), 400  


def hasilCSV(filePath):
    col_names = ['tgl', 'user', 'tweet', 'clean', 'casefold', 'tokenize', 'stopword', 'stemming', 'sentimen']
    csvData = pd.read_csv(filePath, names=col_names, header=None)
    for i, row in csvData.iterrows():
        sql = "INSERT INTO processed_datas (tgl, user, tweet, clean, casefold, tokenize, stopword, stemming, sentimen) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
        value = (row['tgl'], row['user'], row['tweet'], row['clean'], row['casefold'], row['tokenize'], row['stopword'], row['stemming'], row['sentimen'])
        cur = mysql.cursor()
        cur.execute(sql, value)
        mysql.commit()


  
@app.route('/user')
@login_required
def user():
    cur = mysql.cursor(dictionary=True)
    cur.execute("SELECT * FROM users")
    users = cur.fetchall()
    cur.close()
    return render_template('user.html', users=users)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/dataset')
@login_required
def dataset():
    cur = mysql.cursor(dictionary=True)
    cur.execute("SELECT * FROM dataset")
    dataset = cur.fetchall()
    cur.close()
    return render_template('dataset.html', dataset=dataset)

@app.route('/dataset', methods=['POST'])
def uploadFiles():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        uploaded_file.save(file_path)
        parseCSV(file_path)
        flash('Data successfully added!', 'success')  # Flash success message
        # Set a cookie to indicate that file has been processed
        response = redirect(url_for("dataset"))
        response.set_cookie('file_processed', 'true')
        return response
    return redirect(url_for("dataset"))

def parseCSV(filePath):
    col_names = ['full_text', 'sentiment']
    csvData = pd.read_csv(filePath, names=col_names, header=None)
    for i, row in csvData.iterrows():
        sql = "INSERT INTO dataset (full_text, sentiment) VALUES (%s, %s)"
        value = (row['full_text'], row['sentiment'])
        cur = mysql.cursor()
        cur.execute(sql, value)
        mysql.commit()

@app.route('/train', methods=['POST'])
# Fungsi untuk mendapatkan data dari tabel hasil_train
def train():
        # Ambil dataset dari tabel dataset
        cur = mysql.cursor(dictionary=True)
        cur.execute("SELECT * FROM dataset")
        dataset = cur.fetchall()
        cur.close()

        # Ubah menjadi DataFrame
        df = pd.DataFrame(dataset)

        # Menghapus baris dengan nilai NaN dalam kolom 'sentiment'
        df.dropna(subset=['sentiment'], inplace=True)

        # Preprocess data
        df['clean_tweet'] = df['full_text'].apply(preprocess)
        X_train, X_test, y_train, y_test = train_test_split(df['clean_tweet'], df['sentiment'], test_size=0.2, random_state=42)

        # Konversi label menjadi numerik
        sentiment_mapping = {'negatif': 0, 'netral': 1, 'positif': 2}
        y_train = y_train.map(sentiment_mapping)
        y_test = y_test.map(sentiment_mapping)

        # Vectorisasi untuk Naive Bayes
        vectorizer_nb = TfidfVectorizer()
        X_train_tfidf_nb = vectorizer_nb.fit_transform(X_train)
        X_test_tfidf_nb = vectorizer_nb.transform(X_test)

        # Latih dan Evaluasi Naive Bayes
        start_time_nb = time.time()
        naive_bayes = MultinomialNB()
        naive_bayes.fit(X_train_tfidf_nb, y_train)
        end_time_nb = time.time()
        nb_pred = naive_bayes.predict(X_test_tfidf_nb)
        nb_accuracy = accuracy_score(y_test, nb_pred)
        processing_time_nb = end_time_nb - start_time_nb

        # Vectorisasi dan Padding untuk LSTM
        tokenizer = Tokenizer(num_words=5000, split=' ')
        tokenizer.fit_on_texts(df['clean_tweet'].values)
        X_train_seq = tokenizer.texts_to_sequences(X_train)
        X_test_seq = tokenizer.texts_to_sequences(X_test)
        X_train_pad = pad_sequences(X_train_seq)
        X_test_pad = pad_sequences(X_test_seq, maxlen=X_train_pad.shape[1])

        # Bangun model LSTM
        model = Sequential()
        model.add(Embedding(5000, 128, input_length=X_train_pad.shape[1]))
        model.add(SpatialDropout1D(0.2))
        model.add(Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2)))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Latih model LSTM
        start_time_lstm = time.time()
        model.fit(X_train_pad, y_train, epochs=10, batch_size=64, validation_data=(X_test_pad, y_test))
        end_time_lstm = time.time()

        # Evaluasi model LSTM
        lstm_pred = model.predict(X_test_pad, batch_size=64)
        lstm_pred_classes = lstm_pred.argmax(axis=1)
        lstm_accuracy = accuracy_score(y_test, lstm_pred_classes)
        processing_time_lstm = end_time_lstm - start_time_lstm

        # Simpan model Naive Bayes ke dalam folder model
        joblib.dump(naive_bayes, 'model/naive_bayes_model.pkl')

        # Simpan vectorizer Naive Bayes ke dalam folder model
        joblib.dump(vectorizer_nb, 'model/tfidf_vectorizer.pkl')

        # Simpan model LSTM ke dalam folder model
        model_save_path = 'model/lstm_model.h5'
        model.save(model_save_path)

        # Simpan tokenizer LSTM ke dalam folder model
        tokenizer_save_path = 'model/tokenizer.pkl'
        with open(tokenizer_save_path, 'wb') as f:
            pickle.dump(tokenizer, f)

        # Tentukan model terbaik
        if nb_accuracy > lstm_accuracy:
            best_model = 'Naive Bayes'
        else:
            best_model = 'LSTM'

        # Simpan hasil pelatihan ke dalam tabel hasil_train
        cur = mysql.cursor()
        cur.execute("INSERT INTO hasil_train (best_model, acc_nb, processtime_nb, acc_lstm, processtime_lstm) VALUES (%s, %s, %s, %s, %s)",
                    (best_model, nb_accuracy, processing_time_nb, lstm_accuracy, processing_time_lstm))
        mysql.commit()

        # Simpan hasil preprocessing ke dalam tabel hasil_preprocessing
        cur = mysql.cursor()
        for index, row in df.iterrows():
            cur.execute("INSERT INTO hasil_preprocessing (dataset_id, clean_text) VALUES (%s, %s)",
                        (row['id'], row['clean_tweet']))
            mysql.commit()
        cur.close()

        # Flash message untuk memberi tahu pengguna bahwa proses pelatihan berhasil
        flash('Data Hasil Train Berhasil Ditambahkan ke Database', 'success')

        # Redirect ke halaman dashboard setelah pelatihan selesai
        return redirect('/dashboard')
   
    

@app.route('/tes')
def tes():
    # Fetch the dataset
    cur = mysql.cursor(dictionary=True)
    cur.execute("SELECT * FROM dataset")
    dataset = cur.fetchall()
    cur.close()

    # Convert to DataFrame
    df = pd.DataFrame(dataset)

    # Menghapus baris dengan nilai NaN dalam kolom 'sentiment'
    df.dropna(subset=['sentiment'], inplace=True)

    # Preprocess data
    df['clean_tweet'] = df['full_text'].apply(preprocess)
    X_train, X_test, y_train, y_test = train_test_split(df['clean_tweet'], df['sentiment'], test_size=0.2, random_state=42)

    # Convert labels to numeric
    sentiment_mapping = {'negatif': 0, 'netral': 1, 'positif': 2}
    y_train = y_train.map(sentiment_mapping)
    y_test = y_test.map(sentiment_mapping)

    # Vectorization for Naive Bayes
    vectorizer_nb = TfidfVectorizer()
    X_train_tfidf_nb = vectorizer_nb.fit_transform(X_train)
    X_test_tfidf_nb = vectorizer_nb.transform(X_test)

    # Train and Evaluate Naive Bayes
    start_time_nb = time.time()
    naive_bayes = MultinomialNB()
    naive_bayes.fit(X_train_tfidf_nb, y_train)
    end_time_nb = time.time()
    nb_pred = naive_bayes.predict(X_test_tfidf_nb)
    nb_accuracy = accuracy_score(y_test, nb_pred)

    # Vectorization and Padding for LSTM
    tokenizer = Tokenizer(num_words=5000, split=' ')
    tokenizer.fit_on_texts(df['clean_tweet'].values)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_train_pad = pad_sequences(X_train_seq)
    X_test_pad = pad_sequences(X_test_seq, maxlen=X_train_pad.shape[1])

    # Convert labels to numpy array
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Build LSTM model
    model = Sequential()
    model.add(Embedding(5000, 128, input_length=X_train_pad.shape[1]))
    model.add(SpatialDropout1D(0.2))
    model.add(Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train LSTM model
    start_time_lstm = time.time()
    model.fit(X_train_pad, y_train, epochs=10, batch_size=64, validation_data=(X_test_pad, y_test))
    end_time_lstm = time.time()

    # Evaluate LSTM model
    lstm_pred = model.predict(X_test_pad, batch_size=64)
    lstm_pred_classes = lstm_pred.argmax(axis=1)
    lstm_accuracy = accuracy_score(y_test, lstm_pred_classes)

    # Determine the best model
    best_model = 'Naive Bayes' if nb_accuracy > lstm_accuracy else 'LSTM'
    if best_model == 'Naive Bayes':
        model_path = os.path.join('model', 'naive_bayes.pkl')
        joblib.dump(naive_bayes, model_path)
    else:
        model_path = os.path.join('model', 'lstm.h5')
        model.save(model_path)

    # Menyimpan hasil pelatihan ke dalam file CSV
    with open('static/files/tes.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Model', 'Accuracy', 'ProcessingTime'])
        writer.writerow(['Naive Bayes', nb_accuracy, str(end_time_nb - start_time_nb)])
        writer.writerow(['LSTM', lstm_accuracy, str(end_time_lstm - start_time_lstm)])

    training_results = []
    with open('static/files/tes.csv', mode='r') as file:
        reader = csv.reader(file)
        next(reader)  
        for row in reader:
            model_name, accuracy, processing_time = row
            accuracy = round(float(accuracy), 2) * 100  
            processing_time = round(float(processing_time), 2)  
            training_results.append({
                "Model": model_name,
                "Accuracy": accuracy,
                "ProcessingTime": processing_time
            })
    return render_template('tes.html', training_results=training_results)


# CRUD USER
@app.route('/add_user', methods=['POST'])
def add_user():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        cur = mysql.cursor()
        cur.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, password))
        mysql.commit()
        cur.close()
        flash('Data User Berhasil Ditambahkan', 'success')
        return redirect('/user')

@app.route('/delete_user/<int:user_id>')
def delete_user(user_id):
    cur = mysql.cursor()
    cur.execute("DELETE FROM users WHERE id = %s", (user_id,))
    mysql.commit()
    cur.close()
    flash('Data User Berhasil Dihapus', 'success')
    return redirect('/user')

@app.route('/edit_user/<int:user_id>', methods=['POST'])
def edit_user(user_id):
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        cur = mysql.cursor()
        cur.execute("UPDATE users SET username = %s, password = %s WHERE id = %s", (username, password, user_id))
        mysql.commit()
        cur.close()
        flash('Data User Berhasil Diupdate', 'success')
        return redirect('/user')
    
# CRUD SCRAPE
@app.route('/delete_text/<int:data_id>')
def delete_text(data_id):
    cur = mysql.cursor()
    cur.execute("DELETE FROM processed_datas WHERE id = %s", (data_id,))
    mysql.commit()
    cur.close()
    flash('Data Scraping Berhasil Dihapus', 'success')
    return redirect('/scraping')

@app.route('/edit_text/<int:data_id>', methods=['POST'])
def edit_text(data_id):
    if request.method == 'POST':
        text = request.form['text']
        sentiment = request.form['sentiment']
        cur = mysql.cursor()
        cur.execute("UPDATE processed_datas SET text = %s, sentiment = %s WHERE id = %s", (text, sentiment, data_id))
        mysql.commit()
        cur.close()
        flash('Data Scraping Berhasil Diupdate', 'success')
        return redirect('/scraping')
    
    
# CRUD DATASET
@app.route('/add_dataset', methods=['POST'])
def add_dataset():
    if request.method == 'POST':
        full_text = request.form['full_text']
        sentiment = request.form['sentiment']
        cur = mysql.cursor()
        cur.execute("INSERT INTO dataset (full_text, sentiment) VALUES (%s, %s)", (full_text, sentiment))
        mysql.commit()
        cur.close()
        flash('Dataset Berhasil Ditambahkan', 'success')
        return redirect('/dataset')
    
@app.route('/edit_dataset/<int:dataset_id>', methods=['POST'])
def edit_dataset(dataset_id):
    if request.method == 'POST':
        full_text = request.form['full_text']
        sentiment = request.form['sentiment']
        cur = mysql.cursor()
        cur.execute("UPDATE dataset SET full_text = %s, sentiment = %s WHERE id = %s", (full_text, sentiment, dataset_id))
        mysql.commit()
        cur.close()
        flash('Dataset Berhasil Diupdate', 'success')
        return redirect('/dataset')
    
    
@app.route('/delete_dataset/<int:dataset_id>', methods=['POST'])
def delete_dataset(dataset_id):
    cur = mysql.cursor()
    cur.execute("DELETE FROM dataset WHERE id = %s", (dataset_id,))
    mysql.commit()
    cur.close()
    flash('Dataset deleted successfully', 'success')
    return redirect('/dataset')

# Load saved models and necessary objects
model_type = "lstm"  # or "naive_bayes"
lstm_model_path = os.path.join("model", "lstm_model.h5")
naive_bayes_model_path = os.path.join("model", "naive_bayes_model.pkl")
vectorizer_path = os.path.join("model", "tfidf_vectorizer.pkl")
tokenizer_path = os.path.join("model", "tokenizer.pkl")

if model_type == "naive_bayes":
    best_model = joblib.load(naive_bayes_model_path)
    vectorizer = joblib.load(vectorizer_path)
else:
    best_model = load_model(lstm_model_path)
    tokenizer_lstm = joblib.load(tokenizer_path)
    max_words = 200

# Initialize translator
translator = Translator()

# Function to preprocess text
# def clean_text(s):
#     if not isinstance(s, str):
#         return ''
#     s = re.sub(r'http\S+', '', s)
#     s = re.sub('(RT|via)((?:\\b\\W*@\\w+)+)', ' ', s)
#     s = re.sub(r'@\S+', '', s)
#     s = re.sub('&amp', ' ', s)
#     s = re.sub(r'\d+', '', s) # Remove numbers
#     s = re.sub(r'[^\w\s]', '', s) # Remove punctuation
#     s = re.sub(r'[^a-zA-Z0-9\s]', '', s) # Remove special characters
#     return s.lower()

# def remove_stopwords(text):
#     if not isinstance(text, str):
#         return ''
#     factory = StopWordRemoverFactory()
#     stopword_sastrawi = factory.get_stop_words()
#     stop_words = set(stopwords.words('indonesian')) | set(stopword_sastrawi)
#     words = text.split()
#     filtered_words = [word for word in words if word.lower() not in stop_words]
#     return ' '.join(filtered_words)

# def stem_text(text):
#     if not isinstance(text, str):
#         return ''
#     factory = StemmerFactory()
#     stemmer = factory.create_stemmer()
#     return stemmer.stem(text)


def preprocess(text):
    if not isinstance(text, str):
        return ''
    
    # Menghapus URL
    text = re.sub(r'http\S+', '', text)
    text = re.sub('(RT|via)((?:\\b\\W*@\\w+)+)', ' ', text)
    text = re.sub(r'@\S+', '', text)
    text = re.sub('&amp', ' ', text)
    
    # Menghapus angka
    text = re.sub(r'\d+', '', text)
    
    # Menghapus simbol
    text = re.sub(r'[^\w\s]', '', text)
    
    # Menghapus karakter khusus
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Lowercase
    text = text.lower()

    # Normalisasi kata-kata tertentu
    text = re.sub(r'\bMcD\b|\bMCD\b|\bMcDonalds\b|\bMcDonald\'s\b', 'mcd', text)
    text = re.sub(r'\bKFC\b', 'kfc', text)
    text = re.sub(r'\bstarbak\b|\bStarbucks\b|\bstarbuck\b|\bsbuck\b|\bsbux\b', 'starbucks', text)
    
    # Tokenisasi
    tokens = word_tokenize(text)
    
    # Penghapusan stopword
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    
    # Menggabungkan kembali token-token yang sudah di-stem menjadi teks baru
    processed_text = ' '.join(stemmed_tokens)
    
    return processed_text

if best_model == "Naive Bayes":
    nb_model_path = os.path.abspath("model/naive_bayes_model.pkl")
    nb_vectorizer_path = os.path.abspath("model/tfidf_vectorizer.pkl")
    naive_bayes = joblib.load(nb_model_path)
    vectorizer_nb = joblib.load(nb_vectorizer_path)
    model_type = "naive_bayes"
else:
    lstm_model_path = os.path.abspath("model/lstm_model.h5")
    lstm_tokenizer_path = os.path.abspath("model/tokenizer.pkl")
    lstm_model = load_model(lstm_model_path)
    with open(lstm_tokenizer_path, 'rb') as f:
        tokenizer_lstm = pickle.load(f)
    model_type = "lstm"

max_words = 100  # This should be set to the same value used during training
# Define preprocessing and prediction functions
def preprocess_input_sentence(sentence):
    processed_sentence = tokenizer_lstm.texts_to_sequences([sentence])
    processed_sentence = pad_sequences(processed_sentence, maxlen=max_words)
    return processed_sentence

def preprocess_input_sentence_nb(sentence):
    cleaned_sentence = preprocess(sentence)  # Implement your text cleaning here
    return cleaned_sentence

def predict_sentiment_best_model(input_sentence, model_type):
    if model_type == "naive_bayes":
        input_tfidf = vectorizer_nb.transform([input_sentence])
        predicted_sentiment = naive_bayes.predict(input_tfidf)[0]
        sentiment_scores = None  # Naive Bayes might not have probability scores readily available
        return predicted_sentiment, sentiment_scores
    elif model_type == "lstm":
        processed_input = preprocess_input_sentence(input_sentence)
        pred_prob = lstm_model.predict(processed_input)[0]
        sentiment_scores = {
            'Negative': pred_prob[0],
            'Neutral': pred_prob[1],
            'Positive': pred_prob[2]
        }
        max_sentiment = max(sentiment_scores, key=sentiment_scores.get)
        return max_sentiment, sentiment_scores

# Flask routes
@app.route('/analyze', methods=['POST'])
def analyze():
    input_sentence = request.form['text']
    # Clean the input text
    cleaned_input = preprocess(input_sentence)
    
    if model_type == "naive_bayes":
        processed_input = preprocess_input_sentence_nb(input_sentence)
        predicted_sentiment, sentiment_scores = predict_sentiment_best_model(processed_input, model_type)
    elif model_type == "lstm":
        processed_input = preprocess_input_sentence(input_sentence)
        predicted_sentiment, sentiment_scores = predict_sentiment_best_model(input_sentence, model_type)
    
    return render_template('guest.html', input_text=input_sentence, sentiment=predicted_sentiment, scores=sentiment_scores, clean_text=cleaned_input)


# SCRAPING
scraped_files = []  # List untuk menyimpan nama file CSV yang sudah dilakukan scraping

@app.route('/scrape', methods=['POST'])
def scrape():
    since_date = request.form['since_date']
    until_date = request.form['until_date']
    keyword = request.form['keyword']
    return render_template('dataset.html', since_date=since_date, until_date=until_date)

@app.route('/do_scrape', methods=['POST'])
def do_scrape():
    try:
        auth_token = request.json['auth_token']
        since_date = datetime.strptime(request.json['since_date'], '%Y-%m-%d')
        until_date = datetime.strptime(request.json['until_date'], '%Y-%m-%d')
        limit = request.json['limit']
        keyword = request.json['keyword']

        # Jalankan skrip untuk scraping di sini
        data = 'data_boikot.csv'
        search_keyword = f'{keyword} lang:id until:{until_date.strftime("%Y-%m-%d")} since:{since_date.strftime("%Y-%m-%d")}'

        os.system(f'npx tweet-harvest@latest -o "{data}" -s "{search_keyword}" -l {limit} --token "{auth_token}"')

        # Salin file dari tweets-data/data_boikot.csv ke static/files/data_boikot.csv
        source_file = 'tweets-data/data_boikot.csv'
        destination_file = 'static/files/Data Scraping.csv'
        shutil.copyfile(source_file, destination_file)

        # Setelah scraping selesai, lakukan preprocessing dan labeling
        hasil_preprocessing, hasil_labeling = preprocessing_and_labeling_twitter()

        # Menyusun data untuk dikirim ke frontend
        hasil_data = []
        for i, row in enumerate(hasil_preprocessing):
            hasil_data.append({
                "tgl": row[0],
                "user": row[1],
                "tweet": row[2],
                "clean": row[3],
                "casefold": row[4],
                "tokenize": row[5],
                "stopword": row[6],
                "stemming": row[7],
                "sentimen": row[8]
            })

        return jsonify(message="Scraping berhasil!", data=hasil_data)
    except Exception as e:
        return jsonify(error=str(e))


hasil_preprocessing = []
hasil_labeling = []
def preprocessing_and_labeling_twitter():
    # Membuat File CSV untuk preprocessing dan labeling
    file_combined = open('static/files/Data Preprocessing_Labeling.csv', 'w', newline='', encoding='utf-8')
    writer_combined = csv.writer(file_combined)

    hasil_preprocessing.clear()
    hasil_labeling.clear()
    translator = Translator()

    with open("static/files/Data Scraping.csv", "r", encoding='utf-8') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        next(readCSV)
        
        for row in readCSV:
            text_to_process = row[3]

            # Preprocessing
            clean = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text_to_process).split())
            clean = re.sub("\d+", "", clean)
            clean = re.sub(r"\b[a-zA-Z]\b", "", clean)
            clean = re.sub('\s+', ' ', clean)
            clean = clean.translate(clean.maketrans("", "", string.punctuation))
            casefold = clean.casefold()
            tokenizing = nltk.tokenize.word_tokenize(casefold)
            stop_factory = StopWordRemoverFactory().get_stop_words()
            more_stop_word = ['&amp', 'ad', 'ada', 'ae', 'ah', 'aja', 'ajar', 'ajar', 'amp', 'apa', 'aya', 'bab', 'bajo', 'bar', 'bbrp', 'beda', 'begini', 'bgmn', 'bgt', 'bhw', 'biar', 'bikin', 'bilang', 'bkh', 'bkn', 'bln', 'bnyk', 'brt', 'buah', 'cc', 'cc', 'ckp', 'com', 'cuy', 'd', 'dab', 'dah', 'dan', 'dg', 'dgn', 'di', 'dih', 'dlm', 'dm', 'dpo', 'dr', 'dr', 'dri', 'duga', 'duh', 'enth', 'er', 'et', 'ga', 'gak', 'gal', 'gin', 'gitu', 'gk', 'gmn', 'gs', 'gt', 'gue', 'gw', 'hah', 'hallo', 'halo', 'hehe', 'hello', 'hha', 'hrs', 'https', 'ia', 'iii', 'in', 'ini', 'iw', 'jadi', 'jadi', 'jangn', 'jd', 'jg', 'jgn', 'jls', 'kak', 'kali', 'kalo', 'kan', 'kch', 'ke', 'kena', 'ket', 'kl', 'kll', 'klo', 'km', 'kmrn', 'knp', 'kok', 'kpd', 'krn', 'kui', 'lagi', 'lah', 'lahh', 'lalu', 'lbh', 'lewat', 'loh', 'lu', 'mah', 'mau', 'min', 'mlkukan', 'mls', 'mnw', 'mrk', 'n', 'nan', 'ni', 'nih', 'no', 'nti', 'ntt', 'ny', 'nya', 'nyg', 'oleh', 'ono', 'ooooo', 'op', 'org', 'pen', 'pk', 'pun', 'qq', 'rd', 'rt', 'sama', 'sbg', 'sdh', 'sdrhn', 'segera', 'sgt', 'si', 'si', 'sih', 'sj', 'so', 'sy', 't', 'tak', 'tak', 'tara', 'tau', 'td', 'tdk', 'tdk', 'thd', 'thd', 'thn', 'tindkn', 'tkt', 'tp', 'tsb', 'ttg', 'ttp', 'tuh', 'tv', 'u', 'upa', 'utk', 'uyu', 'viral', 'vm', 'wae', 'wah', 'wb', 'wes', 'wk', 'wkwk', 'wkwkwk', 'wn', 'woiii', 'xxxx', 'ya', 'yaa', 'yah', 'ybs', 'ye', 'yg', 'ykm']
            data = stop_factory + more_stop_word
            dictionary = ArrayDictionary(data)
            str = StopWordRemover(dictionary)
            stop_wr = nltk.tokenize.word_tokenize(str.remove(casefold))
            kalimat = ' '.join(stop_wr)
            factory = StemmerFactory()
            stemmer = factory.create_stemmer()
            stemming = stemmer.stem(kalimat)
            
            # Labeling
            try:
                value = translator.translate(stemming, dest='en')
                terjemahan = value.text
                data_label = TextBlob(terjemahan)
                
                # Kata kunci untuk sentimen positif
                kata_positif = ["berhenti mengonsumsi", "berhenti membeli", "tidak akan lagi membeli", "semangat boikot", "beralih produk lokal"]  
                
                # Kata kunci untuk sentimen negatif
                kata_negatif = ["membeli", "mengonsumsi", "menyukai", "memakai"]  
                
                sentiment = "Netral"  
                
                # Periksa apakah ada kata kunci positif atau negatif dalam terjemahan
                if any(kata in terjemahan for kata in kata_positif):
                    sentiment = "Positif"
                elif any(kata in terjemahan for kata in kata_negatif):
                    sentiment = "Negatif"
                elif data_label.sentiment.polarity > 0.0:
                    sentiment = "Positif"
                elif data_label.sentiment.polarity == 0.0:
                    sentiment = "Netral"
                else:
                    sentiment = "Negatif"

                # Menulis baris ke file CSV
                row_combined = [row[1], row[14], row[3], clean, casefold, tokenizing, stop_wr, stemming, sentiment]
                writer_combined.writerow(row_combined)
                
                # Tambahkan hasil preprocessing ke variabel hasil_preprocessing
                hasil_preprocessing.append(row_combined)
            except Exception as e:
                print(f"Error: {e}")

    file_combined.close()

    flash('Preprocessing dan Labeling Berhasil', 'preprocessing_labeling_data')
    return hasil_preprocessing, hasil_labeling

if __name__ == '__main__':
    app.run(debug=True)