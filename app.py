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
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
from sklearn.metrics import accuracy_score, classification_report
from textblob import TextBlob
from googletrans import Translator
from datetime import datetime, timedelta
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
            return redirect(url_for('scraping'))
        else:
            error = 'Invalid username or password. Please try again.'
            return render_template('login.html', error=error)
    return render_template('login.html')

# GUEST
#Fungsi untuk mendapatkan data sentimen dari dataset
def get_sentiment_data(products_to_analyze):
    negative_products = {}
    positive_products = {}
    neutral_products = {}

    cur = mysql.cursor(dictionary=True)
    query = "SELECT tweet, sentimen FROM processed_datas"
    cur.execute(query)

    for row in cur.fetchall():
        tweet = row['tweet']
        sentimen = row['sentimen']
        
        for product in products_to_analyze:
            product_name = product['name']

            if product_name.lower() in tweet.lower():
                if sentimen.lower() == "negatif":
                    if product_name in negative_products:
                        negative_products[product_name] += 1
                    else:
                        negative_products[product_name] = 1
                elif sentimen.lower() == "positif":
                    if product_name in positive_products:
                        positive_products[product_name] += 1
                    else:
                        positive_products[product_name] = 1
                else:
                    if product_name in neutral_products:
                        neutral_products[product_name] += 1
                    else:
                        neutral_products[product_name] = 1
    
    cur.close()
    return negative_products, positive_products, neutral_products

@app.route('/guest')
def guest():
    products_to_analyze = [
        {"name": "KFC", "image": "/static/kfc.png"},
        {"name": "McD", "image": "/static/mcd.png"},
        {"name": "Starbucks", "image": "/static/sbux.png"},
        {"name": "Burger King", "image": "/static/bk.png"},
        {"name": "Aqua", "image": "/static/aqua.png"},
        {"name": "Nestle", "image": "/static/nestle.png"},
        {"name": "Pizza Hut", "image": "/static/ph.png"},
        {"name": "Oreo", "image": "/static/oreo.png"},
        {"name": "Unilever", "image": "/static/unv.png"},
    ]

    negative_products, positive_products, neutral_products = get_sentiment_data(products_to_analyze)

    products_sentiment = {}
    for product in products_to_analyze:
        product_name = product['name']
        positive_count = positive_products.get(product_name, 0)
        negative_count = negative_products.get(product_name, 0)
        neutral_count = neutral_products.get(product_name, 0)
        
        product_sentiment = {
            "positif": {
                "count": positive_count
            },
            "negatif": {
                "count": negative_count
            },
            "netral": {
                "count": neutral_count
            }
        }
        
        products_sentiment[product_name] = product_sentiment

    sorted_products_sentiment = dict(sorted(products_sentiment.items(), 
                                            key=lambda item: (item[1]['positif']['count'], item[1]['netral']['count'], item[1]['negatif']['count']), 
                                            reverse=True))

    sentiment_counts = {
        "positif": sum(positive_products.values()),
        "negatif": sum(negative_products.values()),
        "netral": sum(neutral_products.values())
    }
    overall_majority_sentiment = max(sentiment_counts, key=sentiment_counts.get)

    cur = mysql.cursor(dictionary=True)
    cur.execute("SELECT * FROM users")
    users = cur.fetchall()
    cur.close()

    return render_template('guest.html', 
                           products_sentiment=sorted_products_sentiment,
                           overall_majority_sentiment=overall_majority_sentiment,
                           users=users)


# ADMIN
def get_visdata():
    negative_products = []
    positive_products = []
    neutral_products = []

    cur = mysql.cursor(dictionary=True)
    query = "SELECT full_text, sentiment FROM dataset"
    cur.execute(query)

    for row in cur.fetchall():
        full_text = row['full_text']
        sentiment = row['sentiment']
        if sentiment.lower() == "negatif":
            negative_products.append({"tweet": full_text, "sentiment": "Negatif"})
        elif sentiment.lower() == "positif":
            positive_products.append({"tweet": full_text, "sentiment": "Positif"})
        else:
            neutral_products.append({"tweet": full_text, "sentiment": "Netral"})
    
    cur.close()
    
    # Hitung jumlah tweet untuk setiap kategori sentimen
    total_positive = len(positive_products)
    total_neutral = len(neutral_products)
    total_negative = len(negative_products)

    return negative_products, positive_products, neutral_products, total_positive, total_neutral, total_negative

@app.route('/dataset')
@login_required
def dashboard():
    negative_products, positive_products, neutral_products, total_positive, total_neutral, total_negative = get_visdata()
    
    cur = mysql.cursor(dictionary=True)
    cur.execute("SELECT * FROM users")
    users = cur.fetchall()

    cur.execute("SELECT * FROM processed_datas")
    processed_data = cur.fetchall()

    cur.execute("SELECT * FROM dataset")
    dataset = cur.fetchall()

    cur.execute("SELECT * FROM hasil_train ORDER BY id DESC LIMIT 1")
    hasil_train = cur.fetchone()

    cur.close()
    
    if hasil_train:
        # Convert accuracy to percentage
        acc_nb_percent = round(hasil_train['acc_nb'] * 100, 2)
        acc_lstm_percent = round(hasil_train['acc_lstm'] * 100, 2)
        
        # Update hasil_train with converted values
        hasil_train['acc_nb'] = acc_nb_percent
        hasil_train['acc_lstm'] = acc_lstm_percent

    # Pass all necessary data to the template
    return render_template('dataset.html',
                           dataset=dataset, 
                           users=users, 
                           processed_data=processed_data, 
                           hasil_train=hasil_train,
                           total_positive=total_positive,
                           total_neutral=total_neutral,
                           total_negative=total_negative,
                           chart_data={
                               "labels": ['Mendukung', 'Netral', 'Menolak'],
                               "data": [
                                   total_positive,
                                   total_neutral,
                                   total_negative
                               ]
                           })








@app.route('/scraping', methods=['GET'])
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

    cur.execute("SELECT * FROM hasil_train ORDER BY id DESC LIMIT 1")
    hasil_train = cur.fetchone()

    cur.close()
    return render_template('dataset.html', 
                           dataset=dataset, 
                           hasil_train=hasil_train)

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
def train():
    try:
        # Buat folder 'model' jika belum ada
        model_folder = 'model'
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        # Preprocess data
        df, X_train, X_test, y_train, y_test = preprocess_data()

        # Train models
        nb_accuracy, processing_time_nb, naive_bayes, vectorizer_nb = train_naive_bayes(X_train, X_test, y_train, y_test)
        lstm_accuracy, processing_time_lstm, model, tokenizer = train_lstm(df, X_train, X_test, y_train, y_test)

        # Determine best model
        best_model = evaluate_models(nb_accuracy, lstm_accuracy)

        # Save models
        save_models(model_folder, naive_bayes, vectorizer_nb, model, tokenizer)

        # Save training results
        save_training_results(df, best_model, nb_accuracy, processing_time_nb, lstm_accuracy, processing_time_lstm)

        # Flash message untuk memberi tahu pengguna bahwa proses pelatihan berhasil
        return jsonify({'message': 'Data Hasil Train Berhasil Ditambahkan ke Database', 'status': 'success', 'best_model': best_model})

    except Exception as e:
        print(f"Error during training: {e}")
        return jsonify({'message': f"Error saat pelatihan: {e}", 'status': 'danger'})

def preprocess_data():
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

    # Periksa nilai NaN di y_train dan y_test setelah mapping
    if y_train.isnull().any() or y_test.isnull().any():
        nan_indices_train = y_train[y_train.isnull()].index
        nan_indices_test = y_test[y_test.isnull()].index
        print("Nilai NaN ditemukan di y_train pada indeks:", nan_indices_train)
        print("Nilai NaN ditemukan di y_test pada indeks:", nan_indices_test)
        print("Nilai sentimen asli di y_train:", df.loc[nan_indices_train, 'sentiment'])
        print("Nilai sentimen asli di y_test:", df.loc[nan_indices_test, 'sentiment'])
        raise ValueError("y_train atau y_test mengandung nilai NaN setelah mapping sentimen")

    return df, X_train, X_test, y_train, y_test

def train_naive_bayes(X_train, X_test, y_train, y_test):
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

    return nb_accuracy, processing_time_nb, naive_bayes, vectorizer_nb

def train_lstm(df, X_train, X_test, y_train, y_test):
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

    return lstm_accuracy, processing_time_lstm, model, tokenizer

def evaluate_models(nb_accuracy, lstm_accuracy):
    # Tentukan model terbaik
    if nb_accuracy > lstm_accuracy:
        best_model = 'Naive Bayes'
    else:
        best_model = 'LSTM'

    print(f"Best Model: {best_model}")
    print(f"Akurasi Naive Bayes: {nb_accuracy}")
    print(f"Akurasi LSTM: {lstm_accuracy}")

    return best_model

def save_models(model_folder, naive_bayes, vectorizer_nb, model, tokenizer):
    try:
        # Simpan model Naive Bayes ke dalam folder model
        joblib.dump(naive_bayes, os.path.join(model_folder, 'naive_bayes_model.pkl'))

        # Simpan vectorizer Naive Bayes ke dalam folder model
        joblib.dump(vectorizer_nb, os.path.join(model_folder, 'tfidf_vectorizer.pkl'))

        # Simpan model LSTM ke dalam folder model
        model_save_path = os.path.join(model_folder, 'lstm_model.h5')
        model.save(model_save_path)

        # Simpan tokenizer LSTM ke dalam folder model
        tokenizer_save_path = os.path.join(model_folder, 'tokenizer.pkl')
        with open(tokenizer_save_path, 'wb') as f:
            pickle.dump(tokenizer, f)
    except Exception as e:
        print(f"Error saving model: {e}")
        raise Exception('Error menyimpan model')

def save_training_results(df, best_model, nb_accuracy, processing_time_nb, lstm_accuracy, processing_time_lstm):
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
    try:
        # Hapus records terkait di hasil_preprocessing
        cur.execute("DELETE FROM hasil_preprocessing WHERE dataset_id = %s", (dataset_id,))
        # Hapus record di dataset
        cur.execute("DELETE FROM dataset WHERE id = %s", (dataset_id,))
        mysql.commit()
        flash('Dataset deleted successfully', 'success')
    except mysql.connector.Error as err:
        mysql.rollback()
        flash(f'Error: {err}', 'danger')
    finally:
        cur.close()
    return redirect('/dataset')



def preprocess(clean_text):
    if not isinstance(clean_text, str):
        return ''

    # Menghapus tautan, mention, dan karakter khusus
    clean = ' '.join(re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", clean_text).split())
    
    # Menghapus angka
    clean = re.sub(r"\d+", "", clean)
    
    # Menghapus kata-kata tunggal (single characters)
    clean = re.sub(r"\b[a-zA-Z]\b", "", clean)
    
    # Menghapus spasi berlebih
    clean = re.sub(r'\s+', ' ', clean)
    
    # Menghapus tanda baca
    clean = clean.translate(clean.maketrans("", "", string.punctuation))
    
    # Mengubah teks menjadi lowercase
    casefold = clean.casefold()
    
    # Normalisasi kata kunci seperti McD, McDonalds, Starbucks, dll.
    casefold = re.sub(r'\bMcD\b|\bMCD\b|\bMcDonalds\b|\bMcDonald\'s\b', 'mcd', casefold)
    casefold = re.sub(r'\bKFC\b', 'kfc', casefold)
    casefold = re.sub(r'\bstarbak\b|\bStarbucks\b|\bstarbuck\b|\bsbuck\b|\bsbux\b', 'starbucks', casefold)

    # Tokenisasi
    tokens = nltk.tokenize.word_tokenize(casefold)

    # Penghapusan stopword (menggunakan Sastrawi untuk bahasa Indonesia)
    stop_factory = StopWordRemoverFactory().get_stop_words()
    dictionary = ArrayDictionary(stop_factory)
    stopword_remover = StopWordRemover(dictionary)
    stop_wr = nltk.tokenize.word_tokenize(stopword_remover.remove(casefold))
    kalimat = ' '.join(stop_wr)
    
    # Stemming (menggunakan Sastrawi untuk bahasa Indonesia)
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stemming = stemmer.stem(kalimat)
    
    return stemming



# Load saved models and necessary objects
best_model = "lstm"  # or "naive_bayes"
lstm_model_path = os.path.join("model", "lstm_model.h5")
naive_bayes_model_path = os.path.join("model", "naive_bayes_model.pkl")
vectorizer_path = os.path.join("model", "tfidf_vectorizer.pkl")
tokenizer_path = os.path.join("model", "tokenizer.pkl")

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
    max_words = 100  


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


scraped_files = []  # List to store scraped CSV filenames

# Mendapatkan tanggal hari ini
today = datetime.now()

# Mendapatkan tanggal kemarin
yesterday = today - timedelta(days=1)

# Format tanggal untuk pencarian (YYYY-MM-DD)
since_date = yesterday.strftime('%Y-%m-%d')
until_date = today.strftime('%Y-%m-%d')

def is_scraped_today(existing_data, target_date):
    """
    Check if the target date already exists in the existing data.
    """
    for row in existing_data:
        row_date = datetime.strptime(row['tgl'], '%a %b %d %H:%M:%S %z %Y')
        if row_date.date() == target_date.date():
            return True
    return False

@app.route('/do_scrape', methods=['POST'])
def do_scrape():
    try:
        cur = mysql.cursor(dictionary=True)

        # Periksa apakah ada data untuk tanggal kemarin
        query = "SELECT * FROM processed_datas"
        cur.execute(query)
        existing_data = cur.fetchall()

        # Periksa apakah sudah ada data untuk tanggal kemarin
        if is_scraped_today(existing_data, yesterday):
            # Jika ada data untuk tanggal kemarin, kembalikan data tersebut dan pesan bahwa scraping sudah dilakukan
            cur.close()
            return jsonify(message="Scraping sudah dilakukan kemarin. Data yang sudah di-scrape:", data=existing_data)
        else:
            auth_token = "0e12b16141a80c4510f95de2dcd5ef5b365b3fb3"  # Replace with actual auth token handling logic
            limit = 10  # Example limit, adjust as needed
            keyword = "boikot produk"  # Example keyword, adjust as needed

            # Run scraping script
            data = 'data_boikot.csv'
            search_keyword = f'{keyword} lang:id until:{until_date} since:{since_date}'
            os.system(f'npx tweet-harvest@latest -o "{data}" -s "{search_keyword}" -l {limit} --token "{auth_token}"')

            # Copy the scraped data to a static folder (example path)
            source_file = 'tweets-data/data_boikot.csv'
            destination_file = 'static/files/Data Scraping.csv'
            shutil.copyfile(source_file, destination_file)

            # After scraping, perform preprocessing and labeling
            hasil_preprocessing, hasil_labeling = preprocessing_and_labeling_twitter()

            # Panggil fungsi untuk menyimpan hasil ke database
            save_to_database(cur, hasil_preprocessing)

            # Tutup cursor dan koneksi database
            cur.close()

            # Prepare data for frontend
            hasil_data = []
            for row in hasil_preprocessing:
                hasil_data.append({
                    "tgl": row[0],
                    "user": row[1],
                    "tweet": row[2],
                    "clean": row[3],
                    "sentimen": row[4],
                    "casefold": row[5],
                    "tokenize": row[6],
                    "stopword": row[7],
                    "stemming": row[8]
                })

            accuracy = calculate_accuracy(hasil_preprocessing)  # Adjust based on your actual accuracy calculation

            return jsonify(message="Scraping berhasil!", data=hasil_data, accuracy=accuracy)
    except Exception as e:
        print(f"Error during scraping and processing: {e}")
        return jsonify(error=str(e))

def calculate_accuracy(data):
    # Example function to calculate accuracy
    # Adjust based on your actual accuracy calculation logic
    labels_true = [row[4] for row in data]
    labels_pred = [row[9] for row in data]  # Assuming 'row[9]' contains predicted labels
    accuracy = accuracy_score(labels_true, labels_pred)
    return accuracy * 100

hasil_preprocessing = []
hasil_labeling = []

def preprocessing_and_labeling_twitter():
    try:
        # Create CSV for preprocessing and labeling
        file_combined = open('static/files/Data Preprocessing Labeling.csv', 'w', newline='', encoding='utf-8')
        writer_combined = csv.writer(file_combined)

        hasil_preprocessing.clear()
        hasil_labeling.clear()
        translator = Translator()

        with open("static/files/Data Scraping.csv", "r", encoding='utf-8') as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            next(readCSV)
            
            for row in readCSV:
                if len(row) > 14:
                    text_to_process = row[3]
                    clean = ' '.join(re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text_to_process).split())
                    clean = re.sub(r"\d+", "", clean)
                    clean = re.sub(r"\b[a-zA-Z]\b", "", clean)
                    clean = re.sub(r'\s+', ' ', clean)
                    clean = clean.translate(clean.maketrans("", "", string.punctuation))
                    casefold = clean.casefold()
                    clean = re.sub(r'\bMcD\b|\bMCD\b|\bMcDonalds\b|\bMcDonald\'s\b', 'mcd', clean)
                    clean = re.sub(r'\bKFC\b', 'kfc', clean)
                    clean = re.sub(r'\bstarbak\b|\bStarbucks\b|\bstarbuck\b|\bsbuck\b|\bsbux\b', 'starbucks', clean)

                    tokenizing = nltk.tokenize.word_tokenize(casefold)

                    stop_factory = StopWordRemoverFactory().get_stop_words()
                    dictionary = ArrayDictionary(stop_factory)
                    str = StopWordRemover(dictionary)
                    stop_wr = nltk.tokenize.word_tokenize(str.remove(casefold))
                    kalimat = ' '.join(stop_wr)
                    factory = StemmerFactory()
                    stemmer = factory.create_stemmer()
                    stemming = stemmer.stem(kalimat)
                    
                    try:
                        value = translator.translate(stemming, dest='en')
                        terjemahan = value.text
                        data_label = TextBlob(terjemahan)
                        
                        kata_positif = [
                            "dukung Palestina", "dukung boikot produk", "berhenti mengonsumsi", 
                            "berhenti membeli", "tidak akan lagi membeli", "semangat boikot", 
                            "beralih produk lokal", "dukung boikot", "mendukung boikot",
                            "tolak produk Israel", "mendukung boikot produk",
                            "boikot produk Israel", "mendukung gerakan boikot",
                            "menolak investasi Israel", "ayo boikot terus",
                            "dukung solidaritas Palestina", "stop beli",
                            "tolak produk zionis", "jangan lupa boikot"
                        ]

                        kata_negatif = [
                            "dukung produk Israel", "tolak boikot",
                            "tolak boikot Palestina","masih membeli", 
                            "tetap mengonsumsi", "menyukai", "memakai", 
                            "menolak boikot", "tolak boikot",
                            "tidak setuju dengan boikot",
                            "tolak gerakan boikot"
                        ] 
                        
                        sentiment = "Netral"  
                        
                        if any(kata in terjemahan for kata in kata_positif):
                            sentiment = "Positif"
                        elif any(kata in terjemahan for kata in kata_negatif):
                            sentiment = "Negatif"
                        elif data_label.sentiment.polarity > 0.0:
                            sentiment = "Positif"
                        elif data_label.sentiment.polarity < 0.0:
                            sentiment = "Negatif"
                        else:
                            sentiment = "Netral"

                        row_combined = [row[1], row[14], row[3], clean, sentiment, casefold, tokenizing, stop_wr, stemming]
                        writer_combined.writerow(row_combined)
                        hasil_preprocessing.append(row_combined)
                    except Exception as e:
                        print(f"Translation and labeling error: {e}")

        file_combined.close()

        best_model = "lstm"  # or "naive_bayes"
        lstm_model_path = os.path.join("model", "lstm_model.h5")
        tokenizer_path = os.path.join("model", "tokenizer.pkl")

        if best_model == "Naive Bayes":
            nb_model_path = os.path.abspath("model/naive_bayes_model.pkl")
            nb_vectorizer_path = os.path.abspath("model/tfidf_vectorizer.pkl")
            naive_bayes = joblib.load(nb_model_path)
            vectorizer_nb = joblib.load(nb_vectorizer_path)

            X_test = [row[3] for row in hasil_preprocessing]
            X_test_vect = vectorizer_nb.transform(X_test)
            y_pred_nb = naive_bayes.predict(X_test_vect)

            for i in range(len(hasil_preprocessing)):
                hasil_preprocessing[i].append(y_pred_nb[i])

            with open('static/files/Hasil Preprocessing dengan Sentiment Label.csv', 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerows(hasil_preprocessing)

        elif best_model == "lstm":
            model = load_model(lstm_model_path)
            tokenizer = joblib.load(tokenizer_path)
            max_len = 50
            X_test = tokenizer.texts_to_sequences([row[3] for row in hasil_preprocessing])
            X_test = pad_sequences(X_test, maxlen=max_len)
            y_pred_lstm = model.predict(X_test)

            for i in range(len(hasil_preprocessing)):
                sentiment_label = "Negatif" if y_pred_lstm[i][0] < 0.5 else "Positif"
                hasil_preprocessing[i].append(sentiment_label)

        labels_true = [row[4] for row in hasil_preprocessing]
        labels_pred = [row[9] for row in hasil_preprocessing]
        accuracy = accuracy_score(labels_true, labels_pred)
        print(f"Akurasi model: {accuracy}")
        print(f"Confusion Matrix: {confusion_matrix(labels_true, labels_pred)}")

        return hasil_preprocessing, hasil_labeling

    except Exception as e:
        print(f"Error in preprocessing and labeling: {e}")
        return [], []


@app.route('/fetch_data_from_database', methods=['GET'])
def fetch_data_from_database():
    try:
        cur = mysql.cursor(dictionary=True)
        cur.execute("SELECT * FROM processed_datas")
        processed_data = cur.fetchall()
        cur.close()

        return jsonify(processed_data)

    except Exception as e:
        print(f"Error fetching data from database: {e}")
        return jsonify(error=str(e)), 500
    
# Fungsi untuk mendapatkan tanggal kemarin
def get_yesterday():
    today = datetime.now()
    return today - timedelta(days=1)

# Endpoint untuk mengambil data dari database berdasarkan rentang tanggal
@app.route('/fetch_data_between_dates', methods=['GET'])
def fetch_data_between_dates():
    try:
        start_date_str = request.args.get('startDate')
        end_date_str = request.args.get('endDate')

        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

        # Misalnya, Anda dapat menggunakan range ini untuk mengambil data dari database
        # Di sini Anda harus mengganti sesuai dengan skema database Anda
        cur = mysql.cursor(dictionary=True)
        query = "SELECT * FROM processed_datas WHERE tgl BETWEEN %s AND %s"
        cur.execute(query, (start_date, end_date))
        data = cur.fetchall()
        cur.close()

        return jsonify(data)
    except Exception as e:
        return jsonify(error=str(e))


def save_to_database(cur, hasil_preprocessing):
    try:
        for row in hasil_preprocessing:
            tgl = row[0]    # Tanggal
            user = row[1]   # User
            tweet = row[2]  # Tweet
            clean = row[3]  # Cleaned text
            sentimen = row[4]   # Sentiment
            casefold = row[5]   # Casefold
            tokenizing = ', '.join(row[6])   # Tokenizing
            stopword = ', '.join(row[7])     # Stopword removal
            stemming = row[8]   # Stemming

            # Query SQL untuk memasukkan data ke dalam tabel
            insert_query = """
                INSERT INTO processed_datas (
                    tgl, user, tweet, clean, sentimen, casefold, tokenize, stopword, stemming
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            insert_values = (tgl, user, tweet, clean, sentimen, casefold, tokenizing, stopword, stemming)

            # Eksekusi query
            cur.execute(insert_query, insert_values)
            # Commit perubahan ke database
            mysql.commit()
            print(f"Data berhasil disimpan di database!")

    except Exception as e:
        # Rollback jika terjadi error
        mysql.rollback()
        print(f"Error saat menyimpan data: {e}")


if __name__ == '__main__':
    app.run(debug=True)