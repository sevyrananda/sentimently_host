from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file 
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
import mysql.connector
import os, re, csv, string
import pandas as pd
import joblib
import pickle
import nltk
import time
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Bidirectional
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
from datetime import datetime, timedelta
import shutil
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io

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
    'database': 'new_sentimently'
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
            return redirect(url_for('dataset'))
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
    keywords = set()  # Set untuk menyimpan semua keyword unik

    cur = mysql.cursor(dictionary=True)
    query = "SELECT tweet, sentimen, keyword FROM hasil_crawl"
    cur.execute(query)

    rows = cur.fetchall()
    print(f"Rows fetched from database: {rows}")  # Tambahkan logging

    for row in rows:
        tweet = row['tweet']
        sentimen = row['sentimen']
        keyword = row['keyword']
        keywords.add(keyword)  # Tambahkan keyword ke set

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

    print(f"Negative products: {negative_products}")  # Tambahkan logging
    print(f"Positive products: {positive_products}")  # Tambahkan logging
    print(f"Neutral products: {neutral_products}")    # Tambahkan logging
    
    cur.close()
    return negative_products, positive_products, neutral_products, keywords

def get_overall_sentiment():
    sentiment_counts = {"positif": 0, "negatif": 0, "netral": 0}
    
    cur = mysql.cursor(dictionary=True)
    query = "SELECT sentimen FROM hasil_crawl"
    cur.execute(query)
    
    for row in cur.fetchall():
        sentimen = row['sentimen']
        if sentimen.lower() in sentiment_counts:
            sentiment_counts[sentimen.lower()] += 1

    cur.close()
    
    if all(count == 0 for count in sentiment_counts.values()):
        overall_majority_sentiment = None
    else:
        overall_majority_sentiment = max(sentiment_counts, key=sentiment_counts.get)

    return overall_majority_sentiment

@app.route('/guest')
def guest():
    products_to_analyze = [
        {"name": "kfc", "image": "/static/kfc.png"},
        {"name": "mcd", "image": "/static/mcd.png"},
        {"name": "starbucks", "image": "/static/sbux.png"},
        {"name": "burger king", "image": "/static/bk.png"},
        {"name": "aqua", "image": "/static/aqua.png"},
        {"name": "nestle", "image": "/static/nestle.png"},
        {"name": "pizza hut", "image": "/static/ph.png"},
        {"name": "oreo", "image": "/static/oreo.png"},
        {"name": "unilever", "image": "/static/unv.png"},
    ]

    negative_products, positive_products, neutral_products, keywords = get_sentiment_data(products_to_analyze)

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

    print(f"Products sentiment: {products_sentiment}")  # Tambahkan logging

    sorted_products_sentiment = dict(sorted(products_sentiment.items(), 
                                            key=lambda item: (item[1]['positif']['count'], item[1]['netral']['count'], item[1]['negatif']['count']), 
                                            reverse=True))

    overall_majority_sentiment = get_overall_sentiment()

    # Log overall majority sentiment
    print(f"Overall majority sentiment: {overall_majority_sentiment}")

    cur = mysql.cursor(dictionary=True)
    cur.execute("SELECT * FROM users")
    users = cur.fetchall()
    cur.close()

    # Buat word cloud untuk halaman guest
    wordcloud = generate_word_cloud()

    return render_template('guest.html', 
                           products_sentiment=sorted_products_sentiment,
                           overall_majority_sentiment=overall_majority_sentiment,
                           keywords=keywords,
                           users=users,
                           wordcloud_image=wordcloud)



def generate_word_cloud():
    # Ambil data tweet dari database
    cur = mysql.cursor(dictionary=True)
    query = "SELECT tweet FROM hasil_crawl"
    cur.execute(query)
    tweets = [row['tweet'] for row in cur.fetchall()]
    cur.close()
    
    if not tweets:  # Jika tidak ada tweet, return gambar default atau pesan
        print("No tweets available for word cloud.")
        return None
    
    # Gabungkan semua tweet menjadi satu string
    tweets_text = ' '.join(tweets)
    
    # Buat word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(tweets_text)
    
    # Simpan word cloud ke buffer
    img = io.BytesIO()
    wordcloud.to_image().save(img, format='PNG')
    img.seek(0)
    
    return img

@app.route('/word_cloud_image')
def word_cloud_image():
    img = generate_word_cloud()
    return send_file(img, mimetype='image/png')


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

    cur.execute("SELECT * FROM hasil_crawl")
    processed_data = cur.fetchall()

    cur.execute("SELECT * FROM dataset")
    dataset = cur.fetchall()

    cur.execute("SELECT * FROM evaluasi_train ORDER BY id DESC LIMIT 1")
    evaluasi_train = cur.fetchone()

    cur.close()
    
    if evaluasi_train:
        # Convert accuracy to percentage
        acc_nb_percent = round(evaluasi_train['acc_nb'] * 100, 2)
        acc_lstm_percent = round(evaluasi_train['acc_lstm'] * 100, 2)
        
        # Update evaluasi_train with converted values
        evaluasi_train['acc_nb'] = acc_nb_percent
        evaluasi_train['acc_lstm'] = acc_lstm_percent

    # Pass all necessary data to the template
    return render_template('dataset.html',
                           dataset=dataset, 
                           users=users, 
                           processed_data=processed_data, 
                           evaluasi_train=evaluasi_train,
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

@app.route('/delete_all_datasets', methods=['DELETE'])
def delete_all_datasets():
    cur = mysql.cursor(dictionary=True)
    cur.execute("DELETE FROM dataset")
    mysql.commit()

    cur.close()
    return jsonify({'success': True})
    



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
        sql = "INSERT INTO hasil_crawl (tgl, user, tweet, clean, casefold, tokenize, stopword, stemming, sentimen) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
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

    cur.execute("SELECT * FROM evaluasi_train ORDER BY id DESC LIMIT 1")
    evaluasi_train = cur.fetchone()

    cur.close()
    return render_template('dataset.html', 
                           dataset=dataset, 
                           evaluasi_train=evaluasi_train)

@app.route('/dataset', methods=['POST'])
def uploadFiles():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        if not uploaded_file.filename.endswith('.csv'):
            flash('Only CSV files are allowed!', 'danger')  # Flash error message
            return redirect(url_for("dataset"))
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        uploaded_file.save(file_path)
        parseCSV(file_path)
        flash('Data successfully added!', 'success')  # Flash success message
        # Set a cookie to indicate that file has been processed
        response = redirect(url_for("dataset"))
        response.set_cookie('file_processed', 'true')
        return response
    return redirect(url_for("dataset"))

# LOAD MODEL DARI HUGGING FACE - BERT
def parseCSV(filePath):
    # Check if the CSV contains a 'sentiment' column
    csvData = pd.read_csv(filePath)
    if 'sentiment' in csvData.columns:
        # CSV already contains sentiment labels
        for i, row in csvData.iterrows():
            full_text = row['full_text']
            sentiment = row['sentiment']
            # Insert into database
            sql = "INSERT INTO dataset (full_text, sentiment) VALUES (%s, %s)"
            value = (full_text, sentiment)
            cur = mysql.cursor()
            cur.execute(sql, value)
            mysql.commit()
    else:
        # CSV does not contain sentiment labels, perform sentiment analysis
        for i, row in csvData.iterrows():
            full_text = row['full_text']
            
            # Perform sentiment analysis
            sentiment_result = sentiment_analyzer(full_text)[0]
            sentiment = sentiment_result['label']
            
            # Print sentiment for debugging
            print(f"Original sentiment label: {sentiment}")
            
            # Map sentiment labels to desired format
            if sentiment == 'Negative':
                sentiment = 'negatif'
            elif sentiment == 'Neutral':
                sentiment = 'netral'
            elif sentiment == 'Positive':
                sentiment = 'positif'
            
            # Insert into database
            sql = "INSERT INTO dataset (full_text, sentiment) VALUES (%s, %s)"
            value = (full_text, sentiment)
            cur = mysql.cursor()
            cur.execute(sql, value)
            mysql.commit()

# PEMODELAN MENGGUNAKAN ALGORITMA NAIVE BAYES DAN LSTM
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
        nb_accuracy, processing_time_nb, naive_bayes, vectorizer_nb, nb_pred = train_naive_bayes(X_train, X_test, y_train, y_test)
        lstm_accuracy, processing_time_lstm, model, tokenizer, lstm_pred_classes = train_lstm(df, X_train, X_test, y_train, y_test)

        # Determine best model
        best_model = evaluate_models(nb_accuracy, lstm_accuracy)

        # Generate and save confusion matrix for the best model
        if best_model == 'Naive Bayes':
            save_confusion_matrix(y_test, nb_pred, model_folder, 'confusion_matrix_nb.png')
        else:
            save_confusion_matrix(y_test, lstm_pred_classes, model_folder, 'confusion_matrix_lstm.png')

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
    cur = mysql.cursor(dictionary=True)
    cur.execute("SELECT * FROM dataset")
    dataset = cur.fetchall()
    cur.close()

    df = pd.DataFrame(dataset)
    df.dropna(subset=['sentiment'], inplace=True)

    df['clean_tweet'] = df['full_text'].apply(preprocess)
    X_train, X_test, y_train, y_test = train_test_split(df['clean_tweet'], df['sentiment'], test_size=0.2, random_state=42)

    sentiment_mapping = {'negatif': 0, 'netral': 1, 'positif': 2}
    y_train = y_train.map(sentiment_mapping)
    y_test = y_test.map(sentiment_mapping)

    if y_train.isnull().any() or y_test.isnull().any():
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

    return nb_accuracy, processing_time_nb, naive_bayes, vectorizer_nb, nb_pred

def train_lstm(df, X_train, X_test, y_train, y_test):
    tokenizer = Tokenizer(num_words=5000, split=' ')
    tokenizer.fit_on_texts(df['clean_tweet'].values)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_train_pad = pad_sequences(X_train_seq, maxlen=100)  # Set maxlen
    X_test_pad = pad_sequences(X_test_seq, maxlen=X_train_pad.shape[1])  # Set maxlen sama dengan X_train_pad

    model = Sequential()
    model.add(Embedding(5000, 128, input_length=X_train_pad.shape[1]))
    model.add(SpatialDropout1D(0.2))
    model.add(Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    start_time_lstm = time.time()
    model.fit(X_train_pad, y_train, epochs=10, batch_size=64, validation_data=(X_test_pad, y_test))
    end_time_lstm = time.time()

    lstm_pred = model.predict(X_test_pad, batch_size=64)
    lstm_pred_classes = lstm_pred.argmax(axis=1)
    lstm_accuracy = accuracy_score(y_test, lstm_pred_classes)
    processing_time_lstm = end_time_lstm - start_time_lstm

    return lstm_accuracy, processing_time_lstm, model, tokenizer, lstm_pred_classes

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
    try:
        cur = mysql.cursor()
        cur.execute("INSERT INTO evaluasi_train (best_model, acc_nb, processtime_nb, acc_lstm, processtime_lstm) VALUES (%s, %s, %s, %s, %s)",
                    (best_model, nb_accuracy, processing_time_nb, lstm_accuracy, processing_time_lstm))
        
        mysql.commit()
        cur.close()
    except Exception as e:
        print(f"Error saving training results: {e}")
        raise

def save_confusion_matrix(y_test, predictions, model_folder, filename):
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix: {filename}')
    plt.savefig(os.path.join(model_folder, filename))
    plt.close()


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
        cur.execute("DELETE FROM tweet_clean WHERE dataset_id = %s", (dataset_id,))
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

# Fungsi Preprocess
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
    casefold = re.sub(r'\bKFC\b|\bKfc\b', 'kfc', casefold)
    casefold = re.sub(r'\bstarbak\b|\bStarbucks\b|\bstarbuck\b|\bsbuck\b|\bsbux\b|\bSTARBUCKS\b', 'starbucks', casefold)
    casefold = re.sub(r'\bAqua\b|\bAQUA\b', 'aqua', casefold)
    casefold = re.sub(r'\bOreo\b|\bOREO\b', 'oreo', casefold)
    casefold = re.sub(r'\bNestle\b|\bNESTLE\b', 'nestle', casefold)
    casefold = re.sub(r'\bUnilever\b|\bUNILEVER\b', 'unilever', casefold)
    casefold = re.sub(r'\bPizza Hut\b|\bpizza Hut\b|\bPH\b|\bph\b', 'pizza hut', casefold)
    casefold = re.sub(r'\bBurger King\b|\bBurger king\b|\BK\b|\bk\b', 'burger king', casefold)

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
    cleaned_sentence = preprocess(sentence)
    return cleaned_sentence

def predict_sentiment_best_model(input_sentence, model_type):
    if model_type == "naive_bayes":
        input_tfidf = vectorizer_nb.transform([input_sentence])
        predicted_sentiment = naive_bayes.predict(input_tfidf)[0]
        sentiment_scores = None
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



# FITUR CEK ANALISIS SENTIMEN
@app.route('/analyze', methods=['POST'])
def analyze():
    input_sentence = request.form['text']
    cleaned_input = preprocess(input_sentence)  
    
    if model_type == "naive_bayes":
        processed_input = preprocess_input_sentence_nb(cleaned_input)
        predicted_sentiment, sentiment_scores = predict_sentiment_best_model(processed_input, model_type)
    elif model_type == "lstm":
        predicted_sentiment, sentiment_scores = predict_sentiment_best_model(cleaned_input, model_type)
    
    # Ambil data produk dan keyword untuk ditampilkan di halaman yang sama
    products_to_analyze = [
        {"name": "kfc", "image": "/static/kfc.png"},
        {"name": "mcd", "image": "/static/mcd.png"},
        {"name": "starbucks", "image": "/static/sbux.png"},
        {"name": "burger king", "image": "/static/bk.png"},
        {"name": "aqua", "image": "/static/aqua.png"},
        {"name": "nestle", "image": "/static/nestle.png"},
        {"name": "pizza hut", "image": "/static/ph.png"},
        {"name": "oreo", "image": "/static/oreo.png"},
        {"name": "unilever", "image": "/static/unv.png"},
    ]

    negative_products, positive_products, neutral_products, keywords = get_sentiment_data(products_to_analyze)

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

    overall_majority_sentiment = get_overall_sentiment()

    # Log overall majority sentiment
    print(f"Overall majority sentiment: {overall_majority_sentiment}")

    return render_template('guest.html', 
                           input_text=input_sentence,
                           sentiment=predicted_sentiment,
                           scores=sentiment_scores,
                           clean_text=cleaned_input,
                           products_sentiment=sorted_products_sentiment,
                           overall_majority_sentiment=overall_majority_sentiment,
                           keywords=keywords,
                           wordcloud_image=generate_word_cloud())

# Function to load the best model based on evaluation
def load_best_model():
    try:
        nb_model, nb_accuracy = evaluate_naive_bayes_model()
        lstm_model, lstm_accuracy = evaluate_lstm_model()

        if nb_accuracy > lstm_accuracy:
            return (nb_model, None) # No tokenizer for Naive Bayes
        else:
            return lstm_model
    except Exception as e:
        print(f"Error loading the best model: {e}")
        return None

# Function to evaluate Naive Bayes model
def evaluate_naive_bayes_model():
    try:
        cur = mysql.cursor(dictionary=True)
        cur.execute("SELECT * FROM dataset")
        prepro = cur.fetchall()
        cur.close()

        df = pd.DataFrame(prepro)

        X = df['full_text']
        y = df['sentiment']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        vectorizer = TfidfVectorizer()
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        model = MultinomialNB()
        model.fit(X_train_tfidf, y_train)

        y_pred = model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)

        return (model, vectorizer), accuracy
    except Exception as e:
        print(f"Error during Naive Bayes model evaluation: {e}")
        return None, 0

def evaluate_lstm_model():
    try:
        cur = mysql.cursor(dictionary=True)
        cur.execute("SELECT * FROM dataset")
        prepro = cur.fetchall()
        cur.close()

        df = pd.DataFrame(prepro)

        X = df['full_text']
        y = df['sentiment']

        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(X)
        X_seq = tokenizer.texts_to_sequences(X)
        X_pad = pad_sequences(X_seq, maxlen=100)

        if len(X_pad) != len(y):
            raise ValueError("Found input variables with inconsistent numbers of samples: {} and {}".format(len(X_pad), len(y)))

        X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=42)

        model = load_model('model/lstm_model.h5')

        y_pred = model.predict(X_test)
        y_pred_classes = y_pred.argmax(axis=1)  # Get the index of the max logit
        accuracy = accuracy_score(y_test, y_pred_classes)

        return (model, tokenizer), accuracy
    except Exception as e:
        print(f"Error during LSTM model evaluation: {e}")
        return None, 0

@app.route('/scraping', methods=['GET'])
@login_required
def scraping():
    cur = mysql.cursor(dictionary=True)
    cur.execute("SELECT * FROM users")
    users = cur.fetchall()

    cur.execute("SELECT * FROM hasil_crawl")
    processed_data = cur.fetchall()

    cur.close()

    today = datetime.now()
    yesterday = today - timedelta(days=1)
    yesterday_date = yesterday.strftime('%Y-%m-%d')
    today_date = today.strftime('%Y-%m-%d')
    return render_template('scraping.html', users=users, processed_data=processed_data, yesterday_date=yesterday_date, today_date=today_date)
    
scraped_files = []

today = datetime.now()
yesterday = today - timedelta(days=1)
since_date = yesterday.strftime('%Y-%m-%d')
until_date = today.strftime('%Y-%m-%d')

def is_scraped_today(existing_data, target_date):
    for row in existing_data:
        row_date = datetime.strptime(row['tgl'], '%a %b %d %H:%M:%S %z %Y')
        if row_date.date() == target_date.date():
            return True
    return False

@app.route('/do_scrape', methods=['POST'])
def do_scrape():
    try:
        cur = mysql.cursor(dictionary=True)

        # Get the request JSON data
        request_data = request.get_json()
        since_date = request_data.get('since_date', yesterday.strftime('%Y-%m-%d'))
        until_date = request_data.get('until_date', today.strftime('%Y-%m-%d'))
        keyword = request_data.get('keyword', 'boikot produk')
        scrape_type = request_data.get('scrape_type', 'predict')  # Default to 'predict'

        if scrape_type == 'predict':
            query = "SELECT * FROM hasil_crawl WHERE created_at = (SELECT MAX(created_at) FROM hasil_crawl)"
            cur.execute(query)
            existing_data = cur.fetchall()

            if is_scraped_today(existing_data, yesterday):
                cur.close()
                return jsonify(message="Crawling sudah dilakukan. Data yang sudah di-crawl:", data=existing_data)
    
        auth_token = "0e12b16141a80c4510f95de2dcd5ef5b365b3fb3"
        limit = 10
        search_keyword = f'{keyword} lang:id until:{until_date} since:{since_date}'

        data = 'data_boikot.csv'
        os.system(f'npx tweet-harvest@2.6.1 -o "{data}" -s "{search_keyword}" -l {limit} --token "{auth_token}"')

        source_file = 'tweets-data/data_boikot.csv'
        if scrape_type == 'csv':
            destination_file = 'static/files/Dataset_train.csv'
            shutil.copyfile(source_file, destination_file)
            return jsonify(message="Crawling berhasil dan data disimpan sebagai CSV.", data=[])
        else:
            destination_file = 'static/files/Data Scraping.csv'
            shutil.copyfile(source_file, destination_file)

            hasil_preprocessing = preprocessing_twitter()

            if hasil_preprocessing is not None:
                save_to_database(cur, hasil_preprocessing, keyword)

                test_models_after_scraping()

                major_sentiment = calculate_majority_sentiment(hasil_preprocessing)

                cur.close()

                hasil_data = []
                for row in hasil_preprocessing:
                    hasil_data.append({
                        "tgl": row[0],
                        "user": row[1],
                        "tweet": row[2],
                        "clean": row[3],
                        "sentimen": row[4],
                        "keyword": keyword
                    })

                accuracy = calculate_accuracy(hasil_preprocessing)
                total_tweets = len(hasil_preprocessing)

                print(f"Major sentiment after crawling: {major_sentiment}")

                return jsonify(
                    message="Crawling berhasil!",
                    data=hasil_data,
                    accuracy=accuracy,
                    total_tweets=total_tweets,
                    major_sentiment=major_sentiment
                )
            else:
                cur.close()
                print("No data to process after crawling.")
                return jsonify(message="No data to process after crawling.", data=[])

    except Exception as e:
        print(f"Error during crawling and processing: {e}")
        return jsonify(error=str(e), data=[])


# Function to calculate accuracy based on labels
def calculate_accuracy(data):
    labels_true = [row[4] for row in data]
    labels_pred = [row[4] for row in data]  # Assuming predicted labels are in the same column
    accuracy = accuracy_score(labels_true, labels_pred)
    return accuracy * 100

# Function to calculate majority sentiment
def calculate_majority_sentiment(data):
    sentiments = [row[4] for row in data]
    majority_sentiment = max(set(sentiments), key=sentiments.count)
    return majority_sentiment

hasil_preprocessing = []

def preprocessing_twitter():
    try:
        file_combined = open('static/files/Data Preprocessing Labeling.csv', 'w', newline='', encoding='utf-8')
        writer_combined = csv.writer(file_combined)

        hasil_preprocessing.clear()

        with open("static/files/Data Scraping.csv", "r", encoding='utf-8') as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            next(readCSV)

            for row in readCSV:
                if len(row) <= 14:
                    print("Row does not have enough elements:", row)
                    continue

                tgl = row[1]
                user = row[14]
                tweet = row[3]
                preprocessed_text = preprocess(tweet)

                row_combined = [tgl, user, tweet, preprocessed_text]
                hasil_preprocessing.append(row_combined)
                writer_combined.writerow(row_combined)  # Write to CSV

        file_combined.close()

        model_and_tokenizer, vectorizer_or_tokenizer = load_best_model()

        if model_and_tokenizer is not None:
            model, vectorizer_or_tokenizer = model_and_tokenizer
            if isinstance(model, MultinomialNB):
                predict_with_naive_bayes(model, vectorizer_or_tokenizer)
            else:
                predict_with_lstm(model, vectorizer_or_tokenizer)

        with open("static/files/Data Preprocessing Labeling.csv", "r", encoding='utf-8') as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            for row in readCSV:
                if len(row) >= 5:
                    tgl = row[0]
                    user = row[1]
                    tweet = row[2]
                    clean = row[3]
                    sentimen = row[4]
                    hasil_preprocessing.append([tgl, user, tweet, clean, sentimen])

        return hasil_preprocessing

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None



# Function to predict with Naive Bayes
def predict_with_naive_bayes(model, vectorizer):
    try:
        df = pd.DataFrame(hasil_preprocessing, columns=['tgl', 'user', 'tweet', 'clean'])
        X_tfidf = vectorizer.transform(df['clean'])
        y_pred = model.predict(X_tfidf)
        df['sentimen'] = y_pred

        hasil_preprocessing.clear()
        hasil_preprocessing.extend(df.values.tolist())

    except Exception as e:
        print(f"Error during Naive Bayes prediction: {e}")

# Function to predict with LSTM
def predict_with_lstm(model, tokenizer):
    try:
        if tokenizer is None:
            raise ValueError("Tokenizer is not initialized.")

        df = pd.DataFrame(hasil_preprocessing, columns=['tgl', 'user', 'tweet', 'clean'])

        X_seq = tokenizer.texts_to_sequences(df['clean'])
        X_pad = pad_sequences(X_seq, maxlen=100)

        y_pred = model.predict(X_pad)
        y_pred_classes = y_pred.argmax(axis=1)  
        df['sentimen'] = y_pred_classes

        hasil_preprocessing.clear()
        hasil_preprocessing.extend(df.values.tolist())

    except Exception as e:
        print(f"Error during LSTM prediction: {e}")


def test_models_after_scraping():
    try:
        # Load best model and tokenizer
        model_and_tokenizer, vectorizer_or_tokenizer = load_best_model()

        if model_and_tokenizer is not None:
            model, vectorizer_or_tokenizer = model_and_tokenizer

            # Ambil data terbaru dari database
            cur = mysql.cursor(dictionary=True)
            cur.execute("SELECT * FROM hasil_crawl ORDER BY created_at DESC")
            new_data = cur.fetchall()
            cur.close()

            if not new_data:
                raise ValueError("No new data available for testing.")

            df_new = pd.DataFrame(new_data)
            X_new = df_new['clean']
            y_true = df_new['sentimen']

            # Validasi bahwa X_new dan y_true memiliki jumlah sampel yang sama
            if len(X_new) != len(y_true):
                raise ValueError(f"Inconsistent sample sizes: X_new={len(X_new)}, y_true={len(y_true)}")

            # Catat waktu mulai
            start_time = time.time()

            # Preprocessing for new data
            if isinstance(model, MultinomialNB):
                X_tfidf_new = vectorizer_or_tokenizer.transform(X_new)
                y_pred = model.predict(X_tfidf_new)

                # Evaluate performance
                accuracy = accuracy_score(y_true, y_pred)
                # Output akurasi dan waktu proses
                print(f"Akurasi model pada data baru: {accuracy * 100:.2f}%")

            elif isinstance(model, Sequential):  # LSTM model
                X_seq_new = vectorizer_or_tokenizer.texts_to_sequences(X_new)
                X_pad_new = pad_sequences(X_seq_new, maxlen=100)

                # Validasi bahwa X_pad_new dan y_true memiliki jumlah sampel yang sama
                if len(X_pad_new) != len(y_true):
                    raise ValueError(f"Inconsistent sample sizes after padding: X_pad_new={len(X_pad_new)}, y_true={len(y_true)}")

                y_pred_proba = model.predict(X_pad_new)
                y_pred = (y_pred_proba.argmax(axis=1))  # Mengubah prediksi probabilitas ke kelas

                # Evaluate performance
                accuracy = accuracy_score(y_true, y_pred)
                # Output akurasi dan waktu proses
                print(f"Akurasi model LSTM pada data baru: {accuracy * 100:.2f}%")

            # Catat waktu selesai dan hitung durasi
            end_time = time.time()
            process_time = end_time - start_time
            print(f"Waktu proses: {process_time:.2f} detik")

    except Exception as e:
        print(f"Error during model testing after crawling: {e}")

@app.route('/fetch_data_from_database', methods=['GET'])
def fetch_data_from_database():
    try:
        cur = mysql.cursor(dictionary=True)
        cur.execute("SELECT * FROM hasil_crawl ORDER BY created_at DESC")
        processed_data = cur.fetchall()

        query = """
            SELECT * FROM hasil_crawl
            WHERE created_at = (SELECT MAX(created_at) FROM hasil_crawl)
        """
        cur.execute(query)
        latest_data = cur.fetchall()
        cur.close()

        if latest_data:
            major_sentiment = calculate_majority_sentiment(latest_data)
            accuracy = calculate_accuracy(latest_data)
            total_tweets = len(latest_data)

            response = {
                'data': processed_data,
                'majoritySentiment': major_sentiment,
                'accuracy': accuracy,
                'totalTweets': total_tweets,
            }
        else:
            response = {
                'data': [],
                'majoritySentiment': 'Tidak ada data',
                'accuracy': 0,
                'totalTweets': 0,
            }

        return jsonify(response)
    except Exception as e:
        # print(f"Error fetching data from database: {e}")
        return jsonify(error=str(e)), 500


# Route to fetch data between dates
@app.route('/fetch_data_between_dates', methods=['GET'])
def fetch_data_between_dates():
    try:
        start_date_str = request.args.get('startDate')
        end_date_str = request.args.get('endDate')

        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

        # Mengubah format date untuk mencocokkan dengan format di database
        start_date_formatted = start_date.strftime('%Y-%m-%d 00:00:00')
        end_date_formatted = end_date.strftime('%Y-%m-%d 23:59:59')

        cur = mysql.cursor(dictionary=True)
        query = """
            SELECT * FROM hasil_crawl 
            WHERE created_at BETWEEN %s AND %s
            ORDER BY created_at DESC
        """
        cur.execute(query, (start_date_formatted, end_date_formatted))
        data = cur.fetchall()
        cur.close()

        return jsonify(data)
    except Exception as e:
        print(f"Error fetching data between dates:", e)
        return jsonify(error=str(e)), 500


def save_to_database(cur, hasil_preprocessing, keyword):
    try:
        if not hasil_preprocessing:
            raise ValueError("No data to save.")

        for row in hasil_preprocessing:
            if len(row) < 5:
                print("Row does not have enough elements:", row)
                continue

            tgl = row[0]
            user = row[1]
            tweet = row[2]
            clean = row[3]
            sentimen = row[4]

            # Cek apakah tweet sudah ada di database
            check_query = """
                SELECT COUNT(*) as count FROM hasil_crawl 
                WHERE tgl = %s AND user = %s AND tweet = %s
            """
            check_values = (tgl, user, tweet)
            cur.execute(check_query, check_values)
            result = cur.fetchone()

            if result['count'] == 0:
                # Hanya menyimpan jika tweet belum ada di database
                insert_query = """
                    INSERT INTO hasil_crawl (
                        tgl, user, tweet, clean, sentimen, keyword, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, NOW())
                """
                insert_values = (tgl, user, tweet, clean, sentimen, keyword)

                cur.execute(insert_query, insert_values)
                mysql.commit()
                print(f"Data berhasil disimpan di database!")
            else:
                print(f"Tweet sudah ada di database, tidak disimpan ulang: {tweet}")

    except Exception as e:
        mysql.rollback()
        print(f"Error saat menyimpan data: {e}")


if __name__ == '__main__':
    app.run(debug=True)