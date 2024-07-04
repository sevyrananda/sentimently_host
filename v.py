@app.route('/fetch_data_from_database', methods=['GET'])
def fetch_data_from_database():
    try:
        cur = mysql.cursor(dictionary=True)
        cur.execute("SELECT * FROM hasil_crawl ORDER BY created_at DESC")
        processed_data = cur.fetchall()

        # Ambil data terbaru berdasarkan created_at
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

        return jsonify(processed_data, response)
    except Exception as e:
        print(f"Error fetching data from database: {e}")
        return jsonify(error=str(e)), 500