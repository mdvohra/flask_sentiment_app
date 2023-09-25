from flask import Flask, render_template, request, flash
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import requests
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a secure secret key

# Load the pre-trained CountVectorizer and sentiment prediction model
count_vectorizer = joblib.load('count_vectorizer.pkl')
count_model = joblib.load('count_model.pkl')

# Function to interact with the YouTube API
def google_api(video_id):
    api_service_name = "youtube"
    api_version = "v3"
    DEVELOPER_KEY = "AIzaSyCnbw6996drTKIpgVK55HOgunO1MGK7x40"  # Replace with your YouTube API key

    youtube = build(api_service_name, api_version, developerKey=DEVELOPER_KEY)

    request = youtube.commentThreads().list(
        part="id,snippet",
        maxResults=300,
        order="relevance",
        videoId=video_id
    )

    response = request.execute()
    return response

# Function to extract comments from API response
def get_comments(video_id):
    response = google_api(video_id)
    comments = []

    for item in response["items"]:
        comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        comments.append(comment)

    return comments

# Function to predict sentiment based on comments
def predict_sentiment(comments):
    # Transform the input comments using the loaded count vectorizer
    count_pred = count_vectorizer.transform(comments)

    # Predict sentiment using the loaded model
    count_svc_pred = count_model.predict(count_pred)

    # Count the number of positive, negative, and neutral predictions
    neutral = (count_svc_pred == 0.0).sum()
    positive = (count_svc_pred == 1.0).sum()
    negative = (count_svc_pred < 0).sum()

    # Calculate the overall video sentiment
    overall_sentiment = "Good video" if positive > negative else "Bad video"

    # Return both the overall sentiment and individual comment sentiments
    comment_sentiments = ["Positive" if pred == 1.0 else "Negative" if pred < 0 else "Neutral" for pred in count_svc_pred]
    
    return overall_sentiment, comment_sentiments

# Function to send a message via Telegram
def send_telegram_message(chat_id, message):
    # Your Telegram Bot Token
    telegram_bot_token = "6029756102:AAGTsQbuaIWGc-zt2NIDI0lpvZOlec06M2k"

    telegram_api_url = f"https://api.telegram.org/bot{telegram_bot_token}/sendMessage"

    data = {
        "chat_id": chat_id,
        "text": message
    }

    try:
        response = requests.post(telegram_api_url, data=data)
        if response.status_code == 200:
            print("Message sent successfully via Telegram")
        else:
            print("Error sending message via Telegram:", response.text)
    except requests.RequestException as e:
        print("Error sending message via Telegram:", str(e))

# Flask route
@app.route('/', methods=['GET', 'POST'])
def index():
    overall_sentiment = None
    comments_with_sentiments = []  # Initialize list to store comments with sentiments
    
    if request.method == 'POST':
        video_id = request.form['video_id']
        chat_id = request.form['chat_id'] or "469017414"

        comments = get_comments(video_id)
        overall_sentiment, comment_sentiments = predict_sentiment(comments)

        # Combine comments with predicted sentiments
        comments_with_sentiments = list(zip(comments, comment_sentiments))

        message = f"Sentiment Report for Video ID: {video_id}\n"
        message += f"Total Comments: {len(comments)}\n"
        message += f"Predicted Overall Sentiment: {overall_sentiment}"

        send_telegram_message(chat_id, message)
        flash("Sentiment analysis completed and sent via Telegram!")

    return render_template('index.html', overall_sentiment=overall_sentiment, comments_with_sentiments=comments_with_sentiments)

if __name__ == '__main__':
    app.run(debug=True)
