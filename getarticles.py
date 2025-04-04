import requests
from bs4 import BeautifulSoup
import pandas as pd
from googlenewsdecoder import gnewsdecoder
from urllib.parse import urlparse
from fake_useragent import UserAgent
from newspaper import Article
from urllib.parse import unquote
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import time
import numpy as np # linear algebra
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords') 
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist
import random
import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase
cred = credentials.Certificate(r"C:\Users\Ashwin\Desktop\MAD_LAB_PROJECT\NewsScraper\truthlens-68bed-firebase-adminsdk-fbsvc-4c3626b3ec.json")  # Replace with your JSON file path
firebase_admin.initialize_app(cred)



# 1. Create and fit a temporary vectorizer with sample vocabulary
sample_corpus = [
    "fake news political scandal fraud cryptocurrency",
    "real genuine authentic trustworthy verified"
]
vectorizer = CountVectorizer(max_features=1000)
vectorizer.fit(sample_corpus) 

# 2. Model setup with original architecture
model = Sequential([
    Dense(100, activation='relu', input_dim=1023669),
    Dense(50, activation='relu'),
    Dense(25, activation='relu'),
    Dense(10, activation='relu'),
    Dense(1, activation='sigmoid')
])

# 3. Load weights manually
weights = np.load(r'C:\Users\Ashwin\Desktop\MAD_LAB_PROJECT\NewsScraper\model_weights.npy', allow_pickle=True)
for layer, (weights, bias) in zip(model.layers, weights.reshape(-1, 2)):
    layer.set_weights([weights, bias])
    
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# Initialize Firestore client
db = firestore.client()

# Configure session with retry logic
session = requests.Session()
retries = Retry(
    total=3,
    backoff_factor=0.5,
    status_forcelist=[500, 502, 503, 504],
    allowed_methods=["GET"]
)
session.mount('https://', HTTPAdapter(max_retries=retries))

#Getting top news from google news
def get_news(rss_url="https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en"):
    response = requests.get(rss_url)
    soup = BeautifulSoup(response.content, 'xml')
    
    return [{
        'title': item.title.text,
        'source': item.source.text,
        'time': item.pubDate.text,
        'link': item.link.text
    } for item in soup.find_all('item')]




#Decoding the google news links to original source

def decode_google_url(google_url):
    try:
        path_segments = urlparse(google_url).path.split('/')
        article_id = path_segments[-1] if path_segments else ""
        
        result = gnewsdecoder(
            google_url,
            interval=2,  # 2s delay between requests
            proxy=None  # Add proxy if getting blocked
        )
        
        if result['status']:
            return result['decoded_url']
        return google_url  # Fallback to original URL
    
    except Exception as e:
        print(f"Decoding failed: {str(e)}")
        return google_url



def get_full_article_content(article_url):
    headers = {
        'User-Agent': UserAgent().random,
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://news.google.com/',
        'DNT': '1'
    }
    
    try:
        response = session.get(article_url, headers=headers, timeout=(3.05, 10))
        response.raise_for_status()

        article = Article(article_url)
        article.download(input_html=response.text)
        article.parse()
        return {
            'title': article.title,
            'publish_date': article.publish_date,
            'text': article.text,
            'top_image': article.top_image,
            'source_url': article_url,
        }

    except requests.exceptions.Timeout:
        print(f"⏰ Timeout skipped: {article_url}")
        return None
    except Exception as e:
        print(f"⚠️ Error processing {article_url}: {str(e)}")
        return None
    
# Batch processing with progress tracking
def process_news_items(news_items, delay=1):
    results = []
    for idx, item in enumerate(news_items, 1):
        print(f"Processing article {idx}/{len(news_items)}...")
        article = get_full_article_content(item['link'])
        if article:
            results.append(article)
        time.sleep(delay)  # delay between requests
    return results


def preprocess(text):
    """Basic text cleaning"""
    return ' '.join([
        word.lower() 
        for word in text.split() 
        if word.lower() not in stop_words and word.isalpha()
    ])
def analyze_article(article, top_n=5):
    """
    Predict whether an article is Fake or Real, with confidence and key words.

    Args:
    article (str): The text of the article to analyze.
    top_n (int): Number of top influential words to return.

    Returns:
    dict: Containing prediction, confidence, and top influential words.
    """
    # Preprocess the article
    processed = preprocess(article)

    # Transform the article into a BOW representation
    bow_vector = vectorizer.transform([processed]).toarray()
    # Pad the BOW vector to match the model's input shape (1023669 features)
    padded_bow = np.zeros((1, 1023669))  # Create a zero array with required shape
    padded_bow[0, :bow_vector.shape[1]] = bow_vector  # Copy existing features into padded array

    # Get prediction confidence
    confidence = model.predict(padded_bow, verbose=0)[0][0]
    prediction = 'Fake' if confidence > 0.5 else 'Real'

    # Get feature importance using first layer weights
    input_weights = model.layers[0].get_weights()[0]  # First layer weights
    feature_importance = padded_bow.dot(input_weights)  # Multiply BOW by weights

    # Extract only words present in the article (non-zero BOW entries)
    feature_names = vectorizer.get_feature_names_out()
    non_zero_indices = np.nonzero(bow_vector[0])[0]  # Indices of non-zero features
    keywords_with_scores = [(feature_names[i], feature_importance[0, i]) for i in non_zero_indices]

    # Sort by absolute impact score and select top N keywords
    sorted_keywords = sorted(keywords_with_scores, key=lambda x: abs(x[1]), reverse=True)[:top_n]
    return {
        'prediction': prediction,
        'confidence': float(confidence if prediction == 'Fake' else 1 - confidence),
        'keywords': sorted_keywords
    }
    
def text_summarizer(text, num_sentences=3):
    # Text into sentences
    sentences = sent_tokenize(text)

    # Text into words
    words = word_tokenize(text.lower())

    # Removing stop words
    stop_words = set(stopwords.words("english"))
    filtered_words = [word for word in words if word.casefold() not in stop_words]

    # Calculate word frequencies
    fdist = FreqDist(filtered_words)

    # Assign scores to sentences based on word frequencies
    sentence_scores = [sum(fdist[word] for word in word_tokenize(sentence.lower()) if word in fdist)
                       for sentence in sentences]

    # Create a list of tuples containing sentence index and score
    sentence_scores = list(enumerate(sentence_scores))

    # Sort sentences by scores in descending order
    sorted_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)

    # Randomly select the top `num_sentences` sentences for the summary
    random_sentences = random.sample(sorted_sentences, num_sentences)

    # Sort the randomly selected sentences based on their original order in the text
    summary_sentences = sorted(random_sentences, key=lambda x: x[0])

    # Create the summary
    summary = ' '.join([sentences[i] for i, _ in summary_sentences])

    return summary



def upload_article(title, article, confidence, image, source, tags):
    """
    Uploads an article to the Firestore collection 'articles'.
    
    Args:
        title (str): Title of the article.
        article (str): Full text of the article.
        confidence (float): Confidence score of the article.
        image (str): URL of the article image.
        source (str): Source URL of the article.
        tags (list): List of tags related to the article.

    Returns:
        str: Document ID of the uploaded article.
    """
    data = {
        "title": title,
        "article": article,
        "confidence": confidence,
        "image": image,
        "source": source,
        "tags": tags
    }
    
    doc_ref = db.collection("news").add(data)
    return doc_ref[1].id  


news=get_news()
for item in news:
    item['link'] = decode_google_url(item['link'])
newslist = process_news_items(news)

# Save the news to a CSV file
#df = pd.DataFrame(newslist)
#df.to_csv('news.csv', index=False)
#print(f"Saved {len(df)} articles to news.csv")


for i in newslist:
    topimage=i['top_image']
    asource=i['source_url']
    atitle=i['title']
    text=i['text']
    summary=text_summarizer(text)
    analysis = analyze_article(text)
    confidence_rating = analysis['confidence']
    confidence_rating=round(confidence_rating * 100, 1)
    keywords = analysis['keywords']
    words=[]
    
    for w,s in keywords:
        words.append(w)
    print("Title: ", atitle)
    print("Confidence: ", confidence_rating)
    print("Keywords: ", words)
    
    # # Upload the summarized article along with its metadata to Firestore
    # doc_id = upload_article(
    #     title=atitle,  # Title of the article
    #     article=summary,  # Summarized content of the article
    #     confidence=confidence_rating,  # Confidence score of the analysis
    #     image=topimage,  # URL of the top image from the article
    #     source=asource,  # Source URL of the article
    #     tags=words  # Keywords extracted from the article
    # )
    #print("Document uploaded:",doc_id," with confidence: ", confidence_rating)
    
    