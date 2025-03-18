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

# Configure session with retry logic
session = requests.Session()
retries = Retry(
    total=3,
    backoff_factor=0.5,
    status_forcelist=[500, 502, 503, 504],
    allowed_methods=["GET"]
)
session.mount('https://', HTTPAdapter(max_retries=retries))

#Getting top news from google news (many links to sites)
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
    """Decode Google News tracking URL to original source"""
    try:
        # Extract article ID from URL path
        path_segments = urlparse(google_url).path.split('/')
        article_id = path_segments[-1] if path_segments else ""
        
        # Use the decoder package
        result = gnewsdecoder(
            google_url,
            interval=2,  # Add 2s delay between requests
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
        time.sleep(delay)  # Respectful delay between requests
    return results

news=get_news()
for item in news:
    item['link'] = decode_google_url(item['link'])
newslist = process_news_items(news)

# Save the news to a CSV file
df = pd.DataFrame(newslist)
df.to_csv('news.csv', index=False)
print(f"Saved {len(df)} articles to news.csv")
