# News Scraper  

## Overview  
**News Scraper** is a Python script that collects the latest news from Google News, extracts the full article content, title, images, and other metadata from the referenced source, and stores the data in a CSV file for easy analysis.  

## Features  
- Fetches the latest news from Google News  
- Extracts the original article's content, title, image, and metadata  
- Saves the collected data into a CSV file  
- Uses a fake user-agent to avoid bot detection  
- Implements retry logic for reliable scraping  

## Installation  

### Prerequisites  
Ensure you have Python installed (preferably Python 3.7+).  

### Install Dependencies  
First, clone this repository and navigate into the project folder:  

```bash
git clone https://github.com/ashwin549/NewsScraper.git
cd NewsScraper
```
Then install the required dependencies

```bash
pip install -r requirements.txt
```
Then run getarticles.py

```bash
python3 getarticles.py
```
---

The script will:

1. Fetch the latest news from Google News
2. Extract article details (title, content, images, etc.)
3. Save the results to news_data.csv
4. (If the firebase credentials were replaced with your credentials, also upload it to your firestore)

### Notes
- I have included a webscrapingtrial.ipynb Jupyter notebook file, which contains an example output for the code. This can be viewed from github itself for reference.
- Ensure you have an internet connection while running the script.
- Some articles may require JavaScript rendering; this script only extracts static HTML content.
- A sample text summarizer i tried was also included, credit to [colombomf](https://github.com/colombomf/text-summarizer)
- The programs also include code to upload to firebase, which you can either replace with your own firebase credentials, or just remove. It will still extract the news regardless.


