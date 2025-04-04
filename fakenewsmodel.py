
import numpy as np # linear algebra
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import nltk
from nltk.corpus import stopwords

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
        'confidence': float(1 - confidence),
        'keywords': sorted_keywords
    }
    
article = """
Authentic Investigative Journalism: A Case Study of China's Illegal Fishing Practices in North Korean Waters

Recent investigative reporting by NBC News has uncovered systematic illegal fishing operations by Chinese vessels in North Korean waters, revealing critical insights into maritime resource exploitation and its geopolitical consequences. This report exemplifies rigorous journalistic standards through its methodical evidence collection, multi-source verification, and measurable environmental impact documentation.

Investigative Methodology and Key Findings

1. Satellite Data Analysis and Field Reporting
Investigative journalist Ian Urbina combined synthetic-aperture radar (SAR) satellite imagery with on-the-ground interviews to document over 700 Chinese vessels illegally operating in North Korea's exclusive economic zone between 2017-2021. The analysis revealed:

- 70% decline in North Korea's squid catch (2014-2020)
- 400,000 tons of illegal catch estimated through Automatic Identification System (AIS) tracking gaps
- $440 million annual loss to North Korea's fishing industry

2. Environmental and Economic Impact
The investigation quantified ecological damage through:
- Marine biologists' assessments showing disrupted breeding patterns
- International Union for Conservation of Nature (IUCN) data on species depletion
- Cross-referenced customs records showing mislabeled catches entering Japanese markets

Structural Components of Authentic Journalism

1. Ethical Sourcing and Attribution
The report adheres to Society of Professional Journalists guidelines through:
- Named sources: Captain Zhang Wei (pseudonym protected under journalist safety protocols)
- Document verification: Classified UN Panel of Experts reports obtained through secure channels
- Expert analysis: Dr. Lee Min-ho, marine ecologist at Seoul National University
2. Multi-Platform Presentation
Original reporting incorporated:
- Interactive maps showing vessel movements
- Verified undercover footage from fishing boats
- Historical catch data visualizations (1980-2020)

3. Measurable Outcomes
The investigation prompted:
- EU sanctions against 3 Chinese shipping companies
- Revised UNSC Resolution 2397 enforcement mechanisms
- 15% reduction in illegal transshipments (2023 follow-up analysis)

Verification Process
This article meets authenticity criteria through:

| Verification Metric | Implementation |
|----------------------|----------------|
| Source Transparency | 47 named sources across 6 countries |
| Data Triangulation | Satellite, AIS, and customs record alignment |
| Peer Review | Pre-publication review by maritime law experts |
| Temporal Consistency | 18-month investigation timeline |

Implications for Global Fisheries
The report has become a benchmark in environmental journalism, demonstrating how investigative rigor can:

1. Expose violations of UN Convention on the Law of the Sea (UNCLOS)
2. Document climate change impacts on marine ecosystems
3. Reveal supply chain vulnerabilities in international seafood markets

This work exemplifies journalism's watchdog function through its blend of technological analysis, human storytelling, and policy impact – serving as a template for distinguishing authentic reporting from misinformation.
"""
result = analyze_article(article)
print(f"Prediction: {result['prediction']} ({result['confidence']} confidence)")
print("Key factors:")
for word, score in result['keywords']:
    print(f" - {word}: {score:.4f}")