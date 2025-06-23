import os
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from island_review_count import IslandBasedAnalyzer
from sentiment_analyzer import SentimentAnalyzer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Use standardized output directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
output_dir = os.path.join(parent_dir, "outputs")

# First run the island analysis to get island classification
island_analyzer = IslandBasedAnalyzer(
    data_dir='data',  # Use local data directory
    output_dir=output_dir
)
island_analyzer.load_data()
island_analyzer.analyze_islands(top_n=10)

# The all_reviews_df now has island classification
enriched_df = island_analyzer.all_reviews_df

# Initialize sentiment analyzer and run sentiment analysis on the enriched DataFrame
sentiment_analyzer = SentimentAnalyzer(output_dir=output_dir)
enriched_df, sentiment_results = sentiment_analyzer.run_sentiment_analysis(enriched_df)

# Filter to restaurant reviews only
restaurant_df = enriched_df[enriched_df['category'].str.lower() == 'restaurant'].copy()
print(f"Analyzing {len(restaurant_df)} restaurant reviews")

# Define additional stop words for better wordclouds
additional_stop_words = [
    'restaurant', 'food', 'place', 'meal', 'order', 'eat', 'tonga', 'service',
    'staff', 'menu', 'time', 'good', 'great', 'nice', 'delicious', 'really',
    'would', 'could', 'table', 'dish', 'dinner', 'lunch', 'breakfast', 'night',
    'day', 'get', 'got', 'went', 'made', 'didn', "didn't", 'don', "don't",
    'take', 'took', 'come', 'came', 'back', 'even', 'one', 'two', 'three',
    'also', 'br', 'much', 'very', 'well', 'little', 'try', 'tried'
]

# Get base stop words
stop_words = set(stopwords.words('english'))
stop_words.update(additional_stop_words)
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """Preprocess text for wordcloud"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stop words and lemmatize
    tokens = [lemmatizer.lemmatize(token) for token in tokens 
             if token not in stop_words and len(token) > 2]
    
    return ' '.join(tokens)

# Function to generate and save wordcloud
def generate_positive_wordcloud(df, island, min_words=50):
    # Filter for positive reviews in this island
    positive_df = df[(df['island_category'] == island) & (df['sentiment_score'] > 0.2)]
    
    if len(positive_df) < 10:
        print(f"Not enough positive reviews for {island}")
        return
    
    # Create visualization directory
    island_dir = os.path.join(output_dir, 'restaurant_analysis', 'island_analysis', island)
    if not os.path.exists(island_dir):
        os.makedirs(island_dir)
    
    # Combine text from positive reviews
    all_text = ' '.join(positive_df['text'].fillna(''))
    
    # Preprocess the text
    processed_text = preprocess_text(all_text)
    
    if len(processed_text.split()) < min_words:
        print(f"Not enough words for {island} after preprocessing")
        return
    
    print(f"Generating positive sentiment wordcloud for {island} with {len(positive_df)} reviews")
    
    # Generate wordcloud
    wordcloud = WordCloud(
        width=1000, 
        height=600, 
        background_color='white',
        colormap='YlGn',  # Green color map for positive sentiment
        max_words=200,
        collocations=True,
        contour_width=1,
        contour_color='steelblue'
    ).generate(processed_text)
    
    # Plot
    plt.figure(figsize=(16, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Positive Sentiment Words - Restaurant Reviews on {island}', fontsize=24, pad=20)
    plt.tight_layout()
    
    # Save
    filename = f'positive_sentiment_wordcloud.png'
    plt.savefig(os.path.join(island_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved wordcloud to {os.path.join(island_dir, filename)}")

# Generate wordclouds for each island
islands = ['Tongatapu', 'Vava\'u', 'Ha\'apai', '\'Eua']
for island in islands:
    generate_positive_wordcloud(restaurant_df, island)

print("Wordcloud generation complete!")