import pandas as pd
import numpy as np
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tonga_analysis.visualization_styles import (
    SENTIMENT_COLORS, set_visualization_style, get_sentiment_palette
)

class SentimentAnalyzer:
    """
    Core sentiment analysis functionality for Tonga tourism reviews.
    """
    
    def __init__(self, output_dir=None):
        """
        Initialize the sentiment analyzer.
        
        Parameters:
        - output_dir: Directory to save sentiment analysis outputs
        """
        # Use standardized output directory by default
        if output_dir is None:
            # Use the parent directory's outputs folder
            parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            self.output_dir = os.path.join(parent_dir, "outputs")
        else:
            self.output_dir = output_dir
            
        self.sentiment_dir = os.path.join(self.output_dir, 'sentiment_analysis')
        
        # Create output directories
        if not os.path.exists(self.sentiment_dir):
            os.makedirs(self.sentiment_dir)
            print(f"Created sentiment analysis directory: {self.sentiment_dir}")
            
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            print("Downloading required NLTK data...")
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
            
        # Initialize NLTK tools
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Add domain-specific stop words
        self.stop_words.update([
            'tonga', 'tongan', 'pacific', 'island', 'trip', 'visit', 'stay',
            'would', 'could', 'really', 'got', 'get', 'went', 'made'
        ])
        
        # Define sentiment-related word patterns
        self.sentiment_patterns = {
            'positive': [
                'excellent', 'amazing', 'wonderful', 'fantastic', 'great',
                'awesome', 'perfect', 'best', 'loved', 'beautiful', 'friendly',
                'helpful', 'clean', 'comfortable', 'recommend', 'delicious'
            ],
            'negative': [
                'terrible', 'horrible', 'awful', 'poor', 'bad', 'worst',
                'dirty', 'uncomfortable', 'rude', 'disappointing', 'expensive',
                'overpriced', 'avoid', 'mediocre', 'broken', 'issue'
            ]
        }

    def preprocess_text(self, text):
        """
        Preprocess review text for analysis.
        
        Parameters:
        - text: Raw review text
        
        Returns:
        - Preprocessed text
        """
        if not isinstance(text, str):
            return ''
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stop words and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)

    def analyze_sentiment(self, df):
        """
        Analyze sentiment in review text.
        
        Parameters:
        - df: DataFrame with review data
        
        Returns:
        - DataFrame with sentiment analysis columns added
        """
        print("Analyzing sentiment in reviews...")
        
        # Ensure text column exists
        if 'text' not in df.columns:
            print("Error: No 'text' column found in data")
            return df
            
        # Preprocess text
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        
        # Calculate sentiment scores
        df['sentiment_score'] = df['text'].apply(
            lambda x: TextBlob(str(x)).sentiment.polarity if pd.notnull(x) else 0
        )
        
        # Calculate subjectivity
        df['subjectivity'] = df['text'].apply(
            lambda x: TextBlob(str(x)).sentiment.subjectivity if pd.notnull(x) else 0
        )
        
        # Categorize sentiment
        df['sentiment_category'] = pd.cut(
            df['sentiment_score'],
            bins=[-1, -0.1, 0.1, 1],
            labels=['negative', 'neutral', 'positive']
        )
        
        # Count sentiment-related words
        for sentiment_type, patterns in self.sentiment_patterns.items():
            pattern = '|'.join(patterns)
            col_name = f'{sentiment_type}_word_count'
            df[col_name] = df['text'].str.lower().str.count(pattern)
        
        print("Sentiment analysis complete.")
        return df

    def extract_common_phrases(self, df, min_count=3):
        """
        Extract commonly occurring phrases by sentiment category.
        
        Parameters:
        - df: DataFrame with processed review text
        - min_count: Minimum occurrence count for phrases
        
        Returns:
        - Dictionary of common phrases by sentiment category
        """
        phrases_by_sentiment = {}
        
        for category in ['positive', 'negative', 'neutral']:
            # Filter reviews by sentiment category
            category_df = df[df['sentiment_category'] == category]
            
            if len(category_df) == 0:
                continue
                
            # Combine all processed text
            text = ' '.join(category_df['processed_text'])
            
            # Extract words and their frequencies
            words = text.split()
            word_freq = Counter(words)
            
            # Filter to words appearing at least min_count times
            common_words = {word: count for word, count in word_freq.items() 
                          if count >= min_count}
            
            phrases_by_sentiment[category] = common_words
        
        return phrases_by_sentiment

    def analyze_sentiment_trends(self, df):
        """
        Analyze sentiment trends over time and by category.
        
        Parameters:
        - df: DataFrame with sentiment analysis
        
        Returns:
        - Dictionary with trend analysis results
        """
        trends = {}
        
        # Temporal trends
        if 'published_date' in df.columns:
            df['year'] = pd.to_datetime(df['published_date']).dt.year
            yearly_sentiment = df.groupby('year').agg({
                'sentiment_score': ['mean', 'count', 'std']
            }).reset_index()
            trends['yearly'] = yearly_sentiment.to_dict('records')
        
        # Category trends (using business category)
        category_sentiment = df.groupby('category').agg({
            'sentiment_score': ['mean', 'count', 'std'],
            'positive_word_count': 'sum',
            'negative_word_count': 'sum'
        })
        
        # Convert to more manageable format
        category_stats = []
        for category in category_sentiment.index:
            stats = {
                'name': category,
                'mean_sentiment': category_sentiment.loc[category, ('sentiment_score', 'mean')],
                'review_count': category_sentiment.loc[category, ('sentiment_score', 'count')],
                'positive_words': category_sentiment.loc[category, ('positive_word_count', 'sum')],
                'negative_words': category_sentiment.loc[category, ('negative_word_count', 'sum')]
            }
            category_stats.append(stats)
        
        trends['category'] = category_stats
        return trends

    def generate_sentiment_visualizations(self, df):
        """
        Generate visually appealing visualizations of sentiment analysis results.
        
        Parameters:
        - df: DataFrame with sentiment analysis
        """
        viz_dir = os.path.join(self.sentiment_dir, 'visualizations')
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
            
        # Set consistent style for all plots
        set_visualization_style()
        
        # Get sentiment-specific color palette
        sentiment_palette = get_sentiment_palette()
        
        # 1. Sentiment Distribution with Categories
        plt.figure(figsize=(12, 6))
        
        # Create stacked histogram for different sentiment categories
        n_bins = 30
        
        # Plot in reverse order so positive is on top
        for i, category in enumerate(['negative', 'neutral', 'positive']):
            category_data = df[df['sentiment_category'] == category]['sentiment_score']
            plt.hist(category_data, bins=n_bins, alpha=0.7, 
                    label=category.capitalize(), 
                    color=SENTIMENT_COLORS[category])
        
        # Customize the plot with consistent styling
        plt.title('Distribution of Review Sentiments', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Sentiment Score', fontsize=14)
        plt.ylabel('Number of Reviews', fontsize=14)
        plt.grid(axis='y', alpha=0.3)
        
        # Improve the legend
        plt.legend(frameon=True, framealpha=0.9, facecolor='white', edgecolor='lightgray')
        
        # Add a vertical line at zero
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # Annotate the sentiment regions
        plt.annotate('Negative', xy=(-0.7, plt.ylim()[1]*0.9), 
                    fontsize=12, ha='center', color=SENTIMENT_COLORS['negative'])
        plt.annotate('Neutral', xy=(0, plt.ylim()[1]*0.9), 
                    fontsize=12, ha='center', color=SENTIMENT_COLORS['neutral'])
        plt.annotate('Positive', xy=(0.7, plt.ylim()[1]*0.9), 
                    fontsize=12, ha='center', color=SENTIMENT_COLORS['positive'])
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'sentiment_distribution.png'), 
                    bbox_inches='tight', dpi=300)
        plt.close()
        
        # 2. Sentiment by Business Category
        plt.figure(figsize=(12, 6))
        category_stats = df.groupby('category').agg({
            'sentiment_score': ['mean', 'count', 'std']
        }).reset_index()
        
        # Create bar chart with error bars
        x = range(len(category_stats))
        sentiment_colors = [SENTIMENT_COLORS['positive'], SENTIMENT_COLORS['neutral'], SENTIMENT_COLORS['negative']]
        plt.bar(x, category_stats['sentiment_score']['mean'], 
                yerr=category_stats['sentiment_score']['std'],
                capsize=5, color=sentiment_colors[0], alpha=0.7)
        
        # Customize the plot
        plt.grid(True, alpha=0.3)
        plt.xticks(x, category_stats['category'], rotation=45)
        plt.title('Average Sentiment by Business Category', fontsize=14, pad=20)
        plt.xlabel('Business Category', fontsize=12)
        plt.ylabel('Average Sentiment Score', fontsize=12)
        
        # Add value labels on bars
        for i, stats in enumerate(category_stats.itertuples()):
            mean_val = stats[2]  # Index 2 contains the mean value
            plt.text(i, mean_val, f'{mean_val:.3f}', 
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'sentiment_by_category.png'), dpi=300)
        plt.close()
        
        # 3. Sentiment Trends Over Time
        if 'published_date' in df.columns:
            plt.figure(figsize=(14, 6))
            df['year'] = pd.to_datetime(df['published_date']).dt.year
            yearly_stats = df.groupby('year').agg({
                'sentiment_score': ['mean', 'std']
            }).reset_index()
            
            # Plot mean sentiment line
            plt.plot(yearly_stats['year'], 
                    yearly_stats['sentiment_score']['mean'],
                    color=sentiment_colors[1], linewidth=2, marker='o',
                    label='Mean Sentiment')
            
            # Add confidence band
            plt.fill_between(yearly_stats['year'],
                           yearly_stats['sentiment_score']['mean'] - yearly_stats['sentiment_score']['std'],
                           yearly_stats['sentiment_score']['mean'] + yearly_stats['sentiment_score']['std'],
                           color=sentiment_colors[1], alpha=0.2, label='±1 Std Dev')
            
            plt.grid(True, alpha=0.3)
            plt.title('Sentiment Trends Over Time', fontsize=14, pad=20)
            plt.xlabel('Year', fontsize=12)
            plt.ylabel('Average Sentiment Score', fontsize=12)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'sentiment_over_time.png'), dpi=300)
            plt.close()
        
        # 4. Sentiment Analysis Deep Dive
        plt.figure(figsize=(12, 6))
        
        # Calculate ratios and sizes
        df['pos_neg_ratio'] = df['positive_word_count'] / (df['negative_word_count'] + 1)
        df['total_words'] = df['positive_word_count'] + df['negative_word_count']
        
        # Create scatter plot
        scatter = plt.scatter(df['sentiment_score'], 
                            df['pos_neg_ratio'],
                            alpha=0.5,
                            s=df['total_words'] * 20 + 20,
                            c=df['subjectivity'],
                            cmap='viridis')
        
        plt.grid(True, alpha=0.3)
        plt.colorbar(scatter, label='Subjectivity Score')
        plt.title('Sentiment and Language Analysis', fontsize=14, pad=20)
        plt.xlabel('Overall Sentiment Score', fontsize=12)
        plt.ylabel('Positive to Negative Word Ratio', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'sentiment_deep_dive.png'), dpi=300)
        plt.close()
        
        print(f"Enhanced visualizations saved to {viz_dir}")
        
        # 1. Sentiment Distribution with Categories
        plt.figure(figsize=(12, 6))
        
        # Create stacked histogram for different sentiment categories
        for i, category in enumerate(['positive', 'neutral', 'negative']):
            category_data = df[df['sentiment_category'] == category]
            plt.hist(category_data['sentiment_score'], bins=30, alpha=0.6, 
                    label=category.capitalize(), color=sentiment_colors[i])
        
        plt.title('Distribution of Review Sentiments', fontsize=14, pad=20)
        plt.xlabel('Sentiment Score', fontsize=12)
        plt.ylabel('Number of Reviews', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(viz_dir, 'sentiment_distribution.png'), 
                    bbox_inches='tight', dpi=300)
        plt.close()
        
        # 2. Sentiment by Business Category
        plt.figure(figsize=(12, 6))
        category_means = df.groupby('category')['sentiment_score'].agg(['mean', 'count', 'std']).reset_index()
        
        # Create bar chart with error bars
        bars = plt.bar(category_means['category'], category_means['mean'], 
                      yerr=category_means['std'], capsize=5, color=sentiment_colors[:len(category_means)])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.title('Average Sentiment by Business Category', fontsize=14, pad=20)
        plt.xlabel('Business Category', fontsize=12)
        plt.ylabel('Average Sentiment Score', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(viz_dir, 'sentiment_by_category.png'), 
                    bbox_inches='tight', dpi=300)
        plt.close()
        
        # 3. Sentiment Trends Over Time
        if 'published_date' in df.columns:
            plt.figure(figsize=(14, 6))
            df['year'] = pd.to_datetime(df['published_date']).dt.year
            df['month'] = pd.to_datetime(df['published_date']).dt.month
            
            # Calculate moving average
            yearly_sentiment = df.groupby('year').agg({
                'sentiment_score': ['mean', 'count']
            }).reset_index()
            
            # Create line plot with area and points
            plt.fill_between(yearly_sentiment['year'], 
                           yearly_sentiment[('sentiment_score', 'mean')] - yearly_sentiment[('sentiment_score', 'mean')].std(),
                           yearly_sentiment[('sentiment_score', 'mean')] + yearly_sentiment[('sentiment_score', 'mean')].std(),
                           alpha=0.2, color=sentiment_colors[1])
            
            plt.plot(yearly_sentiment['year'], 
                    yearly_sentiment[('sentiment_score', 'mean')],
                    color=sentiment_colors[1], linewidth=2, marker='o')
            
            plt.title('Sentiment Trends Over Time', fontsize=14, pad=20)
            plt.xlabel('Year', fontsize=12)
            plt.ylabel('Average Sentiment Score', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(viz_dir, 'sentiment_over_time.png'), 
                        bbox_inches='tight', dpi=300)
            plt.close()
        
        # 4. Sentiment and Word Counts
        plt.figure(figsize=(12, 6))
        
        # Create scatter plot with size based on total word count
        positive_counts = df['positive_word_count']
        negative_counts = df['negative_word_count']
        total_counts = positive_counts + negative_counts
        
        plt.scatter(df['sentiment_score'], 
                   positive_counts / (negative_counts + 1),  # Ratio of positive to negative words
                   alpha=0.5, 
                   s=total_counts * 20 + 20,  # Size based on total words
                   c=df['subjectivity'],  # Color based on subjectivity
                   cmap='viridis')
        
        plt.colorbar(label='Subjectivity Score')
        plt.title('Sentiment Analysis Deep Dive', fontsize=14, pad=20)
        plt.xlabel('Overall Sentiment Score', fontsize=12)
        plt.ylabel('Positive to Negative Word Ratio', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(viz_dir, 'sentiment_analysis_deep_dive.png'), 
                    bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Enhanced visualizations saved to {viz_dir}")
        
        print(f"Sentiment visualizations saved to {viz_dir}")

    def analyze_text_sentiment(self, text):
        """
        Analyze sentiment of a single text string.
        
        Parameters:
        - text: Text string to analyze
        
        Returns:
        - Sentiment polarity score (-1 to 1)
        """
        if not isinstance(text, str) or not text:
            return 0
            
        from textblob import TextBlob
        return TextBlob(text).sentiment.polarity
        
    def run_sentiment_analysis(self, df):
        """
        Run complete sentiment analysis pipeline.
        
        Parameters:
        - df: DataFrame with review data
        
        Returns:
        - DataFrame with sentiment analysis added
        - Dictionary with analysis results
        """
        print("\nRunning sentiment analysis pipeline...")
        
        # Analyze sentiment
        df = self.analyze_sentiment(df)
        
        # Extract common phrases
        phrases = self.extract_common_phrases(df)
        
        # Analyze trends
        trends = self.analyze_sentiment_trends(df)
        
        # Generate visualizations
        self.generate_sentiment_visualizations(df)
        
        # Compile results
        results = {
            'summary_stats': {
                'average_sentiment': df['sentiment_score'].mean(),
                'sentiment_std': df['sentiment_score'].std(),
                'positive_reviews': len(df[df['sentiment_category'] == 'positive']),
                'neutral_reviews': len(df[df['sentiment_category'] == 'neutral']),
                'negative_reviews': len(df[df['sentiment_category'] == 'negative'])
            },
            'common_phrases': phrases,
            'trends': trends
        }
        
        return df, results