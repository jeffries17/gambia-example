import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import re
from collections import defaultdict, Counter
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from wordcloud import WordCloud
from review_analyzer import TongaReviewAnalyzer
import matplotlib.cm as cm
from matplotlib.colors import Normalize

class ThemeSentimentDriverAnalyzer:
    """
    Analyzes what drives sentiment within specific themes across different nationalities.
    Identifies key phrases, words, and aspects that contribute to positive or negative sentiment.
    """
    
    def __init__(self, data_dir='tonga_data', output_dir='outputs/theme_sentiment_drivers'):
        """
        Initialize the theme sentiment driver analyzer.
        
        Parameters:
        - data_dir: Directory containing the review data
        - output_dir: Directory to save findings and visualizations
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # Create output directory structure
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Create subdirectories for different categories
        for category in ['accommodations', 'restaurants', 'attractions', 'visualizations']:
            category_dir = os.path.join(output_dir, category)
            if not os.path.exists(category_dir):
                os.makedirs(category_dir)
                
        # Target countries to focus on
        self.target_countries = ['New Zealand', 'Australia', 'United States', 'United Kingdom', 'Tonga']
        
        # Define color scheme for consistent visualizations
        self.country_colors = {
            'New Zealand': '#1E88E5',   # Blue
            'Australia': '#FFC107',     # Amber
            'United States': '#D81B60', # Pink
            'United Kingdom': '#004D40', # Teal
            'Tonga': '#43A047'          # Green
        }
        
        # Define main themes by category for analysis
        self.themes = {
            'accommodation': {
                'cleanliness': ['clean', 'dirty', 'spotless', 'tidy', 'dust', 'hygiene', 'sanitary', 'filthy', 'immaculate'],
                'staff': ['staff', 'service', 'friendly', 'helpful', 'reception', 'rude', 'accommodating', 'attentive'],
                'location': ['location', 'central', 'beach', 'view', 'scenic', 'convenient', 'quiet', 'walking distance', 'accessible'],
                'value': ['value', 'price', 'expensive', 'affordable', 'overpriced', 'budget', 'worth', 'reasonable'],
                'room_quality': ['room', 'bed', 'comfortable', 'spacious', 'small', 'cramped', 'mattress', 'pillow', 'air conditioning', 'shower', 'bathroom'],
                'facilities': ['pool', 'wifi', 'internet', 'breakfast', 'restaurant', 'gym', 'spa', 'amenities', 'facility', 'parking']
            },
            'restaurant': {
                'food_quality': ['food', 'delicious', 'tasty', 'fresh', 'flavor', 'bland', 'quality', 'cooked', 'portion', 'hot', 'cold', 'dishes'],
                'service': ['service', 'staff', 'waiter', 'waitress', 'friendly', 'attentive', 'slow', 'quick', 'prompt'],
                'value': ['price', 'value', 'expensive', 'affordable', 'overpriced', 'cheap', 'cost', 'reasonable'],
                'ambiance': ['ambiance', 'atmosphere', 'decor', 'noisy', 'quiet', 'romantic', 'mood', 'lighting', 'music', 'seating'],
                'menu_variety': ['menu', 'options', 'variety', 'choices', 'selection', 'vegetarian', 'vegan', 'choice', 'diverse'],
                'local_cuisine': ['local', 'authentic', 'traditional', 'tongan', 'island', 'polynesian', 'pacific']
            },
            'attraction': {
                'activity_quality': ['activity', 'tour', 'experience', 'fun', 'exciting', 'boring', 'adventure', 'enjoyable'],
                'natural_beauty': ['beautiful', 'scenic', 'view', 'nature', 'beach', 'landscape', 'stunning', 'paradise', 'gorgeous'],
                'guides': ['guide', 'knowledgeable', 'informative', 'friendly', 'helpful', 'professional'],
                'value': ['price', 'value', 'expensive', 'affordable', 'overpriced', 'cost', 'worth', 'money', 'fee', 'admission'],
                'uniqueness': ['unique', 'special', 'amazing', 'memorable', 'lifetime', 'unforgettable', 'unusual', 'different'],
                'accessibility': ['accessible', 'easy', 'difficult', 'challenging', 'hike', 'walk', 'path', 'distance', 'steps']
            }
        }
        
        # Define subthemes (specific aspects within themes) to analyze deeper
        self.subthemes = {
            'accommodation': {
                'staff': {
                    'friendliness': ['friendly', 'nice', 'welcoming', 'warm', 'smile', 'greet'],
                    'helpfulness': ['helpful', 'assist', 'help', 'accommodating', 'support', 'problem'],
                    'professionalism': ['professional', 'efficient', 'prompt', 'knowledge', 'courteous'],
                    'responsiveness': ['quick', 'response', 'slow', 'wait', 'attend', 'timely']
                },
                'value': {
                    'pricing': ['price', 'cost', 'expensive', 'cheap', 'affordable', 'overpriced'],
                    'worth': ['worth', 'value', 'money', 'paid', 'expect', 'quality for price'],
                    'inclusions': ['include', 'breakfast', 'free', 'extras', 'amenities', 'package'],
                    'comparison': ['compare', 'other hotels', 'similar', 'alternative', 'better deal']
                },
                'room_quality': {
                    'bed': ['bed', 'mattress', 'pillow', 'sheets', 'sleep', 'comfortable'],
                    'bathroom': ['bathroom', 'shower', 'toilet', 'clean', 'water pressure', 'hot water'],
                    'space': ['spacious', 'small', 'cramped', 'tiny', 'large', 'enough room'],
                    'noise': ['quiet', 'noisy', 'noise', 'walls', 'soundproof', 'hear']
                }
            },
            'restaurant': {
                'food_quality': {
                    'taste': ['taste', 'delicious', 'flavor', 'bland', 'seasoning', 'spicy', 'sweet', 'fresh'],
                    'portion': ['portion', 'serving', 'size', 'small', 'large', 'generous', 'enough'],
                    'preparation': ['cooked', 'raw', 'underdone', 'overcooked', 'perfect', 'temperature', 'hot', 'cold'],
                    'presentation': ['presentation', 'beautiful', 'plated', 'look', 'appear', 'attractive']
                },
                'service': {
                    'attentiveness': ['attentive', 'attention', 'ignore', 'forget', 'check', 'visited table'],
                    'speed': ['quick', 'fast', 'slow', 'wait', 'timely', 'prompt', 'delay'],
                    'knowledge': ['knowledge', 'menu', 'recommendation', 'suggest', 'explain', 'describe'],
                    'friendliness': ['friendly', 'smile', 'rude', 'pleasant', 'welcoming', 'attitude', 'nice']
                },
                'value': {
                    'pricing': ['price', 'cost', 'expensive', 'cheap', 'affordable', 'overpriced'],
                    'portion_value': ['worth', 'value', 'money', 'paid', 'expect', 'quality for price', 'size'],
                    'quality_price': ['quality', 'price', 'ratio', 'expect', 'standard', 'level']
                }
            },
            'attraction': {
                'guides': {
                    'knowledge': ['knowledge', 'informed', 'expert', 'learn', 'history', 'facts', 'explain'],
                    'personality': ['friendly', 'funny', 'entertaining', 'humorous', 'engaging', 'personable'],
                    'communication': ['clear', 'understand', 'accent', 'communication', 'language', 'explanation'],
                    'attentiveness': ['attentive', 'care', 'safety', 'concern', 'help', 'assist']
                },
                'value': {
                    'admission_cost': ['admission', 'entry', 'fee', 'ticket', 'cost', 'price'],
                    'experience_value': ['worth', 'worthwhile', 'value', 'experience', 'money', 'pay'],
                    'inclusions': ['include', 'extra', 'additional', 'gear', 'food', 'transport'],
                    'time_value': ['time', 'spent', 'duration', 'length', 'short', 'long']
                },
                'activity_quality': {
                    'enjoyment': ['fun', 'enjoy', 'exciting', 'boring', 'thrill', 'entertainment'],
                    'organization': ['organized', 'schedule', 'plan', 'on time', 'delay', 'wait'],
                    'safety': ['safe', 'danger', 'equipment', 'precaution', 'security', 'trust'],
                    'uniqueness': ['unique', 'special', 'once in a lifetime', 'different', 'standard']
                }
            }
        }
        
        # Load data
        self.reviews_df = None
        
        # Initialize NLTK components
        self._initialize_nltk()
        
    def _initialize_nltk(self):
        """Initialize NLTK components for text analysis."""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            print("Downloading required NLTK data...")
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
            
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def load_data(self):
        """
        Load review data using the TongaReviewAnalyzer class.
        """
        print("Loading review data...")
        
        # Use the existing review analyzer to load data
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        analyzer = TongaReviewAnalyzer(
            data_dir=os.path.join(parent_dir, self.data_dir),
            output_dir=self.output_dir
        )
        analyzer.load_data()
        
        if analyzer.all_reviews_df is None or len(analyzer.all_reviews_df) == 0:
            print("No review data available for analysis!")
            return False
            
        # Store review data
        self.reviews_df = analyzer.all_reviews_df
        
        # Extract country from user_location
        self._extract_country_information()
        
        # Add basic sentiment analysis
        self._add_sentiment_analysis()
        
        print(f"Loaded {len(self.reviews_df)} reviews for analysis.")
        return True
        
    def _extract_country_information(self):
        """
        Extract and standardize country information from user_location.
        """
        if 'user_location' not in self.reviews_df.columns:
            print("Warning: No user_location data available for country extraction.")
            self.reviews_df['country'] = 'Unknown'
            return
            
        # Common country mappings for standardization
        country_mappings = {
            'australia': 'Australia',
            'new zealand': 'New Zealand',
            'nz': 'New Zealand',
            'usa': 'United States',
            'united states': 'United States',
            'us': 'United States',
            'america': 'United States',
            'uk': 'United Kingdom',
            'united kingdom': 'United Kingdom',
            'england': 'United Kingdom',
            'great britain': 'United Kingdom',
            'tonga': 'Tonga'
        }
        
        # Extract country from location string
        def extract_country(location):
            if pd.isna(location) or location == '' or location == 'Unknown':
                return 'Unknown'
                
            location = str(location).lower()
            
            # Direct mapping
            for key, value in country_mappings.items():
                if key in location:
                    return value
                    
            # Try from country format (e.g., "City, Country")
            parts = [part.strip() for part in location.split(',')]
            if len(parts) > 1:
                for key, value in country_mappings.items():
                    if key in parts[-1]:
                        return value
                        
            return 'Other'
            
        # Apply extraction
        self.reviews_df['country'] = self.reviews_df['user_location'].apply(extract_country)
        
        # Convert 'nan' and empty strings to 'Unknown'
        self.reviews_df.loc[self.reviews_df['country'].isin(['nan', '', None]), 'country'] = 'Unknown'
        
        # Print summary of countries
        country_counts = self.reviews_df['country'].value_counts()
        print("\nCountry distribution in data:")
        for country, count in country_counts.items():
            if country in self.target_countries or country == 'Unknown' or country == 'Other':
                print(f"- {country}: {count}")
                
    def _add_sentiment_analysis(self):
        """
        Add sentiment analysis scores to the reviews.
        """
        if 'text' not in self.reviews_df.columns:
            print("Warning: No review text available for sentiment analysis.")
            return
            
        print("Adding sentiment analysis...")
        
        # Function to compute sentiment using TextBlob
        def get_sentiment(text):
            if pd.isna(text) or text == '':
                return 0
            blob = TextBlob(str(text))
            return blob.sentiment.polarity
            
        # Apply sentiment analysis
        self.reviews_df['sentiment'] = self.reviews_df['text'].apply(get_sentiment)
        
        # Add sentiment categories
        self.reviews_df['sentiment_category'] = pd.cut(
            self.reviews_df['sentiment'],
            bins=[-1, -0.1, 0.1, 1],
            labels=['negative', 'neutral', 'positive']
        )
        
        print("Sentiment analysis complete.")
        
    def _extract_theme_sentences(self, text, theme_keywords):
        """
        Extract sentences related to a specific theme from review text.
        
        Parameters:
        - text: The review text
        - theme_keywords: Keywords related to the theme
        
        Returns:
        - List of sentences containing theme keywords
        """
        if pd.isna(text) or text == '':
            return []
            
        text = str(text).lower()
        sentences = sent_tokenize(text)
        
        # Extract sentences containing theme keywords
        theme_sentences = []
        for sentence in sentences:
            if any(keyword in sentence for keyword in theme_keywords):
                theme_sentences.append(sentence)
                
        return theme_sentences
        
    def _extract_subtheme_sentences(self, sentences, subtheme_keywords):
        """
        Extract sentences related to a specific subtheme from sentences.
        
        Parameters:
        - sentences: List of sentences to check
        - subtheme_keywords: Keywords related to the subtheme
        
        Returns:
        - List of sentences containing subtheme keywords
        """
        # Extract sentences containing subtheme keywords
        subtheme_sentences = []
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in subtheme_keywords):
                subtheme_sentences.append(sentence)
                
        return subtheme_sentences
        
    def _get_sentence_sentiment(self, sentence):
        """
        Get sentiment score for a sentence.
        
        Parameters:
        - sentence: Sentence to analyze
        
        Returns:
        - Sentiment score
        """
        blob = TextBlob(sentence)
        return blob.sentiment.polarity
        
    def _extract_sentiment_drivers(self, sentences, sentiment_type='positive'):
        """
        Extract key words and phrases that drive sentiment.
        
        Parameters:
        - sentences: List of sentences to analyze
        - sentiment_type: 'positive' or 'negative'
        
        Returns:
        - Tuple of (words, bigrams) that drive sentiment
        """
        # First, determine the sentiment of each sentence
        if sentiment_type == 'positive':
            filtered_sentences = [s for s in sentences if self._get_sentence_sentiment(s) > 0.2]
        else:
            filtered_sentences = [s for s in sentences if self._get_sentence_sentiment(s) < -0.2]
            
        if not filtered_sentences:
            return ([], [])
            
        # Combine all sentences and process text
        text = ' '.join(filtered_sentences)
        words = self._preprocess_text(text)
        
        # Extract individual words
        word_freq = Counter(words)
        # Remove extremely common words that might not be insightful
        for common in ['good', 'great', 'nice', 'very', 'really']:
            if common in word_freq:
                del word_freq[common]
                
        # Get top words
        top_words = [w for w, _ in word_freq.most_common(20)]
        
        # Extract bigrams
        tokens = word_tokenize(text.lower())
        finder = BigramCollocationFinder.from_words(tokens)
        # Apply word filter
        finder.apply_word_filter(lambda w: w in self.stop_words or len(w) < 3)
        # Score bigrams and get top
        bigram_measures = BigramAssocMeasures()
        top_bigrams = finder.nbest(bigram_measures.pmi, 15)
        
        return (top_words, top_bigrams)
        
    def _preprocess_text(self, text):
        """
        Preprocess text for analysis.
        
        Parameters:
        - text: The text to preprocess
        
        Returns:
        - List of preprocessed words
        """
        if pd.isna(text) or text == '':
            return []
            
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        processed_tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]
        
        return processed_tokens
        
    def analyze_theme_drivers(self, category=None, theme=None, country=None):
        """
        Analyze what drives sentiment for specific themes.
        
        Parameters:
        - category: Category to analyze (if None, analyze all)
        - theme: Theme to analyze (if None, analyze all themes in category)
        - country: Country to analyze (if None, analyze all target countries)
        
        Returns:
        - Dictionary with analysis results
        """
        if self.reviews_df is None:
            print("No data loaded. Please run load_data() first.")
            return {}
            
        # Determine what to analyze
        categories_to_analyze = [category] if category else list(self.themes.keys())
        countries_to_analyze = [country] if country else self.target_countries
        
        # Filter reviews to selected countries
        filtered_df = self.reviews_df[self.reviews_df['country'].isin(countries_to_analyze)]
        
        # Results structure
        results = {}
        
        # Analyze each category
        for cat in categories_to_analyze:
            # Filter to category
            cat_df = filtered_df[filtered_df['category'] == cat]
            if len(cat_df) == 0:
                continue
                
            # Determine themes to analyze
            themes_to_analyze = [theme] if theme and theme in self.themes[cat] else self.themes[cat].keys()
            
            cat_results = {}
            for t in themes_to_analyze:
                print(f"Analyzing drivers for {cat} - {t}...")
                theme_results = self._analyze_theme_drivers(cat_df, cat, t, countries_to_analyze)
                cat_results[t] = theme_results
                
                # Generate visualizations
                self._generate_theme_driver_visualizations(cat, t, theme_results)
                
            results[cat] = cat_results
            
        # Save results to file
        result_path = os.path.join(self.output_dir, 'sentiment_drivers_analysis.json')
        try:
            with open(result_path, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"Results saved to {result_path}")
        except Exception as e:
            print(f"Error saving results: {e}")
        
        return results
        
    def _analyze_theme_drivers(self, df, category, theme, countries):
        """
        Analyze what drives sentiment for a specific theme.
        
        Parameters:
        - df: DataFrame filtered to the category
        - category: Category name
        - theme: Theme name
        - countries: List of countries to analyze
        
        Returns:
        - Dictionary with analysis results
        """
        theme_keywords = self.themes[category][theme]
        theme_results = {
            'overall': {},
            'by_country': {},
            'by_subtheme': {}
        }
        
        # Process each country
        for country in countries:
            country_df = df[df['country'] == country]
            if len(country_df) < 5:  # Skip countries with too few reviews
                continue
                
            # Extract theme sentences for this country
            all_theme_sentences = []
            for _, row in country_df.iterrows():
                sentences = self._extract_theme_sentences(row['text'], theme_keywords)
                all_theme_sentences.extend(sentences)
                
            if not all_theme_sentences:
                continue
                
            # Calculate overall sentiment for theme
            sentiments = [self._get_sentence_sentiment(s) for s in all_theme_sentences]
            avg_sentiment = np.mean(sentiments) if sentiments else 0
            
            # Extract key drivers for positive and negative sentiment
            pos_drivers = self._extract_sentiment_drivers(all_theme_sentences, 'positive')
            neg_drivers = self._extract_sentiment_drivers(all_theme_sentences, 'negative')
            
            # Get example sentences
            pos_examples = []
            for s in all_theme_sentences:
                if self._get_sentence_sentiment(s) > 0.5 and len(s) > 20:
                    pos_examples.append(s)
                    if len(pos_examples) >= 2:
                        break
                        
            neg_examples = []
            for s in all_theme_sentences:
                if self._get_sentence_sentiment(s) < -0.3 and len(s) > 20:
                    neg_examples.append(s)
                    if len(neg_examples) >= 2:
                        break
            
            # Store results for this country
            theme_results['by_country'][country] = {
                'avg_sentiment': avg_sentiment,
                'mention_count': len(all_theme_sentences),
                'positive_drivers': {
                    'words': pos_drivers[0],
                    'phrases': [' '.join(bg) for bg in pos_drivers[1]]
                },
                'negative_drivers': {
                    'words': neg_drivers[0],
                    'phrases': [' '.join(bg) for bg in neg_drivers[1]]
                },
                'examples': {
                    'positive': pos_examples,
                    'negative': neg_examples
                }
            }
            
            # Analyze subthemes if available
            if category in self.subthemes and theme in self.subthemes[category]:
                subtheme_results = {}
                
                for subtheme, keywords in self.subthemes[category][theme].items():
                    # Extract sentences for this subtheme
                    subtheme_sentences = self._extract_subtheme_sentences(all_theme_sentences, keywords)
                    
                    if not subtheme_sentences:
                        continue
                        
                    # Calculate sentiment
                    sub_sentiments = [self._get_sentence_sentiment(s) for s in subtheme_sentences]
                    sub_avg_sentiment = np.mean(sub_sentiments) if sub_sentiments else 0
                    
                    # Extract drivers
                    sub_pos_drivers = self._extract_sentiment_drivers(subtheme_sentences, 'positive')
                    sub_neg_drivers = self._extract_sentiment_drivers(subtheme_sentences, 'negative')
                    
                    # Store subtheme results
                    if country not in theme_results['by_subtheme']:
                        theme_results['by_subtheme'][country] = {}
                        
                    subtheme_display = subtheme.replace('_', ' ').title()
                    theme_results['by_subtheme'][country][subtheme] = {
                        'avg_sentiment': sub_avg_sentiment,
                        'mention_count': len(subtheme_sentences),
                        'mention_percentage': len(subtheme_sentences) / len(all_theme_sentences) * 100 if all_theme_sentences else 0,
                        'positive_drivers': {
                            'words': sub_pos_drivers[0],
                            'phrases': [' '.join(bg) for bg in sub_pos_drivers[1]]
                        },
                        'negative_drivers': {
                            'words': sub_neg_drivers[0],
                            'phrases': [' '.join(bg) for bg in sub_neg_drivers[1]]
                        }
                    }
                    
        return theme_results
        
    def _generate_theme_driver_visualizations(self, category, theme, results):
        """
        Generate visualizations for theme sentiment drivers.
        
        Parameters:
        - category: Category name
        - theme: Theme name
        - results: Results dictionary for the theme
        """
        if 'by_country' not in results or not results['by_country']:
            return
            
        # Create visualization directories
        viz_dir = os.path.join(self.output_dir, 'visualizations')
        category_dir = os.path.join(self.output_dir, f'{category}s')
        for directory in [viz_dir, category_dir]:
            os.makedirs(directory, exist_ok=True)
            
        # Format names for display
        theme_display = theme.replace('_', ' ').title()
        category_display = category.title()
        
        # 1. Create sentiment comparison chart
        self._create_theme_sentiment_chart(category, theme, results)
        
        # 2. Create word clouds for each country
        self._create_driver_wordclouds(category, theme, results)
        
        # 3. Create subtheme analysis charts
        if 'by_subtheme' in results and results['by_subtheme']:
            self._create_subtheme_analysis(category, theme, results)
            
    def _create_theme_sentiment_chart(self, category, theme, results):
        """
        Create a chart comparing sentiment for a theme across countries.
        
        Parameters:
        - category: Category name
        - theme: Theme name
        - results: Results dictionary for the theme
        """
        # Format for display
        theme_display = theme.replace('_', ' ').title()
        category_display = category.title()
        
        # Prepare data
        countries = []
        sentiments = []
        mentions = []
        
        for country, data in results['by_country'].items():
            countries.append(country)
            sentiments.append(data['avg_sentiment'])
            mentions.append(data['mention_count'])
            
        if not countries:
            return
            
        # Create color map
        colors = [self.country_colors.get(country, '#BDBDBD') for country in countries]
        
        # Sort by sentiment
        sorted_data = sorted(zip(countries, sentiments, mentions, colors), key=lambda x: x[1], reverse=True)
        countries, sentiments, mentions, colors = zip(*sorted_data)
        
        # Create the chart
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Bar chart for sentiment
        bars = ax1.bar(countries, sentiments, color=colors, alpha=0.7)
        
        # Add sentiments on bars
        for bar, sentiment in zip(bars, sentiments):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.05 if height >= 0 else height - 0.1,
                f'{height:.2f}',
                ha='center', va='bottom' if height >= 0 else 'top',
                fontweight='bold'
            )
            
        ax1.set_ylabel('Sentiment Score', fontsize=12)
        ax1.set_xlabel('Country', fontsize=12)
        ax1.set_title(f'{theme_display} Sentiment by Country - {category_display}', fontsize=16)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.2)
        ax1.grid(axis='y', alpha=0.3)
        
        # Line for mentions
        ax2 = ax1.twinx()
        ax2.plot(countries, mentions, 'ro-', linewidth=2, markersize=8)
        
        # Add mentions as text
        for i, (country, mention) in enumerate(zip(countries, mentions)):
            ax2.text(i, mention + max(mentions)*0.05, str(mention), ha='center', va='bottom', color='red', fontweight='bold')
            
        ax2.set_ylabel('Mention Count', color='red', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='red')
        
        plt.tight_layout()
        
        # Save the figure
        viz_dir = os.path.join(self.output_dir, 'visualizations')
        plt.savefig(os.path.join(viz_dir, f'{category}_{theme}_sentiment_drivers.png'), dpi=300)
        plt.close()
        
    def _create_driver_wordclouds(self, category, theme, results):
        """
        Create word clouds for theme sentiment drivers.
        
        Parameters:
        - category: Category name
        - theme: Theme name
        - results: Results dictionary for the theme
        """
        # Format for display
        theme_display = theme.replace('_', ' ').title()
        category_display = category.title()
        
        viz_dir = os.path.join(self.output_dir, 'visualizations')
        
        # For each country, create positive and negative word clouds
        for country, data in results['by_country'].items():
            # Create positive word cloud
            pos_words = data['positive_drivers']['words']
            pos_phrases = data['positive_drivers']['phrases']
            
            if pos_words or pos_phrases:
                # Combine words and phrases with weights
                pos_text = ' '.join([word for word in pos_words])
                for phrase in pos_phrases:
                    # Add the phrase multiple times to increase its weight
                    pos_text += ' ' + ' '.join([phrase] * 3)
                    
                if pos_text.strip():
                    # Generate word cloud
                    wordcloud = WordCloud(
                        width=800, height=400,
                        background_color='white',
                        colormap='Greens',
                        max_words=50,
                        collocations=True,
                        min_font_size=10
                    ).generate(pos_text)
                    
                    # Create figure
                    plt.figure(figsize=(10, 6))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis('off')
                    plt.title(f'Positive {theme_display} Drivers - {country}', fontsize=16)
                    plt.tight_layout()
                    
                    # Save figure
                    country_file = country.lower().replace(' ', '_')
                    plt.savefig(os.path.join(viz_dir, f'{category}_{theme}_{country_file}_positive_drivers.png'), dpi=300)
                    plt.close()
                    
            # Create negative word cloud
            neg_words = data['negative_drivers']['words']
            neg_phrases = data['negative_drivers']['phrases']
            
            if neg_words or neg_phrases:
                # Combine words and phrases with weights
                neg_text = ' '.join([word for word in neg_words])
                for phrase in neg_phrases:
                    # Add the phrase multiple times to increase its weight
                    neg_text += ' ' + ' '.join([phrase] * 3)
                    
                if neg_text.strip():
                    # Generate word cloud
                    wordcloud = WordCloud(
                        width=800, height=400,
                        background_color='white',
                        colormap='Reds',
                        max_words=50,
                        collocations=True,
                        min_font_size=10
                    ).generate(neg_text)
                    
                    # Create figure
                    plt.figure(figsize=(10, 6))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis('off')
                    plt.title(f'Negative {theme_display} Drivers - {country}', fontsize=16)
                    plt.tight_layout()
                    
                    # Save figure
                    country_file = country.lower().replace(' ', '_')
                    plt.savefig(os.path.join(viz_dir, f'{category}_{theme}_{country_file}_negative_drivers.png'), dpi=300)
                    plt.close()
                    
    def _create_subtheme_analysis(self, category, theme, results):
        """
        Create subtheme analysis visualizations.
        
        Parameters:
        - category: Category name
        - theme: Theme name
        - results: Results dictionary for the theme
        """
        if 'by_subtheme' not in results or not results['by_subtheme']:
            return
            
        # Format for display
        theme_display = theme.replace('_', ' ').title()
        category_display = category.title()
        
        viz_dir = os.path.join(self.output_dir, 'visualizations')
        
        # Create a heatmap of subtheme sentiment by country
        # Prepare data for heatmap
        heatmap_data = []
        
        # Get all subthemes across all countries
        all_subthemes = set()
        for country, subthemes in results['by_subtheme'].items():
            all_subthemes.update(subthemes.keys())
            
        for country, subthemes in results['by_subtheme'].items():
            for subtheme in all_subthemes:
                if subtheme in subthemes:
                    subtheme_display = subtheme.replace('_', ' ').title()
                    heatmap_data.append({
                        'Country': country,
                        'Subtheme': subtheme_display,
                        'Sentiment': subthemes[subtheme]['avg_sentiment'],
                        'Mentions': subthemes[subtheme]['mention_count']
                    })
                else:
                    subtheme_display = subtheme.replace('_', ' ').title()
                    heatmap_data.append({
                        'Country': country,
                        'Subtheme': subtheme_display,
                        'Sentiment': np.nan,
                        'Mentions': 0
                    })
                    
        if not heatmap_data:
            return
            
        # Convert to DataFrame
        heatmap_df = pd.DataFrame(heatmap_data)
        
        # Create sentiment heatmap
        pivot_df = heatmap_df.pivot(index='Country', columns='Subtheme', values='Sentiment')
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_df, annot=True, cmap='RdYlGn', center=0, fmt='.2f', cbar_kws={'label': 'Sentiment Score'})
        plt.title(f'{theme_display} Subtheme Sentiment by Country - {category_display}', fontsize=16)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(viz_dir, f'{category}_{theme}_subtheme_sentiment_heatmap.png'), dpi=300)
        plt.close()
        
        # Create mention percentage heatmap
        pivot_df = heatmap_df.pivot(index='Country', columns='Subtheme', values='Mentions')
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt='d', cbar_kws={'label': 'Mention Count'})
        plt.title(f'{theme_display} Subtheme Mentions by Country - {category_display}', fontsize=16)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(viz_dir, f'{category}_{theme}_subtheme_mentions_heatmap.png'), dpi=300)
        plt.close()
        
        # For each country, create a bar chart of subtheme sentiment
        for country, subthemes in results['by_subtheme'].items():
            # Skip countries with too few subthemes
            if len(subthemes) < 2:
                continue
                
            # Prepare data
            subtheme_names = []
            sentiments = []
            mentions = []
            
            for subtheme, data in subthemes.items():
                subtheme_display = subtheme.replace('_', ' ').title()
                subtheme_names.append(subtheme_display)
                sentiments.append(data['avg_sentiment'])
                mentions.append(data['mention_count'])
                
            # Sort by sentiment
            sorted_data = sorted(zip(subtheme_names, sentiments, mentions), key=lambda x: x[1], reverse=True)
            subtheme_names, sentiments, mentions = zip(*sorted_data)
            
            # Create chart
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            # Define color map based on sentiment
            cmap = cm.get_cmap('RdYlGn')
            norm = Normalize(vmin=-0.5, vmax=0.5)  # Normalize sentiment to color range
            colors = [cmap(norm(s)) for s in sentiments]
            
            # Bar chart for sentiment
            bars = ax1.bar(subtheme_names, sentiments, color=colors, alpha=0.8)
            
            # Add sentiments on bars
            for bar, sentiment in zip(bars, sentiments):
                height = bar.get_height()
                ax1.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + 0.05 if height >= 0 else height - 0.1,
                    f'{height:.2f}',
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontweight='bold'
                )
                
            ax1.set_ylabel('Sentiment Score', fontsize=12)
            ax1.set_xlabel('Subtheme', fontsize=12)
            ax1.set_title(f'{theme_display} Subtheme Analysis - {country}', fontsize=16)
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.2)
            ax1.grid(axis='y', alpha=0.3)
            
            # Rotate x labels if there are many
            plt.xticks(rotation=45 if len(subtheme_names) > 3 else 0, ha='right' if len(subtheme_names) > 3 else 'center')
            
            # Line for mentions
            ax2 = ax1.twinx()
            ax2.plot(subtheme_names, mentions, 'ro-', linewidth=2, markersize=8)
            
            # Add mentions as text
            for i, mention in enumerate(mentions):
                ax2.text(i, mention + max(mentions)*0.05, str(mention), ha='center', va='bottom', color='red', fontweight='bold')
                
            ax2.set_ylabel('Mention Count', color='red', fontsize=12)
            ax2.tick_params(axis='y', labelcolor='red')
            
            plt.tight_layout()
            
            # Save figure
            country_file = country.lower().replace(' ', '_')
            plt.savefig(os.path.join(viz_dir, f'{category}_{theme}_{country_file}_subtheme_analysis.png'), dpi=300)
            plt.close()
            
    def analyze_category_by_country(self, category, country=None):
        """
        Analyze all themes in a category for a specific country or all target countries.
        
        Parameters:
        - category: Category to analyze
        - country: Country to analyze (if None, analyze all target countries)
        
        Returns:
        - Dictionary with analysis results
        """
        return self.analyze_theme_drivers(category=category, country=country)
        
    def generate_key_findings(self, results):
        """
        Generate key findings from the analysis results.
        
        Parameters:
        - results: Analysis results dictionary
        
        Returns:
        - List of key findings
        """
        findings = []
        
        # Process each category
        for category, category_results in results.items():
            # Format category name
            category_display = category.title()
            
            # Process each theme
            for theme, theme_results in category_results.items():
                # Format theme name
                theme_display = theme.replace('_', ' ').title()
                
                # Skip themes with too little data
                if 'by_country' not in theme_results or len(theme_results['by_country']) < 2:
                    continue
                    
                # Get sentiment range
                countries_sentiment = [(country, data['avg_sentiment']) 
                                     for country, data in theme_results['by_country'].items()]
                
                # Sort by sentiment
                countries_sentiment.sort(key=lambda x: x[1], reverse=True)
                
                # Add finding about highest and lowest sentiment
                if len(countries_sentiment) >= 2:
                    highest_country, highest_sentiment = countries_sentiment[0]
                    lowest_country, lowest_sentiment = countries_sentiment[-1]
                    
                    if highest_sentiment - lowest_sentiment > 0.2:
                        findings.append({
                            'category': category,
                            'theme': theme,
                            'finding': f"For {theme_display} in {category_display}, visitors from {highest_country} " +
                                      f"are most positive ({highest_sentiment:.2f}) while those from {lowest_country} " +
                                      f"are least positive ({lowest_sentiment:.2f})"
                        })
                        
                        # Add reasons if drivers are available
                        if 'positive_drivers' in theme_results['by_country'][highest_country]:
                            pos_words = theme_results['by_country'][highest_country]['positive_drivers']['words']
                            if pos_words:
                                findings.append({
                                    'category': category,
                                    'theme': theme,
                                    'finding': f"{highest_country} visitors highlight these positive aspects of {theme_display}: " +
                                              f"{', '.join(pos_words[:5])}"
                                })
                                
                        if 'negative_drivers' in theme_results['by_country'][lowest_country]:
                            neg_words = theme_results['by_country'][lowest_country]['negative_drivers']['words']
                            if neg_words:
                                findings.append({
                                    'category': category,
                                    'theme': theme,
                                    'finding': f"{lowest_country} visitors mention these negative aspects of {theme_display}: " +
                                              f"{', '.join(neg_words[:5])}"
                                })
                                
                # Add subtheme findings
                if 'by_subtheme' in theme_results and theme_results['by_subtheme']:
                    for country, subthemes in theme_results['by_subtheme'].items():
                        # Skip countries with too few subthemes
                        if len(subthemes) < 2:
                            continue
                            
                        # Find highest and lowest sentiment subthemes
                        subtheme_sentiments = [(subtheme, data['avg_sentiment']) 
                                             for subtheme, data in subthemes.items()]
                        
                        # Sort by sentiment
                        subtheme_sentiments.sort(key=lambda x: x[1], reverse=True)
                        
                        if len(subtheme_sentiments) >= 2:
                            highest_subtheme, highest_sub_sentiment = subtheme_sentiments[0]
                            lowest_subtheme, lowest_sub_sentiment = subtheme_sentiments[-1]
                            
                            if highest_sub_sentiment - lowest_sub_sentiment > 0.3:
                                highest_display = highest_subtheme.replace('_', ' ').title()
                                lowest_display = lowest_subtheme.replace('_', ' ').title()
                                
                                findings.append({
                                    'category': category,
                                    'theme': theme,
                                    'finding': f"For {country} visitors, the {highest_display} aspect of {theme_display} " +
                                              f"is rated highest ({highest_sub_sentiment:.2f}), while {lowest_display} " +
                                              f"is rated lowest ({lowest_sub_sentiment:.2f})"
                                })
                                
        return findings
            
def run_theme_sentiment_drivers_analysis():
    """
    Run the theme sentiment drivers analysis.
    """
    # Set up paths
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(parent_dir, 'tonga_data')
    output_dir = os.path.join(parent_dir, 'outputs', 'theme_sentiment_drivers')
    
    # Initialize analyzer
    analyzer = ThemeSentimentDriverAnalyzer(data_dir=data_dir, output_dir=output_dir)
    
    # Load data
    if analyzer.load_data():
        # Run analysis by category
        results = {}
        
        # Get category from command line arguments if provided
        import sys
        if len(sys.argv) > 1:
            categories_to_analyze = [sys.argv[1]]
        else:
            # Default to analyzing attractions
            categories_to_analyze = ['attraction']
        
        # Optionally get specific theme if provided
        specific_theme = None
        if len(sys.argv) > 2:
            specific_theme = sys.argv[2]
            
        for category in categories_to_analyze:
            print(f"\nAnalyzing {category} sentiment drivers...")
            if specific_theme:
                print(f"Focusing on the {specific_theme} theme...")
                category_results = analyzer.analyze_theme_drivers(category=category, theme=specific_theme)
            else:
                category_results = analyzer.analyze_category_by_country(category)
            results[category] = category_results
            
        # Generate key findings
        findings = analyzer.generate_key_findings(results)
        
        # Print key findings
        print("\n=== Key Findings on Theme Sentiment Drivers ===")
        if findings:
            for finding in findings:
                print(f"- {finding['finding']}")
        else:
            print("No significant findings detected. Check the visualizations and JSON output for detailed insights.")
            
        print(f"\nDetailed results and visualizations saved to: {output_dir}")
        
        # Provide some examples from the data to help interpret the results
        print("\nSample insights from the analysis:")
        
        for category in results:
            for theme in results[category]:
                print(f"\nTheme: {theme.replace('_', ' ').title()} in {category.title()}")
                
                # Check if we have country-specific data
                if 'by_country' in results[category][theme]:
                    # Get a sample country that has examples
                    sample_countries = list(results[category][theme]['by_country'].keys())
                    
                    if sample_countries:
                        # Show information for first available country
                        sample_country = sample_countries[0]
                        country_data = results[category][theme]['by_country'][sample_country]
                        
                        # Print sentiment score
                        if 'avg_sentiment' in country_data:
                            print(f"{sample_country} sentiment score: {country_data['avg_sentiment']:.2f}")
                        
                        # Show positive drivers
                        if 'positive_drivers' in country_data and country_data['positive_drivers']['words']:
                            pos_words = country_data['positive_drivers']['words']
                            print(f"{sample_country} positive aspects: {', '.join(pos_words[:5])}")
                            
                        # Show negative drivers
                        if 'negative_drivers' in country_data and country_data['negative_drivers']['words']:
                            neg_words = country_data['negative_drivers']['words']
                            print(f"{sample_country} negative aspects: {', '.join(neg_words[:5])}")
                            
                        # Show an example quote if available
                        if 'examples' in country_data and 'positive' in country_data['examples'] and country_data['examples']['positive']:
                            print(f"Example positive comment: \"{country_data['examples']['positive'][0]}\"")
                
                # Check if we have subtheme analysis
                if 'by_subtheme' in results[category][theme]:
                    print("\nSubtheme analysis available. Check the visualizations for detailed breakdown of specific aspects.")
                
                # Only show one theme's examples to avoid overwhelming output
                break
        
    else:
        print("Analysis failed due to data loading issues.")

if __name__ == "__main__":
    run_theme_sentiment_drivers_analysis()