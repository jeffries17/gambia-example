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
from review_analyzer import TongaReviewAnalyzer

class ThematicNationalityAnalyzer:
    """
    Analyzes visitor experiences by nationality with a focus on thematic sentiment -
    extracting key themes and sentiment by nationality across different categories.
    """
    
    def __init__(self, data_dir='tonga_data', output_dir='outputs/thematic_nationality'):
        """
        Initialize the thematic nationality analyzer.
        
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
        for category in ['accommodations', 'restaurants', 'attractions', 'themes', 'visualizations']:
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
        
    def _preprocess_text(self, text):
        """
        Preprocess text for thematic analysis.
        
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
        
    def _get_theme_sentiment(self, review_text, theme_keywords):
        """
        Extract sentiment for a specific theme from review text.
        
        Parameters:
        - review_text: The full review text
        - theme_keywords: List of keywords related to the theme
        
        Returns:
        - Tuple of (sentiment_score, mentions)
        """
        if pd.isna(review_text) or review_text == '':
            return (0, 0)
            
        review_text = str(review_text).lower()
        sentences = sent_tokenize(review_text)
        
        theme_sentiments = []
        mention_count = 0
        
        # Check each sentence for theme keywords
        for sentence in sentences:
            if any(keyword in sentence for keyword in theme_keywords):
                mention_count += 1
                blob = TextBlob(sentence)
                theme_sentiments.append(blob.sentiment.polarity)
                
        # If theme is mentioned, return average sentiment and mention count
        if theme_sentiments:
            return (np.mean(theme_sentiments), mention_count)
        else:
            return (0, 0)
            
    def analyze_themes_by_nationality(self):
        """
        Analyze themes by nationality across different categories.
        """
        if self.reviews_df is None:
            print("No data loaded. Please run load_data() first.")
            return {}
            
        print("\nAnalyzing themes by nationality...")
        
        # Filter to include only target countries
        analysis_df = self.reviews_df[
            self.reviews_df['country'].isin(self.target_countries)
        ].copy()
        
        # Results structure
        results = {
            'by_category': {}
        }
        
        # Process each category
        for category, themes in self.themes.items():
            # Filter reviews for this category
            category_df = analysis_df[analysis_df['category'] == category].copy()
            if len(category_df) == 0:
                continue
                
            results['by_category'][category] = self._analyze_themes_for_category(category_df, category, themes)
            
        # Save results to JSON
        with open(os.path.join(self.output_dir, 'thematic_nationality_analysis.json'), 'w') as f:
            json.dump(results, f, indent=4)
            
        # Generate visualizations
        self._generate_visualizations(results)
        
        return results
        
    def _analyze_themes_for_category(self, df, category, themes):
        """
        Analyze themes for a specific category.
        
        Parameters:
        - df: DataFrame filtered for the category
        - category: Category name
        - themes: Dictionary of themes and their keywords
        
        Returns:
        - Dictionary with theme analysis results
        """
        print(f"Analyzing themes for {category}...")
        
        # Category results structure
        category_results = {
            'themes': {},
            'findings': []
        }
        
        # Process each theme
        for theme, keywords in themes.items():
            theme_results = self._analyze_theme(df, theme, keywords)
            category_results['themes'][theme] = theme_results
            
            # Generate key findings for this theme
            self._add_theme_findings(category_results['findings'], theme, theme_results, category)
            
        # Save category-level data
        category_dir = os.path.join(self.output_dir, f"{category}s")
        # Save theme_results data frame
        with open(os.path.join(category_dir, f"theme_analysis_by_nationality.json"), 'w') as f:
            json.dump(category_results, f, indent=4)
            
        return category_results
        
    def _analyze_theme(self, df, theme, keywords):
        """
        Analyze a specific theme across nationalities.
        
        Parameters:
        - df: DataFrame filtered for the category
        - theme: Theme name
        - keywords: List of keywords for the theme
        
        Returns:
        - Dictionary with theme analysis results
        """
        # Initialize results
        theme_results = {
            'by_nationality': {},
            'overall': {
                'avg_sentiment': 0,
                'mention_count': 0,
                'mention_percentage': 0
            }
        }
        
        # Extract theme sentiment for each review
        sentiments = []
        mentions = []
        
        for _, row in df.iterrows():
            sent, ment = self._get_theme_sentiment(row['text'], keywords)
            sentiments.append(sent)
            mentions.append(ment > 0)
            
        # Add theme sentiment and mention columns to df
        df['theme_sentiment'] = sentiments
        df['theme_mentioned'] = mentions
        
        # Calculate overall stats
        if len(df) > 0:
            mentioned_df = df[df['theme_mentioned']]
            if len(mentioned_df) > 0:
                theme_results['overall']['avg_sentiment'] = mentioned_df['theme_sentiment'].mean()
                theme_results['overall']['mention_count'] = len(mentioned_df)
                theme_results['overall']['mention_percentage'] = len(mentioned_df) / len(df) * 100
            
        # Analyze by nationality
        for nationality in self.target_countries:
            nat_df = df[df['country'] == nationality]
            if len(nat_df) == 0:
                continue
                
            nat_mentioned = nat_df[nat_df['theme_mentioned']]
            
            if len(nat_mentioned) > 0:
                avg_sentiment = nat_mentioned['theme_sentiment'].mean()
                mention_count = len(nat_mentioned)
                mention_percentage = mention_count / len(nat_df) * 100
                
                # Extract example quotes for extreme sentiments
                positive_examples = []
                negative_examples = []
                
                if len(nat_mentioned) >= 3:
                    # Sort by sentiment and get extremes
                    pos_df = nat_mentioned.sort_values('theme_sentiment', ascending=False).head(3)
                    neg_df = nat_mentioned.sort_values('theme_sentiment').head(3)
                    
                    # Extract positive quotes
                    for _, row in pos_df.iterrows():
                        if row['theme_sentiment'] > 0:
                            positive_examples.append({
                                'text': row['text'][:200] + "..." if len(row['text']) > 200 else row['text'],
                                'sentiment': row['theme_sentiment']
                            })
                    
                    # Extract negative quotes
                    for _, row in neg_df.iterrows():
                        if row['theme_sentiment'] < 0:
                            negative_examples.append({
                                'text': row['text'][:200] + "..." if len(row['text']) > 200 else row['text'],
                                'sentiment': row['theme_sentiment']
                            })
                
                # Store results
                theme_results['by_nationality'][nationality] = {
                    'avg_sentiment': avg_sentiment,
                    'mention_count': mention_count,
                    'mention_percentage': mention_percentage,
                    'positive_examples': positive_examples[:2],  # Limit to 2 examples
                    'negative_examples': negative_examples[:2]   # Limit to 2 examples
                }
                
        return theme_results
        
    def _add_theme_findings(self, findings, theme, theme_results, category):
        """
        Generate key findings for a theme.
        
        Parameters:
        - findings: List to add findings to
        - theme: Theme name
        - theme_results: Theme analysis results
        - category: Category name
        
        Returns:
        - None (modifies findings list in place)
        """
        nat_results = theme_results['by_nationality']
        
        if len(nat_results) < 2:
            return
            
        # Find highest and lowest sentiment nationalities
        sorted_by_sentiment = sorted(
            nat_results.items(),
            key=lambda x: x[1]['avg_sentiment'],
            reverse=True
        )
        
        highest_sentiment_nat = sorted_by_sentiment[0][0]
        highest_sentiment = sorted_by_sentiment[0][1]['avg_sentiment']
        
        lowest_sentiment_nat = sorted_by_sentiment[-1][0]
        lowest_sentiment = sorted_by_sentiment[-1][1]['avg_sentiment']
        
        # Format theme name for display
        display_theme = theme.replace('_', ' ').title()
        
        # Add findings if there's a significant difference
        if highest_sentiment - lowest_sentiment > 0.3:
            findings.append({
                'type': 'sentiment_gap',
                'theme': theme,
                'finding': f"Visitors from {highest_sentiment_nat} are most positive about {display_theme} " + 
                           f"({highest_sentiment:.2f}), while those from {lowest_sentiment_nat} " +
                           f"are least positive ({lowest_sentiment:.2f})"
            })
            
        # Find most frequently mentioned
        sorted_by_mention = sorted(
            nat_results.items(),
            key=lambda x: x[1]['mention_percentage'],
            reverse=True
        )
        
        most_mentioned_nat = sorted_by_mention[0][0]
        most_mentioned_pct = sorted_by_mention[0][1]['mention_percentage']
        
        findings.append({
            'type': 'most_mentioned',
            'theme': theme,
            'finding': f"Visitors from {most_mentioned_nat} mention {display_theme} most frequently " +
                       f"({most_mentioned_pct:.1f}% of reviews)"
        })
        
        # Check for strongly positive or negative sentiment
        if highest_sentiment > 0.5:
            findings.append({
                'type': 'strong_positive',
                'theme': theme,
                'finding': f"Visitors from {highest_sentiment_nat} are very positive about {display_theme} " +
                           f"({highest_sentiment:.2f})"
            })
            
        if lowest_sentiment < -0.2:
            findings.append({
                'type': 'strong_negative',
                'theme': theme,
                'finding': f"Visitors from {lowest_sentiment_nat} are negative about {display_theme} " +
                           f"({lowest_sentiment:.2f})"
            })
            
    def _generate_visualizations(self, results):
        """
        Generate visualizations for thematic nationality analysis.
        
        Parameters:
        - results: Dictionary with analysis results
        """
        print("\nGenerating theme visualizations...")
        
        # Create visualization directory
        viz_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # For each category, create visualizations
        for category, category_data in results['by_category'].items():
            self._create_category_theme_visualizations(category, category_data, viz_dir)
            
        print(f"Visualizations saved to {viz_dir}")
        
    def _create_category_theme_visualizations(self, category, category_data, viz_dir):
        """
        Create visualizations for themes within a category.
        
        Parameters:
        - category: Category name
        - category_data: Category analysis data
        - viz_dir: Visualization output directory
        """
        # Category title for display
        category_title = category.title()
        
        # Theme data
        themes_data = category_data['themes']
        
        # 1. Create a heatmap of theme sentiment by nationality
        self._create_theme_sentiment_heatmap(category, themes_data, viz_dir)
        
        # 2. Create theme mention percentage by nationality
        self._create_theme_mention_chart(category, themes_data, viz_dir)
        
        # 3. Create individual theme charts for each nationality
        self._create_nationality_theme_charts(category, themes_data, viz_dir)
        
        # 4. Create nationality comparison for each theme
        self._create_theme_nationality_charts(category, themes_data, viz_dir)
        
    def _create_theme_sentiment_heatmap(self, category, themes_data, viz_dir):
        """
        Create a heatmap of theme sentiment by nationality.
        
        Parameters:
        - category: Category name
        - themes_data: Theme analysis data
        - viz_dir: Visualization output directory
        """
        # Prepare data for heatmap
        heatmap_data = []
        
        for theme, theme_data in themes_data.items():
            # Skip themes with too little data
            if len(theme_data['by_nationality']) < 2:
                continue
                
            for nationality, nat_data in theme_data['by_nationality'].items():
                heatmap_data.append({
                    'Theme': theme.replace('_', ' ').title(),
                    'Nationality': nationality,
                    'Sentiment': nat_data['avg_sentiment']
                })
                
        if not heatmap_data:
            return
            
        # Convert to DataFrame
        heatmap_df = pd.DataFrame(heatmap_data)
        
        # Pivot for heatmap
        pivot_df = heatmap_df.pivot(index='Nationality', columns='Theme', values='Sentiment')
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_df, annot=True, cmap='RdYlGn', center=0, fmt='.2f')
        plt.title(f'Theme Sentiment by Nationality - {category.title()}', fontsize=16, pad=20)
        plt.tight_layout()
        
        # Save
        plt.savefig(os.path.join(viz_dir, f'{category}_theme_sentiment_heatmap.png'), dpi=300)
        plt.close()
        
    def _create_theme_mention_chart(self, category, themes_data, viz_dir):
        """
        Create chart showing theme mention percentage by nationality.
        
        Parameters:
        - category: Category name
        - themes_data: Theme analysis data
        - viz_dir: Visualization output directory
        """
        # Prepare data for chart
        mention_data = []
        
        for theme, theme_data in themes_data.items():
            # Skip themes with too little data
            if len(theme_data['by_nationality']) < 2:
                continue
                
            for nationality, nat_data in theme_data['by_nationality'].items():
                mention_data.append({
                    'Theme': theme.replace('_', ' ').title(),
                    'Nationality': nationality,
                    'Mention_Percentage': nat_data['mention_percentage']
                })
                
        if not mention_data:
            return
            
        # Convert to DataFrame
        mention_df = pd.DataFrame(mention_data)
        
        # Pivot for heatmap
        pivot_df = mention_df.pivot(index='Nationality', columns='Theme', values='Mention_Percentage')
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt='.1f')
        plt.title(f'Theme Mention Percentage by Nationality - {category.title()}', fontsize=16, pad=20)
        plt.tight_layout()
        
        # Save
        plt.savefig(os.path.join(viz_dir, f'{category}_theme_mention_heatmap.png'), dpi=300)
        plt.close()
        
    def _create_nationality_theme_charts(self, category, themes_data, viz_dir):
        """
        Create charts showing theme sentiment for each nationality.
        
        Parameters:
        - category: Category name
        - themes_data: Theme analysis data
        - viz_dir: Visualization output directory
        """
        # Group data by nationality
        nationality_data = defaultdict(list)
        
        for theme, theme_data in themes_data.items():
            # Format theme name
            theme_display = theme.replace('_', ' ').title()
            
            for nationality, nat_data in theme_data['by_nationality'].items():
                nationality_data[nationality].append({
                    'Theme': theme_display,
                    'Sentiment': nat_data['avg_sentiment'],
                    'Mention_Percentage': nat_data['mention_percentage']
                })
                
        # Create a chart for each nationality
        for nationality, data_list in nationality_data.items():
            if len(data_list) < 2:
                continue
                
            # Convert to DataFrame
            nat_df = pd.DataFrame(data_list)
            
            # Sort by sentiment
            nat_df = nat_df.sort_values('Sentiment', ascending=False)
            
            # Create sentiment chart
            plt.figure(figsize=(12, 6))
            colors = []
            
            # Color bars based on sentiment
            for sentiment in nat_df['Sentiment']:
                if sentiment > 0.3:
                    colors.append('#4CAF50')  # Green
                elif sentiment > 0:
                    colors.append('#CDDC39')  # Lime
                elif sentiment > -0.3:
                    colors.append('#FFC107')  # Amber
                else:
                    colors.append('#F44336')  # Red
            
            bars = plt.bar(nat_df['Theme'], nat_df['Sentiment'], color=colors)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + 0.05 if height >= 0 else height - 0.1,
                    f'{height:.2f}',
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontweight='bold'
                )
                
            plt.title(f'Theme Sentiment for {nationality} Visitors - {category.title()}', fontsize=16, pad=20)
            plt.ylabel('Sentiment Score', fontsize=12)
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.2)
            plt.grid(axis='y', alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save
            filename = f'{category}_{nationality.lower().replace(" ", "_")}_theme_sentiment.png'
            plt.savefig(os.path.join(viz_dir, filename), dpi=300)
            plt.close()
            
    def _create_theme_nationality_charts(self, category, themes_data, viz_dir):
        """
        Create charts comparing nationalities for each theme.
        
        Parameters:
        - category: Category name
        - themes_data: Theme analysis data
        - viz_dir: Visualization output directory
        """
        # Process each theme
        for theme, theme_data in themes_data.items():
            # Skip themes with too little data
            if len(theme_data['by_nationality']) < 2:
                continue
                
            # Prepare data
            nat_list = []
            for nationality, nat_data in theme_data['by_nationality'].items():
                nat_list.append({
                    'Nationality': nationality,
                    'Sentiment': nat_data['avg_sentiment'],
                    'Mention_Percentage': nat_data['mention_percentage']
                })
                
            # Convert to DataFrame
            theme_df = pd.DataFrame(nat_list)
            
            # Sort by sentiment
            theme_df = theme_df.sort_values('Sentiment', ascending=False)
            
            # Create color list based on nationality
            colors = [self.country_colors.get(nat, '#BDBDBD') for nat in theme_df['Nationality']]
            
            # Create sentiment chart
            plt.figure(figsize=(12, 6))
            bars = plt.bar(theme_df['Nationality'], theme_df['Sentiment'], color=colors)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + 0.05 if height >= 0 else height - 0.1,
                    f'{height:.2f}',
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontweight='bold'
                )
                
            # Format theme for display
            theme_display = theme.replace('_', ' ').title()
            
            plt.title(f'{theme_display} Sentiment by Nationality - {category.title()}', fontsize=16, pad=20)
            plt.ylabel('Sentiment Score', fontsize=12)
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.2)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            # Save
            filename = f'{category}_{theme}_nationality_sentiment.png'
            plt.savefig(os.path.join(viz_dir, filename), dpi=300)
            plt.close()
            
            # Create mention percentage chart
            plt.figure(figsize=(12, 6))
            bars = plt.bar(theme_df['Nationality'], theme_df['Mention_Percentage'], color=colors)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + 1,
                    f'{height:.1f}%',
                    ha='center', va='bottom',
                    fontweight='bold'
                )
                
            plt.title(f'{theme_display} Mention Percentage by Nationality - {category.title()}', fontsize=16, pad=20)
            plt.ylabel('Mention Percentage', fontsize=12)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            # Save
            filename = f'{category}_{theme}_nationality_mention.png'
            plt.savefig(os.path.join(viz_dir, filename), dpi=300)
            plt.close()

def run_thematic_nationality_analysis():
    """
    Run the thematic nationality analysis.
    """
    # Set up paths
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(parent_dir, 'tonga_data')
    output_dir = os.path.join(parent_dir, 'outputs', 'thematic_nationality')
    
    # Initialize analyzer
    analyzer = ThematicNationalityAnalyzer(data_dir=data_dir, output_dir=output_dir)
    
    # Load data
    if analyzer.load_data():
        # Run analysis
        results = analyzer.analyze_themes_by_nationality()
        
        # Print key findings
        print("\nKey Thematic Findings by Nationality:")
        
        # Print category-specific findings
        for category, data in results['by_category'].items():
            print(f"\n{category.title()} Findings:")
            for finding in data['findings']:
                print(f"- {finding['finding']}")
                
        print(f"\nDetailed results and visualizations saved to: {output_dir}")
        
    else:
        print("Analysis failed due to data loading issues.")

if __name__ == "__main__":
    run_thematic_nationality_analysis()