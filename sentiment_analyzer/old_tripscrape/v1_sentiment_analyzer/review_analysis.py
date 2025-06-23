import pandas as pd
import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
from wordcloud import WordCloud
from datetime import datetime
import os
import numpy as np

class TongaTourismAnalysis:
    """
    A class to analyze TripAdvisor reviews for Tonga tourism planning.
    Now supports extensions for specialized analysis.
    """

    def __init__(self, output_dir='tonga_analysis_output'):
        """Initialize the analysis framework."""
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        self.stop_words = set(stopwords.words('english'))

        # Add common words in reviews that don't add analytical value
        custom_stopwords = {'hotel', 'resort', 'restaurant', 'place', 'stay', 'room', 'island', 'just', 'definitely', 'would'}
        self.stop_words.update(custom_stopwords)

        # Create output directory if it doesn't exist
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        # Define categories for theme extraction
        self.theme_categories = {
            'food_dining': ['food', 'meal', 'breakfast', 'dinner', 'lunch', 'cuisine', 'dish',
                            'restaurant', 'cafe', 'menu', 'chef', 'delicious', 'taste', 'flavor'],
            'accommodation': ['room', 'bed', 'accommodation', 'hotel', 'resort', 'villa',
                              'apartment', 'bathroom', 'shower', 'clean', 'spacious'],
            'service': ['service', 'staff', 'friendly', 'helpful', 'attentive', 'professional',
                        'welcoming', 'reception', 'manager', 'waitress', 'waiter'],
            'activities': ['tour', 'activity', 'swim', 'snorkel', 'dive', 'beach', 'hike',
                           'kayak', 'boat', 'cruise', 'excursion', 'trip', 'guide'],
            'value': ['price', 'value', 'worth', 'expensive', 'cheap', 'reasonable', 'cost',
                      'overpriced', 'budget', 'money'],
            'location': ['location', 'central', 'view', 'walking', 'distance', 'close', 'far',
                         'nearby', 'convenient', 'setting', 'surroundings', 'scenic'],
            'culture': ['culture', 'traditional', 'local', 'authentic', 'cultural', 'history',
                        'tradition', 'performance', 'dance', 'music', 'ceremony'],
            'natural_environment': ['beach', 'ocean', 'sea', 'water', 'nature', 'tropical',
                                    'scenic', 'beautiful', 'landscape', 'paradise', 'whale',
                                    'wildlife', 'coral', 'reef'],
            'infrastructure': ['wifi', 'internet', 'road', 'airport', 'transport', 'taxi',
                               'infrastructure', 'electricity', 'water', 'facilities']
        }

        # Add storage for extensions
        self.extensions = []

        # Add storage for processed data
        self.processed_data = None

    def get_sentiment(self, text):
        """Get sentiment from text using the transformer model."""
        if text.strip() == "":  # Check if the text is empty
            return {'label': 'NEUTRAL', 'score': 0.0}
        result = self.sentiment_analyzer(text)[0]
        return {'label': result['label'], 'score': result['score']}

    def add_processed_columns(self, df):
        """Add preprocessed text and sentiment columns to the DataFrame."""
        # Ensure the 'text' column exists and contains valid data
        if 'text' not in df.columns:
            print("Error: 'text' column not found in the data")
            df['text'] = ''
            df['processed_text'] = ''
            df['sentiment_label'] = 'NEUTRAL'
            df['sentiment_score'] = 0.0
            return df

        # Preprocess the review text
        print("Preprocessing review text...")
        df['processed_text'] = df['text'].apply(self.preprocess_text)

        # Extract sentiment scores
        print("Calculating sentiment scores...")
        df['sentiment_result'] = df['text'].apply(self.get_sentiment)
        df['sentiment_label'] = df['sentiment_result'].apply(lambda x: x['label'])
        df['sentiment_score'] = df['sentiment_result'].apply(lambda x: x['score'])

        return df

    def preprocess_text(self, text):
        """Clean and preprocess review text for analysis."""
        if not isinstance(text, str) or text.lower() == 'nan' or not text.strip():
            return ''

        # Convert to lowercase and remove non-alphabetic characters
        text = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())

        # Tokenize the text
        tokens = word_tokenize(text)

        # Remove stopwords and apply lemmatization
        filtered_tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words and len(word) > 2]

        return ' '.join(filtered_tokens)
    
    def extract_themes(self, df):
        """
        Extract themes from review text based on predefined categories.
        
        Parameters:
        - df: DataFrame with review data including processed text
        
        Returns:
        - DataFrame with theme detection columns
        """
        print("Extracting themes from reviews...")
        
        # For each theme category, check if any keywords are present in the processed text
        for theme, keywords in self.theme_categories.items():
            pattern = r'\b(' + '|'.join(keywords) + r')\b'
            df[f'theme_{theme}'] = df['processed_text'].astype(str).str.contains(pattern, case=False, regex=True).astype(int)
        
        # Add a column for the primary theme of each review
        theme_columns = [f'theme_{theme}' for theme in self.theme_categories.keys()]
        
        # Handle cases where no themes are detected
        if df[theme_columns].sum(axis=1).min() == 0:
            # For rows with no themes, set a default "unclassified" theme
            df['theme_sum'] = df[theme_columns].sum(axis=1)
            df.loc[df['theme_sum'] == 0, 'theme_food_dining'] = 1  # Default to food as a fallback
        
        df['primary_theme'] = df[theme_columns].idxmax(axis=1)
        df['primary_theme'] = df['primary_theme'].str.replace('theme_', '')
        
        # Count mentions of each theme
        print("Theme distribution:")
        theme_counts = df[theme_columns].sum().sort_values(ascending=False)
        for theme, count in theme_counts.items():
            theme_name = theme.replace('theme_', '')
            print(f"  {theme_name}: {count} mentions")
        
        return df
    
    def analyze_ratings_by_theme(self, df):
        """
        Analyze how ratings vary by different themes.
        
        Parameters:
        - df: DataFrame with review data including themes
        
        Returns:
        - DataFrame with aggregated theme ratings
        """
        print("\nAnalyzing ratings by theme...")
        
        # Make sure rating is numeric
        if 'rating' not in df.columns:
            print("Rating column not found, cannot analyze ratings by theme")
            return None
        
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        
        # Group by primary theme and calculate average rating
        theme_ratings = df.groupby('primary_theme')['rating'].agg(['mean', 'count']).reset_index()
        theme_ratings = theme_ratings.sort_values('mean', ascending=False)
        
        print(theme_ratings)
        
        # Create a bar chart of average ratings by theme
        plt.figure(figsize=(12, 6))
        # Fix: Use hue instead of palette to avoid warning
        ax = sns.barplot(x='primary_theme', y='mean', hue='primary_theme', data=theme_ratings, legend=False)
        plt.title('Average Rating by Primary Theme')
        plt.xlabel('Theme')
        plt.ylabel('Average Rating')
        plt.ylim(1, 5)
        
        # Add count labels
        for i, row in enumerate(theme_ratings.itertuples()):
            ax.text(i, row.mean - 0.2, f'n={row.count}', ha='center', color='white', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/ratings_by_theme.png')
        print(f"Saved chart to {self.output_dir}/ratings_by_theme.png")
        
        return theme_ratings
    
    def analyze_by_trip_type(self, df):
        """
        Analyze reviews by trip type to identify different traveler needs.
        
        Parameters:
        - df: DataFrame with review data including trip type
        
        Returns:
        - DataFrame with aggregated trip type data
        """
        if 'trip_type_standard' not in df.columns:
            print("Trip type information not available")
            return None
        
        print("\nAnalyzing reviews by trip type...")
        
        # Make sure rating is numeric
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        
        # Calculate the average rating by trip type
        trip_type_ratings = df.groupby('trip_type_standard')['rating'].agg(['mean', 'count']).reset_index()
        trip_type_ratings = trip_type_ratings.sort_values('mean', ascending=False)
        
        print(trip_type_ratings)
        
        # Analyze theme preferences by trip type
        theme_columns = [f'theme_{theme}' for theme in self.theme_categories.keys()]
        trip_type_themes = df.groupby('trip_type_standard')[theme_columns].mean().reset_index()
        
        # Create a heatmap of theme prevalence by trip type
        plt.figure(figsize=(14, 8))
        pivot_df = trip_type_themes.set_index('trip_type_standard')
        pivot_df.columns = [col.replace('theme_', '') for col in pivot_df.columns]
        sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt='.2f')
        plt.title('Theme Prevalence by Trip Type')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/themes_by_trip_type.png')
        print(f"Saved chart to {self.output_dir}/themes_by_trip_type.png")
        
        return trip_type_ratings
    
    def analyze_sentiment_patterns(self, df):
        """
        Analyze sentiment patterns in the reviews.
        
        Parameters:
        - df: DataFrame with review data including sentiment scores
        
        Returns:
        - DataFrame with aggregated sentiment data
        """
        print("\nAnalyzing sentiment patterns...")
        
        # Distribution of sentiment categories
        sentiment_dist = df['sentiment_category'].value_counts().reset_index()
        sentiment_dist.columns = ['sentiment', 'count']
        
        print(sentiment_dist)
        
        # Create a pie chart of sentiment distribution
        plt.figure(figsize=(10, 6))
        plt.pie(sentiment_dist['count'], labels=sentiment_dist['sentiment'], 
                autopct='%1.1f%%', startangle=90, colors=['#5CB85C', '#F0AD4E', '#D9534F'])
        plt.axis('equal')
        plt.title('Sentiment Distribution in Reviews')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/sentiment_distribution.png')
        print(f"Saved chart to {self.output_dir}/sentiment_distribution.png")
        
        # Analyze sentiment by theme
        theme_sentiment = df.groupby('primary_theme')['sentiment_score'].mean().reset_index()
        theme_sentiment = theme_sentiment.sort_values('sentiment_score', ascending=False)
        
        # Create a bar chart of sentiment by theme
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x='primary_theme', y='sentiment_score', data=theme_sentiment, palette='RdYlGn')
        plt.title('Average Sentiment Score by Primary Theme')
        plt.xlabel('Theme')
        plt.ylabel('Sentiment Score (-1 to 1)')
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/sentiment_by_theme.png')
        print(f"Saved chart to {self.output_dir}/sentiment_by_theme.png")
        
        return theme_sentiment
    
    def extract_key_phrases(self, df, min_count=3):
        """
        Extract common key phrases from reviews by sentiment category.
        
        Parameters:
        - df: DataFrame with review data
        - min_count: Minimum frequency for phrases to be included
        
        Returns:
        - Dictionary of key phrases by sentiment
        """
        print("\nExtracting key phrases by sentiment...")
        
        # Group by sentiment category
        key_phrases_by_sentiment = {}
        
        for sentiment in ['positive', 'negative', 'neutral']:
            # Filter reviews by sentiment
            filtered_reviews = df[df['sentiment_category'] == sentiment]
            
            if len(filtered_reviews) == 0:
                continue
                
            # Extract all words from processed text
            all_words = ' '.join(filtered_reviews['processed_text'].fillna('')).split()
            
            # Count word frequencies
            word_counts = Counter(all_words)
            common_words = {word: count for word, count in word_counts.items() if count >= min_count}
            
            # Sort by frequency
            sorted_words = sorted(common_words.items(), key=lambda x: x[1], reverse=True)
            key_phrases_by_sentiment[sentiment] = sorted_words[:30]  # Top 30 phrases
            
            print(f"\n{sentiment.capitalize()} key phrases:")
            for word, count in sorted_words[:10]:  # Print top 10
                print(f"  {word}: {count}")
                
            # Create word cloud for this sentiment
            if sorted_words:
                wordcloud = WordCloud(width=800, height=400, background_color='white', 
                                     max_words=100, contour_width=3, contour_color='steelblue')
                wordcloud.generate_from_frequencies(dict(sorted_words))
                
                plt.figure(figsize=(10, 6))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title(f'Common Words in {sentiment.capitalize()} Reviews')
                plt.tight_layout()
                plt.savefig(f'{self.output_dir}/wordcloud_{sentiment}.png')
                print(f"Saved word cloud to {self.output_dir}/wordcloud_{sentiment}.png")
        
        return key_phrases_by_sentiment
    
    def temporal_analysis(self, df):
        """
        Analyze changes in ratings and themes over time.
        
        Parameters:
        - df: DataFrame with review data including publication year
        
        Returns:
        - DataFrame with aggregated temporal data
        """
        if 'publication_year' not in df.columns or df['publication_year'].isna().all():
            print("Publication year information not available")
            return None
        
        print("\nAnalyzing trends over time...")
        
        # Ensure publication_year is numeric
        df['publication_year'] = pd.to_numeric(df['publication_year'], errors='coerce')
        
        # Filter to years with sufficient data
        year_counts = df['publication_year'].value_counts()
        valid_years = year_counts[year_counts >= 5].index
        
        if len(valid_years) <= 1:
            print("Not enough temporal data to analyze trends")
            return None
            
        yearly_df = df[df['publication_year'].isin(valid_years)]
        
        # Calculate yearly averages for ratings
        yearly_ratings = yearly_df.groupby('publication_year')['rating'].mean().reset_index()
        
        # Plot yearly ratings trend
        plt.figure(figsize=(12, 6))
        sns.lineplot(x='publication_year', y='rating', data=yearly_ratings, marker='o')
        plt.title('Average Rating Trend by Year')
        plt.xlabel('Year')
        plt.ylabel('Average Rating')
        plt.ylim(1, 5)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/rating_trend_by_year.png')
        print(f"Saved chart to {self.output_dir}/rating_trend_by_year.png")
        
        # Analyze theme evolution over time
        theme_columns = [f'theme_{theme}' for theme in self.theme_categories.keys()]
        yearly_themes = yearly_df.groupby('publication_year')[theme_columns].mean().reset_index()
        
        # Convert theme columns to more readable format
        yearly_themes_long = pd.melt(yearly_themes, id_vars=['publication_year'], 
                                    value_vars=theme_columns, 
                                    var_name='theme', value_name='prevalence')
        yearly_themes_long['theme'] = yearly_themes_long['theme'].str.replace('theme_', '')
        
        # Plot theme evolution
        plt.figure(figsize=(14, 8))
        sns.lineplot(x='publication_year', y='prevalence', hue='theme', 
                    data=yearly_themes_long, marker='o')
        plt.title('Theme Prevalence Over Time')
        plt.xlabel('Year')
        plt.ylabel('Theme Prevalence')
        plt.grid(True, alpha=0.3)
        plt.legend(title='Theme', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/theme_evolution.png')
        print(f"Saved chart to {self.output_dir}/theme_evolution.png")
        
        return yearly_ratings
    
    def generate_recommendations(self, df):
        """
        Generate tourism planning recommendations based on the analysis.
        
        Parameters:
        - df: DataFrame with review data and analysis
        
        Returns:
        - Dictionary with recommendations
        """
        print("\nGenerating recommendations for tourism planning...")
        
        recommendations = {}
        
        # Identify top rated and problematic themes
        try:
            theme_ratings = df.groupby('primary_theme')['rating'].mean().sort_values()
            bottom_themes = theme_ratings.head(3).index.tolist()
            top_themes = theme_ratings.tail(3).index.tolist()
            
            recommendations['areas_for_improvement'] = bottom_themes
            recommendations['strengths_to_leverage'] = top_themes
        except:
            print("Could not analyze theme ratings")
            recommendations['areas_for_improvement'] = []
            recommendations['strengths_to_leverage'] = []
        
        # Extract negative reviews for areas needing improvement
        negative_reviews = df[df['sentiment_category'] == 'negative']
        improvement_insights = {}
        
        for theme in bottom_themes:
            theme_negatives = negative_reviews[negative_reviews['primary_theme'] == theme]
            if len(theme_negatives) > 0:
                # Get the most common words in negative reviews for this theme
                words = ' '.join(theme_negatives['processed_text'].fillna('')).split()
                word_counts = Counter(words)
                common_issues = dict(word_counts.most_common(10))
                improvement_insights[theme] = common_issues
        
        recommendations['improvement_insights'] = improvement_insights
        
        # Extract positive insights to highlight in marketing
        positive_reviews = df[df['sentiment_category'] == 'positive']
        marketing_insights = {}
        
        for theme in top_themes:
            theme_positives = positive_reviews[positive_reviews['primary_theme'] == theme]
            if len(theme_positives) > 0:
                # Get the most common words in positive reviews for this theme
                words = ' '.join(theme_positives['processed_text'].fillna('')).split()
                word_counts = Counter(words)
                marketing_points = dict(word_counts.most_common(10))
                marketing_insights[theme] = marketing_points
        
        recommendations['marketing_insights'] = marketing_insights
        
        # Analyze visitor segment preferences if trip type data is available
        if 'trip_type_standard' in df.columns:
            segment_preferences = {}
            
            for trip_type in df['trip_type_standard'].unique():
                if pd.notna(trip_type) and trip_type != 'unknown':
                    segment_df = df[df['trip_type_standard'] == trip_type]
                    if len(segment_df) >= 5:  # Only analyze segments with enough data
                        # Calculate theme prevalence
                        theme_columns = [f'theme_{theme}' for theme in self.theme_categories.keys()]
                        theme_prevalence = segment_df[theme_columns].mean()
                        theme_prevalence.index = theme_prevalence.index.str.replace('theme_', '')
                        
                        # Identify top themes for this segment
                        top_segment_themes = theme_prevalence.nlargest(3)
                        segment_preferences[trip_type] = dict(top_segment_themes)
            
            recommendations['segment_preferences'] = segment_preferences
            
            # Generate targeted recommendations for product development
            product_recommendations = []
            
            for segment, preferences in segment_preferences.items():
                if preferences:
                    top_theme = list(preferences.keys())[0]
                    product_idea = f"Develop new {top_theme} experiences targeted at {segment} travelers"
                    product_recommendations.append(product_idea)
            
            recommendations['product_development_ideas'] = product_recommendations
        
        # Print key recommendations
        print("\nKey areas for improvement:")
        for area in recommendations['areas_for_improvement']:
            print(f"  - {area}")
        
        print("\nKey strengths to leverage in marketing:")
        for strength in recommendations['strengths_to_leverage']:
            print(f"  - {strength}")
        
        if 'segment_preferences' in recommendations:
            print("\nKey visitor segment preferences:")
            for segment, prefs in recommendations['segment_preferences'].items():
                if prefs:
                    top_theme = list(prefs.keys())[0]
                    print(f"  - {segment} travelers value {top_theme} the most")
        
        # Save recommendations as JSON
        with open(f'{self.output_dir}/recommendations.json', 'w', encoding='utf-8') as f:
            json.dump(recommendations, f, indent=2)
        print(f"Saved recommendations to {self.output_dir}/recommendations.json")
        
        return recommendations
    
    def run_extensions(self, df):
        """
        Run all registered extensions.
        
        Parameters:
        - df: DataFrame with review data
        
        Returns:
        - Dictionary with results from all extensions
        """
        extension_results = {}
        
        for extension in self.extensions:
            print(f"\nRunning extension: {extension.__class__.__name__}")
            try:
                result = extension.run_analysis(df)
                extension_results[extension.__class__.__name__] = result
            except Exception as e:
                print(f"Error in extension {extension.__class__.__name__}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        return extension_results
    
    def get_processed_data(self):
        """
        Return the processed DataFrame for use by extensions.
        
        Returns:
        - Pandas DataFrame with processed review data
        """
        return self.processed_data
    
    def register_extension(self, extension):
        """
        Register an extension to run with this analyzer.
        
        Parameters:
        - extension: An extension object with a run_analysis method
        """
        self.extensions.append(extension)
        print(f"Registered extension: {extension.__class__.__name__}")
        
    def run_full_analysis(self, json_file_path):
        """
        Run the complete analysis pipeline on the review data.
        
        Parameters:
        - json_file_path: Path to the JSON file containing reviews
        
        Returns:
        - DataFrame with complete analysis results
        """
        print(f"Starting full analysis of {json_file_path}...")
        
        # Load and prepare data
        df = self.load_data(json_file_path)
        if df is None or len(df) == 0:
            print("No valid data to analyze")
            return None
        
        try:
            # Run preprocessing and basic analysis
            df = self.add_processed_columns(df)
            df = self.extract_themes(df)
            
            # Store the processed data
            self.processed_data = df
            
            # Run various analyses
            self.analyze_ratings_by_theme(df)
            self.analyze_by_trip_type(df)
            self.analyze_sentiment_patterns(df)
            self.extract_key_phrases(df)
            self.temporal_analysis(df)
            
            # Generate recommendations
            self.generate_recommendations(df)
            
            # Run any registered extensions
            if self.extensions:
                self.run_extensions(df)
            
            # Save processed data
            output_file = f'{self.output_dir}/analyzed_reviews.csv'
            df.to_csv(output_file, index=False)
            print(f"Analysis complete. Full results saved to {output_file}")
            
            return df
            
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return None