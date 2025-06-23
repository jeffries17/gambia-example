import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import numpy as np
import sys
import re

# Add the parent directory to the path so we can import sentiment_analyzer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from sentiment_analyzer import SentimentAnalyzer
except ImportError:
    # Try to import from the tonga_analysis module
    try:
        from tonga_analysis.sentiment_analyzer import SentimentAnalyzer
    except ImportError:
        print("Warning: Could not import SentimentAnalyzer module")
        
        # Define a minimal class for compatibility
        class SentimentAnalyzer:
            def __init__(self, output_dir=None):
                self.output_dir = output_dir
                
            def analyze_sentiment(self, df):
                print("Minimal sentiment analyzer fallback")
                return df
                
            def analyze_text_sentiment(self, text):
                from textblob import TextBlob
                if not isinstance(text, str):
                    return 0
                return TextBlob(text).sentiment.polarity
from tonga_analysis.accommodation_analyzer import AccommodationAnalyzer
from tonga_analysis.restaurant_analyzer import RestaurantAnalyzer
from tonga_analysis.attractions_analyzer import AttractionAnalyzer
from tonga_analysis.visualization_styles import (
    REGION_COLORS, SENTIMENT_COLORS, 
    set_visualization_style, get_regional_palette,
    apply_regional_style
)

class RegionalComparisonAnalyzer:
    """Analyzes and compares tourism data across multiple countries in the region."""
    
    def __init__(self, countries=None, base_dir="regional_data", output_dir=None):
        """
        Initialize the regional comparison analyzer.
        
        Args:
            countries (list): List of countries to include in the comparison
            base_dir (str): Base directory containing country data folders
            output_dir (str): Directory to save output files
        """
        self.countries = countries or ["tonga", "fiji", "samoa", "tahiti"]
        self.base_dir = base_dir
        
        # Set default output directory to the standardized location
        if output_dir is None:
            # Use the parent directory's outputs folder
            parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            self.output_dir = os.path.join(parent_dir, "outputs", "regional_comparison")
        else:
            self.output_dir = output_dir
            
        # Initialize sentiment analyzer with the standardized output path
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.sentiment_analyzer = SentimentAnalyzer(output_dir=os.path.join(parent_dir, "outputs"))
        
        # Create output directories
        self.create_output_dirs()
        
        # Store data by country and category
        self.data = {
            country: {
                "accommodations": None,
                "attractions": None,
                "restaurants": None
            } for country in self.countries
        }
        
        # Analysis results
        self.results = {
            "accommodations": {},
            "attractions": {},
            "restaurants": {},
            "overall": {}
        }
    
    def create_output_dirs(self):
        """Create output directories for regional comparison results."""
        # Main output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Category subdirectories
        for category in ["accommodations", "attractions", "restaurants", "overall"]:
            os.makedirs(f"{self.output_dir}/{category}", exist_ok=True)
            os.makedirs(f"{self.output_dir}/{category}/visualizations", exist_ok=True)
    
    def load_country_data(self):
        """Load review data for all countries."""
        for country in self.countries:
            country_dir = os.path.join(self.base_dir, country)
            
            # Skip if country directory doesn't exist
            if not os.path.exists(country_dir):
                # Try with lowercase directory name
                lowercase_dir = os.path.join(self.base_dir, country.lower())
                if os.path.exists(lowercase_dir):
                    country_dir = lowercase_dir
                    print(f"Found data directory for {country} at {country_dir}")
                else:
                    print(f"Warning: Data directory for {country} not found at {country_dir}")
                    continue
            
            # Try multiple filename patterns for accommodations
            accom_found = False
            # List all files in the directory to help with debugging
            try:
                files_in_dir = os.listdir(country_dir)
                print(f"Files in {country_dir}: {files_in_dir}")
            except Exception as e:
                print(f"Error listing files in {country_dir}: {e}")
                
            for filename_pattern in [
                f"{country.lower()}_accommodations.json",  # fiji_accommodations.json
                f"{country}_accommodations.json",          # Fiji_accommodations.json
                f"accommodations_{country.lower()}.json",  # accommodations_fiji.json
                f"accommodations_{country}.json"           # accommodations_Fiji.json
            ]:
                accom_path = os.path.join(country_dir, filename_pattern)
                print(f"Looking for: {accom_path}, exists: {os.path.exists(accom_path)}")
                if os.path.exists(accom_path):
                    print(f"Loading accommodations for {country} from {accom_path}")
                    try:
                        with open(accom_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            self.data[country]["accommodations"] = pd.DataFrame(data)
                        # Add country column
                        self.data[country]["accommodations"]["country"] = country
                        accom_found = True
                        break
                    except Exception as e:
                        print(f"Error loading {accom_path}: {e}")
                        continue
            
            if not accom_found:
                print(f"Warning: Could not find accommodations data for {country}")
            
            # Try multiple filename patterns for attractions
            attr_found = False
            for filename_pattern in [
                f"{country.lower()}_attractions.json",     # fiji_attractions.json
                f"{country}_attractions.json",             # Fiji_attractions.json
                f"attractions_{country.lower()}.json",     # attractions_fiji.json
                f"attractions_{country}.json"              # attractions_Fiji.json
            ]:
                attr_path = os.path.join(country_dir, filename_pattern)
                if os.path.exists(attr_path):
                    print(f"Loading attractions for {country} from {attr_path}")
                    with open(attr_path, 'r', encoding='utf-8') as f:
                        self.data[country]["attractions"] = pd.DataFrame(json.load(f))
                    # Add country column
                    self.data[country]["attractions"]["country"] = country
                    attr_found = True
                    break
            
            if not attr_found:
                print(f"Warning: Could not find attractions data for {country}")
            
            # Try multiple filename patterns for restaurants
            rest_found = False
            for filename_pattern in [
                f"{country.lower()}_restaurants.json",     # fiji_restaurants.json
                f"{country}_restaurants.json",             # Fiji_restaurants.json
                f"restaurants_{country.lower()}.json",     # restaurants_fiji.json
                f"restaurants_{country}.json"              # restaurants_Fiji.json
            ]:
                rest_path = os.path.join(country_dir, filename_pattern)
                if os.path.exists(rest_path):
                    print(f"Loading restaurants for {country} from {rest_path}")
                    with open(rest_path, 'r', encoding='utf-8') as f:
                        self.data[country]["restaurants"] = pd.DataFrame(json.load(f))
                    # Add country column
                    self.data[country]["restaurants"]["country"] = country
                    rest_found = True
                    break
            
            if not rest_found:
                print(f"Warning: Could not find restaurants data for {country}")
    
    def apply_sentiment_analysis(self):
        """Apply sentiment analysis to all reviews."""
        for country in self.countries:
            for category in ["accommodations", "attractions", "restaurants"]:
                if self.data[country][category] is not None:
                    df = self.data[country][category]
                    
                    # Apply sentiment analysis if review_text column exists
                    # First ensure we have a 'text' column that sentiment_analyzer expects
                    if "review_text" in df.columns:
                        print(f"Applying sentiment analysis to {country} {category}...")
                        # Create a text column if it doesn't exist
                        if 'text' not in df.columns:
                            df['text'] = df['review_text']
                        # Run the sentiment analysis
                        df = self.sentiment_analyzer.analyze_sentiment(df)
                        
                        # Extract common phrases for qualitative insights
                        phrases = self.sentiment_analyzer.extract_common_phrases(df)
                        
                        # Save extracted phrases for this country/category
                        df_phrases = {}
                        for sentiment_type, phrases_dict in phrases.items():
                            phrases_list = sorted(phrases_dict.items(), key=lambda x: x[1], reverse=True)
                            df_phrases[sentiment_type] = phrases_list[:20]  # Keep top 20 phrases
                        
                        # Add to dataframe for future analysis
                        df.attrs['common_phrases'] = df_phrases  # Store in dataframe attributes
                        
                        self.data[country][category] = df
    
    def combine_country_data(self):
        """Combine data from all countries into single dataframes by category."""
        combined_data = {
            "accommodations": [],
            "attractions": [],
            "restaurants": []
        }
        
        for country in self.countries:
            for category in ["accommodations", "attractions", "restaurants"]:
                if self.data[country][category] is not None:
                    combined_data[category].append(self.data[country][category])
        
        # Concatenate dataframes
        for category in combined_data:
            if combined_data[category]:
                combined_data[category] = pd.concat(combined_data[category], ignore_index=True)
            else:
                combined_data[category] = None
        
        return combined_data
    
    def analyze_accommodations(self):
        """Run accommodation analysis for all countries and create comparisons."""
        combined_data = self.combine_country_data()
        
        if combined_data["accommodations"] is None:
            print("No accommodation data available for comparison")
            return
        
        # Individual country analysis
        country_results = {}
        for country in self.countries:
            if self.data[country]["accommodations"] is not None:
                print(f"Analyzing {country} accommodations...")
                analyzer = AccommodationAnalyzer(sentiment_analyzer=self.sentiment_analyzer, 
                                               output_dir=f"{self.output_dir}/accommodations/{country.lower()}")
                analyzer_results = analyzer.run_analysis(self.data[country]["accommodations"])
                country_results[country] = analyzer_results
        
        # Create comparative visualizations
        self.compare_accommodation_types(combined_data["accommodations"])
        self.compare_accommodation_features(combined_data["accommodations"])
        self.compare_accommodation_ratings(combined_data["accommodations"])
        
        # Save results
        self.results["accommodations"] = {
            "country_results": country_results,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Convert numpy values to Python natives for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.core.series.Series):
                return obj.to_list()
            return obj
        
        # Clean results for JSON serialization
        clean_results = json.loads(
            json.dumps(self.results["accommodations"], default=convert_to_serializable)
        )
        
        with open(f"{self.output_dir}/accommodations/regional_comparison.json", 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, indent=4)
    
    def analyze_attractions(self):
        """Run attraction analysis for all countries and create comparisons."""
        combined_data = self.combine_country_data()
        
        if combined_data["attractions"] is None:
            print("No attraction data available for comparison")
            return
        
        # Individual country analysis
        country_results = {}
        for country in self.countries:
            if self.data[country]["attractions"] is not None:
                print(f"Analyzing {country} attractions...")
                analyzer = AttractionAnalyzer(sentiment_analyzer=self.sentiment_analyzer,
                                            output_dir=f"{self.output_dir}/attractions/{country.lower()}")
                analyzer_results = analyzer.run_analysis(self.data[country]["attractions"])
                country_results[country] = analyzer_results
        
        # Save initial results
        self.results["attractions"] = {
            "country_results": country_results,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Create comparative visualizations - this will also add activity_sentiment data
        self.compare_attraction_types(combined_data["attractions"])
        self.compare_attraction_experiences(combined_data["attractions"])
        self.compare_attraction_ratings(combined_data["attractions"])
        
        # Verify if activity_sentiment was added
        print("\nActivity sentiment analysis results:")
        for country in self.results["attractions"]["country_results"]:
            if "activity_sentiment" in self.results["attractions"]["country_results"][country]:
                print(f"  - {country}: Activity sentiment data available")
            else:
                print(f"  - {country}: No activity sentiment data")
                
            # Add empty activity_sentiment if missing
            if "activity_sentiment" not in self.results["attractions"]["country_results"][country]:
                self.results["attractions"]["country_results"][country]["activity_sentiment"] = {}
            if "aspect_sentiment" not in self.results["attractions"]["country_results"][country]:
                self.results["attractions"]["country_results"][country]["aspect_sentiment"] = {}
        
        # Convert numpy values to Python natives for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.core.series.Series):
                return obj.to_list()
            return obj
        
        # Clean results for JSON serialization
        clean_results = json.loads(
            json.dumps(self.results["attractions"], default=convert_to_serializable)
        )
        
        with open(f"{self.output_dir}/attractions/regional_comparison.json", 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, indent=4)
    
    def analyze_restaurants(self):
        """Run restaurant analysis for all countries and create comparisons."""
        combined_data = self.combine_country_data()
        
        if combined_data["restaurants"] is None:
            print("No restaurant data available for comparison")
            return
        
        # Individual country analysis
        country_results = {}
        for country in self.countries:
            if self.data[country]["restaurants"] is not None:
                print(f"Analyzing {country} restaurants...")
                analyzer = RestaurantAnalyzer(sentiment_analyzer=self.sentiment_analyzer,
                                            output_dir=f"{self.output_dir}/restaurants/{country.lower()}")
                analyzer_results = analyzer.run_analysis(self.data[country]["restaurants"])
                country_results[country] = analyzer_results
        
        # Create comparative visualizations
        self.compare_cuisine_types(combined_data["restaurants"])
        self.compare_dining_experience(combined_data["restaurants"])
        self.compare_restaurant_ratings(combined_data["restaurants"])
        
        # Save results
        self.results["restaurants"] = {
            "country_results": country_results,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Convert numpy values to Python natives for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.core.series.Series):
                return obj.to_list()
            return obj
        
        # Clean results for JSON serialization
        clean_results = json.loads(
            json.dumps(self.results["restaurants"], default=convert_to_serializable)
        )
        
        with open(f"{self.output_dir}/restaurants/regional_comparison.json", 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, indent=4)
    
    def compare_accommodation_types(self, df):
        """Compare accommodation types across countries."""
        if df is None or df.empty:
            return
        
        # Extract accommodation types for each country
        if 'accommodation_type' not in df.columns:
            print("Warning: accommodation_type column not found, skipping comparison")
            return
        
        plt.figure(figsize=(12, 8))
        count_df = df.groupby(['country', 'accommodation_type']).size().reset_index(name='count')
        
        # Calculate percentage within each country
        total_by_country = count_df.groupby('country')['count'].sum().reset_index()
        count_df = count_df.merge(total_by_country, on='country', suffixes=('', '_total'))
        count_df['percentage'] = count_df['count'] / count_df['count_total'] * 100
        
        # Plot
        sns.barplot(x='accommodation_type', y='percentage', hue='country', data=count_df)
        plt.title('Accommodation Types by Country (% of Total)')
        plt.xlabel('Accommodation Type')
        plt.ylabel('Percentage')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Country')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/accommodations/visualizations/accommodation_types_comparison.png")
        plt.close()
    
    def compare_accommodation_features(self, df):
        """Compare accommodation features across countries."""
        if df is None or df.empty:
            return
        
        # Extract common room and property features
        if 'room_features' not in df.columns or 'property_features' not in df.columns:
            print("Warning: feature columns not found, skipping comparison")
            return
        
        # Process room features
        room_features = {}
        for country in df['country'].unique():
            country_df = df[df['country'] == country]
            features = []
            for features_list in country_df['room_features'].dropna():
                if isinstance(features_list, list):
                    features.extend(features_list)
            
            # Get top 10 features
            feature_counts = pd.Series(features).value_counts().nlargest(10)
            room_features[country] = feature_counts
        
        # Create DataFrame for plotting
        room_data = []
        for country, features in room_features.items():
            for feature, count in features.items():
                room_data.append({
                    'country': country,
                    'feature': feature,
                    'count': count
                })
        
        if room_data:
            room_df = pd.DataFrame(room_data)
            
            plt.figure(figsize=(14, 8))
            
            # Only keep features that appear in at least 2 countries
            feature_countries = room_df.groupby('feature')['country'].nunique()
            common_features = feature_countries[feature_countries >= 2].index.tolist()
            if common_features:
                plot_df = room_df[room_df['feature'].isin(common_features)]
                
                # Convert to percentages within each country
                total_by_country = plot_df.groupby('country')['count'].sum().reset_index()
                plot_df = plot_df.merge(total_by_country, on='country', suffixes=('', '_total'))
                plot_df['percentage'] = plot_df['count'] / plot_df['count_total'] * 100
                
                # Plot
                sns.barplot(x='feature', y='percentage', hue='country', data=plot_df)
                plt.title('Common Room Features by Country (% of Total)')
                plt.xlabel('Feature')
                plt.ylabel('Percentage')
                plt.xticks(rotation=45, ha='right')
                plt.legend(title='Country')
                plt.tight_layout()
                plt.savefig(f"{self.output_dir}/accommodations/visualizations/room_features_comparison.png")
                plt.close()
        
        # Similar analysis for property features
        property_features = {}
        for country in df['country'].unique():
            country_df = df[df['country'] == country]
            features = []
            for features_list in country_df['property_features'].dropna():
                if isinstance(features_list, list):
                    features.extend(features_list)
            
            # Get top 10 features
            feature_counts = pd.Series(features).value_counts().nlargest(10)
            property_features[country] = feature_counts
        
        # Create DataFrame for plotting
        property_data = []
        for country, features in property_features.items():
            for feature, count in features.items():
                property_data.append({
                    'country': country,
                    'feature': feature,
                    'count': count
                })
        
        if property_data:
            property_df = pd.DataFrame(property_data)
            
            plt.figure(figsize=(14, 8))
            
            # Only keep features that appear in at least 2 countries
            feature_countries = property_df.groupby('feature')['country'].nunique()
            common_features = feature_countries[feature_countries >= 2].index.tolist()
            if common_features:
                plot_df = property_df[property_df['feature'].isin(common_features)]
                
                # Convert to percentages within each country
                total_by_country = plot_df.groupby('country')['count'].sum().reset_index()
                plot_df = plot_df.merge(total_by_country, on='country', suffixes=('', '_total'))
                plot_df['percentage'] = plot_df['count'] / plot_df['count_total'] * 100
                
                # Plot
                sns.barplot(x='feature', y='percentage', hue='country', data=plot_df)
                plt.title('Common Property Features by Country (% of Total)')
                plt.xlabel('Feature')
                plt.ylabel('Percentage')
                plt.xticks(rotation=45, ha='right')
                plt.legend(title='Country')
                plt.tight_layout()
                plt.savefig(f"{self.output_dir}/accommodations/visualizations/property_features_comparison.png")
                plt.close()
    
    def compare_accommodation_ratings(self, df):
        """Compare accommodation ratings across countries."""
        if df is None or df.empty:
            return
        
        # Compare rating distributions
        if 'rating' not in df.columns:
            print("Warning: rating column not found, skipping comparison")
            return
        
        # Apply consistent visualization style
        set_visualization_style()
        
        # Box plot with consistent colors
        plt.figure(figsize=(10, 6))
        # Use country-specific colors from our style guide
        palette = {country: REGION_COLORS.get(country.lower(), '#AAAAAA') 
                  for country in df['country'].unique()}
        
        ax = sns.boxplot(x='country', y='rating', data=df, palette=palette)
        apply_regional_style(ax, 
                            title='Accommodation Ratings by Country',
                            x_label='Country', 
                            y_label='Rating')
        plt.savefig(f"{self.output_dir}/accommodations/visualizations/rating_distribution_comparison.png", dpi=300)
        plt.close()
        
        # Bar chart with average ratings
        plt.figure(figsize=(10, 6))
        avg_ratings = df.groupby('country')['rating'].mean().reset_index()
        ax = sns.barplot(x='country', y='rating', data=avg_ratings, palette=palette)
        apply_regional_style(ax, 
                           title='Average Accommodation Ratings by Country',
                           x_label='Country', 
                           y_label='Average Rating')
        plt.ylim(0, 5)  # Assuming 5-star rating scale
        
        # Add data labels
        for i, row in enumerate(avg_ratings.itertuples()):
            plt.text(i, row.rating + 0.1, f'{row.rating:.2f}', ha='center', fontweight='bold')
            
        plt.savefig(f"{self.output_dir}/accommodations/visualizations/average_rating_comparison.png", dpi=300)
        plt.close()
        
        # Sentiment comparison if available
        if 'sentiment_score' in df.columns:
            # Filter out rows with null sentiment_score
            sentiment_df = df.dropna(subset=['sentiment_score'])
            if not sentiment_df.empty:  # Only create visualization if we have data
                plt.figure(figsize=(10, 6))
                avg_sentiment = sentiment_df.groupby('country')['sentiment_score'].mean().reset_index()
                
                ax = sns.barplot(x='country', y='sentiment_score', data=avg_sentiment, palette=palette)
                apply_regional_style(ax, 
                                   title='Average Sentiment Score for Accommodations by Country',
                                   x_label='Country', 
                                   y_label='Average Sentiment Score')
                
                # Add data labels
                for i, row in enumerate(avg_sentiment.itertuples()):
                    plt.text(i, row.sentiment_score + 0.02, f'{row.sentiment_score:.2f}', 
                             ha='center', fontweight='bold')
                    
                plt.savefig(f"{self.output_dir}/accommodations/visualizations/sentiment_comparison.png", dpi=300)
                plt.close()
    
    def compare_attraction_types(self, df):
        """Compare attraction types across countries."""
        if df is None or df.empty:
            return
        
        # Extract attraction types for each country
        if 'attraction_type' not in df.columns:
            print("Warning: attraction_type column not found, skipping comparison")
            return
        
        plt.figure(figsize=(12, 8))
        count_df = df.groupby(['country', 'attraction_type']).size().reset_index(name='count')
        
        # Calculate percentage within each country
        total_by_country = count_df.groupby('country')['count'].sum().reset_index()
        count_df = count_df.merge(total_by_country, on='country', suffixes=('', '_total'))
        count_df['percentage'] = count_df['count'] / count_df['count_total'] * 100
        
        # Plot
        sns.barplot(x='attraction_type', y='percentage', hue='country', data=count_df)
        plt.title('Attraction Types by Country (% of Total)')
        plt.xlabel('Attraction Type')
        plt.ylabel('Percentage')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Country')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/attractions/visualizations/attraction_types_comparison.png")
        plt.close()
    
    def compare_attraction_experiences(self, df):
        """Compare attraction experiences across countries with sentiment analysis."""
        if df is None or df.empty:
            return
            
        # Use 'text' or 'review_text' column depending on what's available
        text_column = 'text' if 'text' in df.columns else 'review_text'
        if text_column not in df.columns:
            print(f"Warning: Neither 'text' nor 'review_text' column found in attractions data")
            return
        
        # Define activity types and their related terms
        activity_types = {
            'water_activities': ['swim', 'snorkel', 'dive', 'kayak', 'paddle', 'boat', 'sail', 'surf', 'fishing', 'beach', 'ocean', 'sea', 'lagoon'],
            'cultural_experiences': ['culture', 'history', 'traditional', 'authentic', 'museum', 'heritage', 'local', 'ceremony', 'dance', 'music', 'festival'],
            'nature_activities': ['hike', 'walk', 'trek', 'nature', 'wildlife', 'bird', 'forest', 'mountain', 'volcano', 'cave', 'waterfall', 'landscape'],
            'tours_excursions': ['tour', 'guide', 'excursion', 'trip', 'expedition', 'safari', 'sightseeing', 'day trip', 'cruise', 'boat tour']
        }
        
        # Define experience aspects
        experience_aspects = {
            'guide_quality': ['guide', 'knowledge', 'informative', 'explain', 'friendly', 'professional'],
            'safety_comfort': ['safety', 'safe', 'comfortable', 'equipment', 'secure', 'clean', 'well-maintained'],
            'organization': ['organized', 'punctual', 'timely', 'schedule', 'efficient', 'preparation'],
            'education_info': ['learn', 'educational', 'information', 'history', 'facts', 'interesting'],
            'value': ['value', 'price', 'worth', 'money', 'expensive', 'cheap', 'cost', 'fee']
        }
        
        # Initialize sentiment analyzer if needed
        from textblob import TextBlob
        
        # Extract sentences containing activity terms
        def get_relevant_sentences(text, terms):
            if not isinstance(text, str):
                return []
                
            text = text.lower()
            sentences = re.split(r'[.!?]+', text)
            relevant = []
            
            for sentence in sentences:
                if any(term in sentence for term in terms):
                    relevant.append(sentence)
            
            return relevant
        
        # Calculate sentiment for a list of sentences
        def calculate_sentiment(sentences):
            if not sentences:
                return None
                
            combined_text = ". ".join(sentences)
            return TextBlob(combined_text).sentiment.polarity
        
        # Create storage for activity sentiment analysis
        activity_sentiment = {country: {} for country in df['country'].unique()}
        aspect_sentiment = {country: {} for country in df['country'].unique()}
        
        # Analyze each review by country
        for country in df['country'].unique():
            country_data = df[df['country'] == country]
            
            # Analyze activities
            for activity_type, terms in activity_types.items():
                relevant_reviews = []
                sentiments = []
                
                for _, row in country_data.iterrows():
                    sentences = get_relevant_sentences(row.get(text_column, ''), terms)
                    if sentences:
                        sentiment = calculate_sentiment(sentences)
                        if sentiment is not None:
                            sentiments.append(sentiment)
                            relevant_reviews.append(row.name)
                
                if sentiments:
                    activity_sentiment[country][activity_type] = {
                        'avg_sentiment': sum(sentiments) / len(sentiments),
                        'review_count': len(relevant_reviews),
                        'percentage': (len(relevant_reviews) / len(country_data)) * 100
                    }
                else:
                    activity_sentiment[country][activity_type] = {
                        'avg_sentiment': 0,
                        'review_count': 0,
                        'percentage': 0
                    }
            
            # Analyze aspects
            for aspect, terms in experience_aspects.items():
                relevant_reviews = []
                sentiments = []
                
                for _, row in country_data.iterrows():
                    sentences = get_relevant_sentences(row.get(text_column, ''), terms)
                    if sentences:
                        sentiment = calculate_sentiment(sentences)
                        if sentiment is not None:
                            sentiments.append(sentiment)
                            relevant_reviews.append(row.name)
                
                if sentiments:
                    aspect_sentiment[country][aspect] = {
                        'avg_sentiment': sum(sentiments) / len(sentiments),
                        'review_count': len(relevant_reviews),
                        'percentage': (len(relevant_reviews) / len(country_data)) * 100
                    }
                else:
                    aspect_sentiment[country][aspect] = {
                        'avg_sentiment': 0,
                        'review_count': 0,
                        'percentage': 0
                    }
        
        # Add results to the analysis output
        if 'attractions' in self.results and 'country_results' in self.results['attractions']:
            print(f"Found attraction results, countries: {list(self.results['attractions']['country_results'].keys())}")
            for country in activity_sentiment:
                if country in self.results['attractions']['country_results']:
                    print(f"Adding activity_sentiment for {country}")
                    self.results['attractions']['country_results'][country]['activity_sentiment'] = activity_sentiment[country]
                    self.results['attractions']['country_results'][country]['aspect_sentiment'] = aspect_sentiment[country]
        
        # Create visualizations for activity sentiment comparison
        # 1. Activity type sentiment by country
        activity_data = []
        for country, activities in activity_sentiment.items():
            for activity_type, data in activities.items():
                if data['review_count'] > 0:  # Only include activities with data
                    activity_data.append({
                        'country': country,
                        'activity_type': activity_type,
                        'avg_sentiment': data['avg_sentiment'],
                        'review_count': data['review_count'],
                        'percentage': data['percentage']
                    })
        
        if activity_data:
            activity_df = pd.DataFrame(activity_data)
            
            # Plot sentiment comparison
            plt.figure(figsize=(12, 8))
            # Use regional colors from visualization_styles
            palette = {country: REGION_COLORS.get(country.lower(), '#AAAAAA') 
                      for country in activity_df['country'].unique()}
            
            ax = sns.barplot(x='activity_type', y='avg_sentiment', hue='country', data=activity_df, palette=palette)
            apply_regional_style(ax, 
                               title='Activity Type Sentiment by Country',
                               x_label='Activity Type', 
                               y_label='Average Sentiment Score')
            
            # Add data labels
            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f', padding=3)
            
            # Manually remap the x-axis tick labels
            labels = ['Water Activities', 'Cultural Experiences', 'Nature Activities', 'Tours & Excursions']
            plt.xticks(range(len(labels)), labels, rotation=0)
            
            plt.legend(title='Country')
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/attractions/visualizations/activity_sentiment_comparison.png", dpi=300)
            plt.close()
        
        # 2. Experience aspects sentiment by country
        aspect_data = []
        for country, aspects in aspect_sentiment.items():
            for aspect_type, data in aspects.items():
                if data['review_count'] > 0:  # Only include aspects with data
                    aspect_data.append({
                        'country': country,
                        'aspect_type': aspect_type,
                        'avg_sentiment': data['avg_sentiment'],
                        'review_count': data['review_count'],
                        'percentage': data['percentage']
                    })
        
        if aspect_data:
            aspect_df = pd.DataFrame(aspect_data)
            
            # Plot sentiment comparison
            plt.figure(figsize=(12, 8))
            palette = {country: REGION_COLORS.get(country.lower(), '#AAAAAA') 
                      for country in aspect_df['country'].unique()}
            
            ax = sns.barplot(x='aspect_type', y='avg_sentiment', hue='country', data=aspect_df, palette=palette)
            apply_regional_style(ax, 
                               title='Experience Aspect Sentiment by Country',
                               x_label='Experience Aspect', 
                               y_label='Average Sentiment Score')
            
            # Add data labels
            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f', padding=3)
            
            # Manually remap the x-axis tick labels with proper capitalization and spacing
            aspect_labels = {
                'guide_quality': 'Guide Quality',
                'safety_comfort': 'Safety & Comfort',
                'organization': 'Organization',
                'education_info': 'Educational Info',
                'value': 'Value'
            }
            
            # Get unique aspect types in the data
            aspect_types = aspect_df['aspect_type'].unique()
            
            # Create labels in the same order as they appear in the plot
            labels = [aspect_labels.get(aspect, aspect.replace('_', ' ').title()) for aspect in aspect_types]
            
            # Apply the new labels
            plt.xticks(range(len(labels)), labels, rotation=0)
            
            plt.legend(title='Country')
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/attractions/visualizations/aspect_sentiment_comparison.png", dpi=300)
            plt.close()
            
        # 3. Create heatmap for sentiment across activities and countries
        if activity_data:
            plt.figure(figsize=(12, 8))
            pivot_df = activity_df.pivot_table(index='country', columns='activity_type', values='avg_sentiment')
            
            # Create heatmap
            ax = sns.heatmap(pivot_df, annot=True, cmap='RdYlGn', center=0, fmt='.2f')
            plt.title('Sentiment Heatmap by Activity Type and Country', fontsize=14, pad=20)
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/attractions/visualizations/activity_sentiment_heatmap.png", dpi=300)
            plt.close()
        
        # 4. Create heatmap for sentiment across aspects and countries
        if aspect_data:
            plt.figure(figsize=(12, 8))
            pivot_df = aspect_df.pivot_table(index='country', columns='aspect_type', values='avg_sentiment')
            
            # Create heatmap
            ax = sns.heatmap(pivot_df, annot=True, cmap='RdYlGn', center=0, fmt='.2f', annot_kws={"size": 14})
            
            # Remap the column labels for display
            aspect_labels = {
                'guide_quality': 'Guide Quality',
                'safety_comfort': 'Safety & Comfort',
                'organization': 'Organization',
                'education_info': 'Educational Info',
                'value': 'Value'
            }
            
            # Apply new column labels
            new_columns = [aspect_labels.get(col, col.replace('_', ' ').title()) for col in pivot_df.columns]
            ax.set_xticklabels(new_columns, fontsize=18, rotation=0)
            
            # Format y-tick labels (countries)
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=18)
            
            plt.title('Sentiment Heatmap by Experience Aspect and Country', fontsize=22, pad=20, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/attractions/visualizations/aspect_sentiment_heatmap.png", dpi=300)
            plt.close()
            
        # Original experience comparison visualization (kept for backward compatibility)
        experience_terms = {
            'scenic': ['scenic', 'view', 'landscape', 'beautiful', 'stunning'],
            'adventure': ['adventure', 'exciting', 'thrill', 'adrenaline'],
            'cultural': ['culture', 'history', 'traditional', 'authentic'],
            'relaxing': ['relax', 'peaceful', 'calm', 'quiet'],
            'family-friendly': ['family', 'kids', 'children', 'child-friendly'],
            'guided': ['guide', 'tour', 'guided', 'explained']
        }
        
        # Score each review for experience terms
        for experience, terms in experience_terms.items():
            df[experience] = df[text_column].str.lower().apply(
                lambda x: sum(1 for term in terms if term in str(x).lower())
            )
        
        # Aggregate by country
        experience_data = []
        for country in df['country'].unique():
            country_data = df[df['country'] == country]
            for experience in experience_terms.keys():
                # Calculate percentage of reviews mentioning this experience
                count = (country_data[experience] > 0).sum()
                percentage = count / len(country_data) * 100
                experience_data.append({
                    'country': country,
                    'experience': experience,
                    'percentage': percentage
                })
        
        experience_df = pd.DataFrame(experience_data)
        
        # Plot
        plt.figure(figsize=(12, 8))
        sns.barplot(x='experience', y='percentage', hue='country', data=experience_df)
        plt.title('Attraction Experience Types by Country')
        plt.xlabel('Experience Type')
        plt.ylabel('Percentage of Reviews Mentioning')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Country')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/attractions/visualizations/experience_comparison.png")
        plt.close()
    
    def compare_attraction_ratings(self, df):
        """Compare attraction ratings across countries."""
        if df is None or df.empty:
            return
        
        # Compare rating distributions
        if 'rating' not in df.columns:
            print("Warning: rating column not found, skipping comparison")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Box plot
        sns.boxplot(x='country', y='rating', data=df)
        plt.title('Attraction Ratings by Country')
        plt.xlabel('Country')
        plt.ylabel('Rating')
        plt.savefig(f"{self.output_dir}/attractions/visualizations/rating_distribution_comparison.png")
        plt.close()
        
        # Bar chart with average ratings
        plt.figure(figsize=(10, 6))
        avg_ratings = df.groupby('country')['rating'].mean().reset_index()
        sns.barplot(x='country', y='rating', data=avg_ratings)
        plt.title('Average Attraction Ratings by Country')
        plt.xlabel('Country')
        plt.ylabel('Average Rating')
        plt.ylim(0, 5)  # Assuming 5-star rating scale
        for i, row in enumerate(avg_ratings.itertuples()):
            plt.text(i, row.rating + 0.1, f'{row.rating:.2f}', ha='center')
        plt.savefig(f"{self.output_dir}/attractions/visualizations/average_rating_comparison.png")
        plt.close()
        
        # Sentiment comparison if available
        if 'sentiment_score' in df.columns:
            # Filter out rows with null sentiment_score
            sentiment_df = df.dropna(subset=['sentiment_score'])
            if not sentiment_df.empty:  # Only create visualization if we have data
                plt.figure(figsize=(10, 6))
                avg_sentiment = sentiment_df.groupby('country')['sentiment_score'].mean().reset_index()
                sns.barplot(x='country', y='sentiment_score', data=avg_sentiment)
                plt.title('Average Sentiment Score for Attractions by Country')
                plt.xlabel('Country')
                plt.ylabel('Average Sentiment Score')
                for i, row in enumerate(avg_sentiment.itertuples()):
                    plt.text(i, row.sentiment_score + 0.02, f'{row.sentiment_score:.2f}', ha='center')
                plt.savefig(f"{self.output_dir}/attractions/visualizations/sentiment_comparison.png")
                plt.close()
    
    def compare_cuisine_types(self, df):
        """Compare cuisine types across countries."""
        if df is None or df.empty or 'review_text' not in df.columns:
            return
        
        # Define cuisine types to look for
        cuisine_types = {
            'seafood': ['seafood', 'fish', 'shrimp', 'lobster', 'crab', 'oyster'],
            'local': ['local', 'traditional', 'native', 'authentic'],
            'international': ['international', 'continental', 'fusion', 'modern'],
            'vegetarian': ['vegetarian', 'vegan', 'plant-based'],
            'asian': ['asian', 'chinese', 'japanese', 'thai', 'vietnamese'],
            'european': ['european', 'italian', 'french', 'greek'],
            'american': ['american', 'burger', 'steak', 'bbq']
        }
        
        # Score each review for cuisine types
        for cuisine, terms in cuisine_types.items():
            df[cuisine] = df['review_text'].str.lower().apply(
                lambda x: sum(1 for term in terms if term in str(x).lower())
            )
        
        # Aggregate by country
        cuisine_data = []
        for country in df['country'].unique():
            country_data = df[df['country'] == country]
            for cuisine in cuisine_types.keys():
                # Calculate percentage of reviews mentioning this cuisine
                count = (country_data[cuisine] > 0).sum()
                percentage = count / len(country_data) * 100
                cuisine_data.append({
                    'country': country,
                    'cuisine': cuisine,
                    'percentage': percentage
                })
        
        cuisine_df = pd.DataFrame(cuisine_data)
        
        # Plot
        plt.figure(figsize=(12, 8))
        sns.barplot(x='cuisine', y='percentage', hue='country', data=cuisine_df)
        plt.title('Cuisine Types by Country')
        plt.xlabel('Cuisine Type')
        plt.ylabel('Percentage of Reviews Mentioning')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Country')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/restaurants/visualizations/cuisine_comparison.png")
        plt.close()
    
    def compare_dining_experience(self, df):
        """Compare dining experience factors across countries."""
        if df is None or df.empty or 'review_text' not in df.columns:
            return
        
        # Define dining experience factors
        experience_factors = {
            'service': ['service', 'staff', 'waiter', 'waitress', 'attentive'],
            'ambiance': ['ambiance', 'atmosphere', 'decor', 'setting', 'view'],
            'value': ['value', 'price', 'worth', 'expensive', 'cheap', 'affordable'],
            'quality': ['quality', 'fresh', 'delicious', 'tasty', 'flavor'],
            'portion': ['portion', 'serving', 'size', 'generous', 'large', 'small'],
            'waiting': ['wait', 'time', 'quick', 'slow', 'fast']
        }
        
        # Score each review for experience factors
        for factor, terms in experience_factors.items():
            df[factor] = df['review_text'].str.lower().apply(
                lambda x: sum(1 for term in terms if term in str(x).lower())
            )
        
        # Aggregate by country and factor
        experience_data = []
        for country in df['country'].unique():
            country_data = df[df['country'] == country]
            for factor in experience_factors.keys():
                # Calculate sentiment for reviews mentioning this factor
                factor_reviews = country_data[country_data[factor] > 0]
                if len(factor_reviews) > 0 and 'sentiment_score' in factor_reviews.columns:
                    avg_sentiment = factor_reviews['sentiment_score'].mean()
                    experience_data.append({
                        'country': country,
                        'factor': factor,
                        'avg_sentiment': avg_sentiment,
                        'review_count': len(factor_reviews)
                    })
        
        experience_df = pd.DataFrame(experience_data)
        
        if not experience_df.empty:
            # Plot sentiment for dining factors
            plt.figure(figsize=(12, 8))
            sns.barplot(x='factor', y='avg_sentiment', hue='country', data=experience_df)
            plt.title('Sentiment for Dining Experience Factors by Country')
            plt.xlabel('Dining Experience Factor')
            plt.ylabel('Average Sentiment Score')
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Country')
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/restaurants/visualizations/dining_experience_sentiment.png")
            plt.close()
            
            # Plot review count (popularity of factors)
            plt.figure(figsize=(12, 8))
            
            # Convert to percentage of country's total
            total_reviews = experience_df.groupby('country')['review_count'].sum().reset_index()
            experience_df = experience_df.merge(total_reviews, on='country', suffixes=('', '_total'))
            experience_df['percentage'] = (experience_df['review_count'] / experience_df['review_count_total']) * 100
            
            sns.barplot(x='factor', y='percentage', hue='country', data=experience_df)
            plt.title('Dining Experience Factors Mentioned by Country')
            plt.xlabel('Dining Experience Factor')
            plt.ylabel('Percentage of Reviews Mentioning')
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Country')
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/restaurants/visualizations/dining_experience_mentions.png")
            plt.close()
    
    def compare_restaurant_ratings(self, df):
        """Compare restaurant ratings across countries."""
        if df is None or df.empty:
            return
        
        # Compare rating distributions
        if 'rating' not in df.columns:
            print("Warning: rating column not found, skipping comparison")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Box plot
        sns.boxplot(x='country', y='rating', data=df)
        plt.title('Restaurant Ratings by Country')
        plt.xlabel('Country')
        plt.ylabel('Rating')
        plt.savefig(f"{self.output_dir}/restaurants/visualizations/rating_distribution_comparison.png")
        plt.close()
        
        # Bar chart with average ratings
        plt.figure(figsize=(10, 6))
        avg_ratings = df.groupby('country')['rating'].mean().reset_index()
        sns.barplot(x='country', y='rating', data=avg_ratings)
        plt.title('Average Restaurant Ratings by Country')
        plt.xlabel('Country')
        plt.ylabel('Average Rating')
        plt.ylim(0, 5)  # Assuming 5-star rating scale
        for i, row in enumerate(avg_ratings.itertuples()):
            plt.text(i, row.rating + 0.1, f'{row.rating:.2f}', ha='center')
        plt.savefig(f"{self.output_dir}/restaurants/visualizations/average_rating_comparison.png")
        plt.close()
        
        # Sentiment comparison if available
        if 'sentiment_score' in df.columns:
            # Filter out rows with null sentiment_score
            sentiment_df = df.dropna(subset=['sentiment_score'])
            if not sentiment_df.empty:  # Only create visualization if we have data
                plt.figure(figsize=(10, 6))
                avg_sentiment = sentiment_df.groupby('country')['sentiment_score'].mean().reset_index()
                sns.barplot(x='country', y='sentiment_score', data=avg_sentiment)
                plt.title('Average Sentiment Score for Restaurants by Country')
                plt.xlabel('Country')
                plt.ylabel('Average Sentiment Score')
                for i, row in enumerate(avg_sentiment.itertuples()):
                    plt.text(i, row.sentiment_score + 0.02, f'{row.sentiment_score:.2f}', ha='center')
                plt.savefig(f"{self.output_dir}/restaurants/visualizations/sentiment_comparison.png")
                plt.close()
    
    def create_cross_sector_comparisons(self):
        """Create visualizations comparing across sectors (accommodations, attractions, restaurants)."""
        # Combine data from all sectors and countries
        combined_data = self.combine_country_data()
        all_sectors = []
        
        for sector, df in combined_data.items():
            if df is not None and not df.empty:
                df['sector'] = sector
                all_sectors.append(df)
        
        if not all_sectors:
            return
        
        all_data = pd.concat(all_sectors, ignore_index=True)
        
        # Overall rating comparison by country and sector
        if 'rating' in all_data.columns:
            plt.figure(figsize=(14, 8))
            avg_ratings = all_data.groupby(['country', 'sector'])['rating'].mean().reset_index()
            pivot_data = avg_ratings.pivot(index='country', columns='sector', values='rating')
            
            # Plot
            ax = pivot_data.plot(kind='bar', figsize=(14, 8))
            plt.title('Average Ratings by Country and Sector')
            plt.xlabel('Country')
            plt.ylabel('Average Rating')
            plt.ylim(0, 5)  # Assuming 5-star rating scale
            plt.legend(title='Sector')
            
            # Add data labels
            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f')
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/overall/visualizations/ratings_by_country_sector.png")
            plt.close()
            
            # Save the data to CSV
            pivot_data.to_csv(f"{self.output_dir}/overall/ratings_by_country_sector.csv")
        
        # Sentiment comparison if available
        if 'sentiment_score' in all_data.columns:
            # Filter out rows with null sentiment_score
            sentiment_df = all_data.dropna(subset=['sentiment_score'])
            if not sentiment_df.empty:  # Only create visualization if we have data
                plt.figure(figsize=(14, 8))
                avg_sentiment = sentiment_df.groupby(['country', 'sector'])['sentiment_score'].mean().reset_index()
                
                # Only create pivot if we have data for at least one country and sector
                if not avg_sentiment.empty:
                    pivot_sentiment = avg_sentiment.pivot(index='country', columns='sector', values='sentiment_score')
                    
                    # Plot
                    ax = pivot_sentiment.plot(kind='bar', figsize=(14, 8))
                    plt.title('Average Sentiment by Country and Sector')
                    plt.xlabel('Country')
                    plt.ylabel('Average Sentiment Score')
                    plt.legend(title='Sector')
                    
                    # Add data labels
                    for container in ax.containers:
                        ax.bar_label(container, fmt='%.2f')
                    
                    plt.tight_layout()
                    plt.savefig(f"{self.output_dir}/overall/visualizations/sentiment_by_country_sector.png")
                    plt.close()
                    
                    # Save the data to CSV
                    pivot_sentiment.to_csv(f"{self.output_dir}/overall/sentiment_by_country_sector.csv")
        
        # Create a heatmap of all metrics
        self.create_competitive_position_heatmap(all_data)
    
    def create_competitive_position_heatmap(self, df):
        """Create a heatmap showing Tonga's position compared to other countries."""
        if df is None or df.empty:
            return
        
        # Only proceed if Tonga is in the data
        if 'Tonga' not in df['country'].unique():
            print("Warning: Tonga data not found, skipping competitive position heatmap")
            return
        
        # Create metrics for heatmap
        metrics = {}
        
        # Rating difference
        if 'rating' in df.columns:
            avg_ratings = df.groupby(['country', 'sector'])['rating'].mean().reset_index()
            tonga_ratings = avg_ratings[avg_ratings['country'] == 'Tonga']
            
            for sector in tonga_ratings['sector'].unique():
                tonga_rating = tonga_ratings[tonga_ratings['sector'] == sector]['rating'].values[0]
                
                for country in df['country'].unique():
                    if country != 'Tonga':
                        country_rating = avg_ratings[
                            (avg_ratings['country'] == country) & 
                            (avg_ratings['sector'] == sector)
                        ]['rating'].values
                        
                        if len(country_rating) > 0:
                            diff = tonga_rating - country_rating[0]
                            metrics[(f"Rating: {sector}", country)] = diff
        
        # Sentiment difference
        if 'sentiment_score' in df.columns:
            # Filter out rows with null sentiment_score to avoid issues
            sentiment_df = df.dropna(subset=['sentiment_score'])
            # Only proceed if we have data after filtering
            if not sentiment_df.empty:
                avg_sentiment = sentiment_df.groupby(['country', 'sector'])['sentiment_score'].mean().reset_index()
                # Check if Tonga is still in the data after filtering
                if 'Tonga' in avg_sentiment['country'].values:
                    tonga_sentiment = avg_sentiment[avg_sentiment['country'] == 'Tonga']
                    
                    for sector in tonga_sentiment['sector'].unique():
                        tonga_sent = tonga_sentiment[tonga_sentiment['sector'] == sector]['sentiment_score'].values[0]
                        
                        for country in sentiment_df['country'].unique():
                            if country != 'Tonga':
                                country_sent = avg_sentiment[
                                    (avg_sentiment['country'] == country) & 
                                    (avg_sentiment['sector'] == sector)
                                ]['sentiment_score'].values
                                
                                if len(country_sent) > 0:
                                    diff = tonga_sent - country_sent[0]
                                    metrics[(f"Sentiment: {sector}", country)] = diff
        
        if not metrics:
            return
        
        # Convert to DataFrame for heatmap
        heatmap_data = []
        for (metric, country), value in metrics.items():
            heatmap_data.append({
                'Metric': metric,
                'Country': country,
                'Difference': value
            })
        
        heatmap_df = pd.DataFrame(heatmap_data)
        pivot_data = heatmap_df.pivot(index='Metric', columns='Country', values='Difference')
        
        # Plot heatmap with improved styling 
        plt.figure(figsize=(12, 10))
        
        # Apply consistent visualization style
        set_visualization_style()
        
        # Create a custom colormap that uses our brand colors for diverging data
        # Use red from Tonga for positive values (where Tonga performs better)
        # and blue/teal from other countries for negative values
        custom_cmap = sns.diverging_palette(
            h_neg=200, h_pos=10,  # Blue to Red
            s=80, l=55,
            as_cmap=True
        )
        
        ax = sns.heatmap(pivot_data, annot=True, cmap=custom_cmap, 
                     center=0, fmt='.2f', linewidths=0.5)
        
        # Style the heatmap
        ax.set_title("Tonga's Competitive Position\n(Positive values = Tonga performs better)", 
                  fontsize=16, fontweight='bold', pad=20)
        
        # Adjust font sizes for better readability
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        
        # Add a note about the color interpretation
        ax.annotate(
            "Red = Tonga outperforms\nBlue = Competitor outperforms", 
            xy=(0.02, 0.02), 
            xycoords='figure fraction', 
            fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8)
        )
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/overall/visualizations/competitive_position_heatmap.png", dpi=300)
        plt.close()
        
        # Save to CSV
        pivot_data.to_csv(f"{self.output_dir}/overall/competitive_position_metrics.csv")
    
    def analyze_sentiment_aspects(self, df, aspects):
        """
        Analyze sentiment around specific aspects in reviews.
        
        Args:
            df: DataFrame with review text and sentiment data
            aspects: Dictionary mapping aspect names to related keywords
            
        Returns:
            DataFrame with aspect sentiment analysis
        """
        aspect_results = []
        
        # Ensure we have the necessary columns
        if 'review_text' not in df.columns or 'sentiment_score' not in df.columns:
            return pd.DataFrame()
        
        for country in df['country'].unique():
            country_df = df[df['country'] == country]
            
            for aspect_name, keywords in aspects.items():
                # Find reviews mentioning this aspect
                aspect_mentioned = country_df['review_text'].str.lower().apply(
                    lambda x: any(keyword in str(x).lower() for keyword in keywords)
                )
                aspect_reviews = country_df[aspect_mentioned]
                
                if len(aspect_reviews) > 0:
                    avg_sentiment = aspect_reviews['sentiment_score'].mean()
                    count = len(aspect_reviews)
                    positive = len(aspect_reviews[aspect_reviews['sentiment_score'] > 0.1])
                    negative = len(aspect_reviews[aspect_reviews['sentiment_score'] < -0.1])
                    
                    aspect_results.append({
                        'country': country,
                        'aspect': aspect_name,
                        'avg_sentiment': avg_sentiment,
                        'mention_count': count,
                        'positive_mentions': positive,
                        'negative_mentions': negative,
                        'percentage_positive': (positive / count) * 100 if count > 0 else 0
                    })
        
        return pd.DataFrame(aspect_results)
    
    def generate_cross_regional_sentiment_analysis(self):
        """Generate detailed cross-regional sentiment comparisons and visualizations."""
        print("Generating cross-regional sentiment analysis...")
        combined_data = self.combine_country_data()
        
        # Create output directory
        sentiment_dir = f"{self.output_dir}/overall/sentiment_analysis"
        os.makedirs(sentiment_dir, exist_ok=True)
        os.makedirs(f"{sentiment_dir}/visualizations", exist_ok=True)
        
        # Common aspects to analyze across sectors
        common_aspects = {
            "cleanliness": ["clean", "dirt", "hygiene", "spotless", "tidy", "messy"],
            "staff": ["staff", "service", "employee", "attendant", "worker", "host", "hostess"],
            "value": ["value", "price", "cost", "cheap", "expensive", "worth", "overpriced"],
            "location": ["location", "situated", "area", "central", "convenient", "accessible"],
            "facilities": ["facility", "amenity", "feature", "equipment", "infrastructure"],
            "local_experience": ["local", "authentic", "traditional", "cultural", "unique"],
            "family_friendly": ["family", "kid", "child", "children", "family-friendly"],
        }
        
        # Sector-specific aspects
        sector_aspects = {
            "accommodations": {
                "room_quality": ["room", "bed", "bathroom", "shower", "spacious", "comfortable"],
                "sleep_quality": ["sleep", "quiet", "noise", "peaceful", "rest", "mattress"],
                "wifi": ["wifi", "internet", "connection", "online", "connectivity"]
            },
            "attractions": {
                "scenery": ["view", "scenery", "landscape", "beautiful", "picturesque", "scenic"],
                "activities": ["activity", "tour", "excursion", "adventure", "experience"],
                "educational": ["learn", "educational", "informative", "history", "knowledge"]
            },
            "restaurants": {
                "food_quality": ["food", "dish", "meal", "delicious", "taste", "flavor"],
                "menu_variety": ["menu", "variety", "selection", "choice", "options"],
                "ambiance": ["ambiance", "atmosphere", "decor", "cozy", "romantic", "setting"]
            }
        }
        
        # Process each sector
        for sector, df in combined_data.items():
            if df is None or df.empty or 'review_text' not in df.columns:
                continue
                
            print(f"Analyzing {sector} sentiment by aspect...")
            
            # Combine common and sector-specific aspects
            all_aspects = {**common_aspects, **sector_aspects.get(sector, {})}
            
            # Analyze aspects
            aspect_results = self.analyze_sentiment_aspects(df, all_aspects)
            
            if not aspect_results.empty:
                # Save results
                aspect_results.to_csv(f"{sentiment_dir}/{sector}_aspect_sentiment.csv", index=False)
                
                # Create visualizations
                plt.figure(figsize=(14, 10))
                
                # Pivot data for heatmap
                pivot_data = aspect_results.pivot_table(
                    values='avg_sentiment', 
                    index='aspect',
                    columns='country'
                )
                
                # Plot heatmap
                sns.heatmap(pivot_data, annot=True, cmap='RdYlGn', center=0, fmt='.2f')
                plt.title(f"Sentiment by Aspect for {sector.capitalize()}")
                plt.tight_layout()
                plt.savefig(f"{sentiment_dir}/visualizations/{sector}_aspect_sentiment_heatmap.png")
                plt.close()
                
                # Plot bar chart of aspect mentions
                plt.figure(figsize=(14, 10))
                aspect_counts = aspect_results.pivot_table(
                    values='mention_count',
                    index='aspect',
                    columns='country',
                    aggfunc='sum'
                )
                
                aspect_counts.plot(kind='barh', figsize=(14, 10))
                plt.title(f"Aspect Mention Counts for {sector.capitalize()}")
                plt.xlabel('Number of Reviews Mentioning Aspect')
                plt.tight_layout()
                plt.savefig(f"{sentiment_dir}/visualizations/{sector}_aspect_mentions_barplot.png")
                plt.close()
                
                # Generate positive vs negative comparison
                plt.figure(figsize=(14, 10))
                
                # For each aspect, show percentage positive by country
                pivot_positive = aspect_results.pivot_table(
                    values='percentage_positive',
                    index='aspect',
                    columns='country'
                )
                
                sns.heatmap(pivot_positive, annot=True, cmap='RdYlGn', vmin=0, vmax=100, fmt='.1f')
                plt.title(f"Percentage of Positive Mentions by Aspect for {sector.capitalize()}")
                plt.tight_layout()
                plt.savefig(f"{sentiment_dir}/visualizations/{sector}_positive_sentiment_percentage.png")
                plt.close()
                
        # Extract qualitative themes from reviews
        self.extract_qualitative_themes(combined_data, sentiment_dir)
                
        print("Cross-regional sentiment analysis complete.")
    
    def extract_qualitative_themes(self, combined_data, output_dir):
        """Extract qualitative themes and phrases from reviews for comparison."""
        print("Extracting qualitative themes from reviews...")
        
        # Themes to look for in positive and negative reviews
        themes = {
            "service": ["service", "staff", "friendly", "helpful", "attentive", "professional"],
            "cleanliness": ["clean", "cleanliness", "tidy", "spotless", "dirty", "hygiene"],
            "food": ["food", "delicious", "tasty", "meal", "dish", "flavor", "dining"],
            "accommodation": ["room", "bed", "comfortable", "spacious", "quiet", "noisy"],
            "value": ["value", "price", "expensive", "affordable", "worth", "cost"],
            "location": ["location", "central", "convenient", "walking", "distance", "accessible"],
            "facilities": ["facilities", "amenities", "pool", "wifi", "breakfast", "beach"],
            "experience": ["experience", "enjoy", "memorable", "disappointing", "recommend", "return"],
            "culture": ["culture", "traditional", "local", "authentic", "unique", "heritage"]
        }
        
        results = {}
        
        for sector, df in combined_data.items():
            if df is None or df.empty or 'review_text' not in df.columns:
                continue
                
            sector_results = {}
            
            # For each theme, extract top phrases from positive and negative reviews
            for theme, keywords in themes.items():
                theme_results = {}
                
                # For each country
                for country in df['country'].unique():
                    country_df = df[df['country'] == country]
                    
                    # Filter for reviews mentioning this theme
                    theme_pattern = '|'.join(keywords)
                    theme_reviews = country_df[country_df['review_text'].str.lower().str.contains(theme_pattern, na=False)]
                    
                    if len(theme_reviews) > 0:
                        # Split by sentiment
                        positive_reviews = theme_reviews[theme_reviews['sentiment_score'] > 0.1]
                        negative_reviews = theme_reviews[theme_reviews['sentiment_score'] < -0.1]
                        
                        # Extract phrases (simplified approach)
                        positive_text = ' '.join(positive_reviews['review_text'].fillna('').str.lower())
                        negative_text = ' '.join(negative_reviews['review_text'].fillna('').str.lower())
                        
                        # Find phrase contexts around keywords
                        positive_phrases = []
                        negative_phrases = []
                        
                        for keyword in keywords:
                            # Get phrases for positive reviews
                            matches = re.findall(r'([^.!?]*\b' + keyword + r'\b[^.!?]*[.!?])', positive_text)
                            positive_phrases.extend(matches[:5])  # Limit to 5 phrases per keyword
                            
                            # Get phrases for negative reviews
                            matches = re.findall(r'([^.!?]*\b' + keyword + r'\b[^.!?]*[.!?])', negative_text)
                            negative_phrases.extend(matches[:5])  # Limit to 5 phrases per keyword
                        
                        theme_results[country] = {
                            "positive_count": len(positive_reviews),
                            "negative_count": len(negative_reviews),
                            "positive_phrases": positive_phrases[:10],  # Top 10 phrases
                            "negative_phrases": negative_phrases[:10]   # Top 10 phrases
                        }
                
                if theme_results:
                    sector_results[theme] = theme_results
            
            if sector_results:
                results[sector] = sector_results
        
        # Save results to JSON
        with open(f"{output_dir}/qualitative_themes.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
        
        print("Qualitative themes extracted and saved.")
    
    def generate_competitive_insights(self):
        """Generate insights about Tonga's competitive position."""
        # This would typically involve more sophisticated analysis
        # and possibly NLP or rules-based insights generation
        
        # For now, we'll do a simple calculation based on available metrics
        combined_data = self.combine_country_data()
        
        insights = {
            "accommodations": {},
            "attractions": {},
            "restaurants": {},
            "overall": {}
        }
        
        # Generate insights for each sector
        for sector, df in combined_data.items():
            if df is None or df.empty or 'Tonga' not in df['country'].unique():
                continue
                
            # Rating comparison
            if 'rating' in df.columns:
                avg_ratings = df.groupby('country')['rating'].mean()
                tonga_rating = avg_ratings.get('Tonga')
                if tonga_rating is not None:
                    # Find countries with higher ratings
                    better_countries = avg_ratings[avg_ratings > tonga_rating].index.tolist()
                    worse_countries = avg_ratings[avg_ratings < tonga_rating].index.tolist()
                    
                    insights[sector]["rating_comparison"] = {
                        "tonga_avg_rating": tonga_rating,
                        "better_than_tonga": better_countries,
                        "worse_than_tonga": worse_countries,
                        "tonga_position": len(worse_countries) + 1,
                        "total_countries": len(avg_ratings)
                    }
            
            # Sentiment comparison
            if 'sentiment_score' in df.columns:
                # Filter out rows with null sentiment_score
                sentiment_df = df.dropna(subset=['sentiment_score'])
                if not sentiment_df.empty and 'Tonga' in sentiment_df['country'].unique():
                    avg_sentiment = sentiment_df.groupby('country')['sentiment_score'].mean()
                    tonga_sentiment = avg_sentiment.get('Tonga')
                    if tonga_sentiment is not None:
                        # Find countries with higher sentiment
                        better_countries = avg_sentiment[avg_sentiment > tonga_sentiment].index.tolist()
                        worse_countries = avg_sentiment[avg_sentiment < tonga_sentiment].index.tolist()
                        
                        insights[sector]["sentiment_comparison"] = {
                            "tonga_avg_sentiment": tonga_sentiment,
                            "better_than_tonga": better_countries,
                            "worse_than_tonga": worse_countries,
                            "tonga_position": len(worse_countries) + 1,
                            "total_countries": len(avg_sentiment)
                        }
                        
            # Add aspect-based comparisons if available
            aspect_file = f"{self.output_dir}/overall/sentiment_analysis/{sector}_aspect_sentiment.csv"
            if os.path.exists(aspect_file):
                aspect_df = pd.read_csv(aspect_file)
                
                # Extract Tonga's strengths and weaknesses
                if 'Tonga' in aspect_df['country'].unique():
                    tonga_aspects = aspect_df[aspect_df['country'] == 'Tonga']
                    
                    # Top and bottom aspects by sentiment
                    if not tonga_aspects.empty:
                        aspects_by_sentiment = tonga_aspects.sort_values('avg_sentiment', ascending=False)
                        top_aspects = aspects_by_sentiment.head(3)['aspect'].tolist()
                        bottom_aspects = aspects_by_sentiment.tail(3)['aspect'].tolist()
                        
                        insights[sector]["aspect_strengths_weaknesses"] = {
                            "top_aspects": top_aspects,
                            "bottom_aspects": bottom_aspects
                        }
                        
                        # Aspects where Tonga outperforms other countries
                        other_countries = [c for c in aspect_df['country'].unique() if c != 'Tonga']
                        outperforming_aspects = []
                        underperforming_aspects = []
                        
                        for aspect in tonga_aspects['aspect'].unique():
                            tonga_sentiment = tonga_aspects[tonga_aspects['aspect'] == aspect]['avg_sentiment'].values[0]
                            
                            # Compare with other countries
                            for country in other_countries:
                                country_data = aspect_df[(aspect_df['country'] == country) & 
                                                       (aspect_df['aspect'] == aspect)]
                                
                                if not country_data.empty:
                                    country_sentiment = country_data['avg_sentiment'].values[0]
                                    
                                    # If Tonga outperforms significantly
                                    if tonga_sentiment > country_sentiment + 0.1:
                                        outperforming_aspects.append({
                                            "aspect": aspect,
                                            "compared_to": country,
                                            "tonga_sentiment": tonga_sentiment,
                                            "other_sentiment": country_sentiment,
                                            "difference": tonga_sentiment - country_sentiment
                                        })
                                    
                                    # If Tonga underperforms significantly
                                    if tonga_sentiment < country_sentiment - 0.1:
                                        underperforming_aspects.append({
                                            "aspect": aspect,
                                            "compared_to": country,
                                            "tonga_sentiment": tonga_sentiment,
                                            "other_sentiment": country_sentiment,
                                            "difference": tonga_sentiment - country_sentiment
                                        })
                        
                        # Sort by difference magnitude
                        outperforming_aspects.sort(key=lambda x: x['difference'], reverse=True)
                        underperforming_aspects.sort(key=lambda x: x['difference'])
                        
                        insights[sector]["comparative_aspects"] = {
                            "outperforming": outperforming_aspects[:5],  # Top 5
                            "underperforming": underperforming_aspects[:5]  # Bottom 5
                        }
        
        # Overall insights
        all_sectors = []
        for sector, df in combined_data.items():
            if df is not None and not df.empty:
                df['sector'] = sector
                all_sectors.append(df)
        
        if all_sectors:
            all_data = pd.concat(all_sectors, ignore_index=True)
            
            if 'rating' in all_data.columns and 'Tonga' in all_data['country'].unique():
                # Overall rating position
                avg_ratings = all_data.groupby('country')['rating'].mean()
                tonga_rating = avg_ratings.get('Tonga')
                if tonga_rating is not None:
                    better_countries = avg_ratings[avg_ratings > tonga_rating].index.tolist()
                    worse_countries = avg_ratings[avg_ratings < tonga_rating].index.tolist()
                    
                    insights["overall"]["rating_comparison"] = {
                        "tonga_avg_rating": tonga_rating,
                        "better_than_tonga": better_countries,
                        "worse_than_tonga": worse_countries,
                        "tonga_position": len(worse_countries) + 1,
                        "total_countries": len(avg_ratings)
                    }
            
            # Identify Tonga's strengths and weaknesses
            if 'sector' in all_data.columns and 'rating' in all_data.columns and 'Tonga' in all_data['country'].unique():
                tonga_data = all_data[all_data['country'] == 'Tonga']
                sector_ratings = tonga_data.groupby('sector')['rating'].mean()
                
                # Find Tonga's strongest and weakest sectors
                strongest_sector = sector_ratings.idxmax() if not sector_ratings.empty else None
                weakest_sector = sector_ratings.idxmin() if not sector_ratings.empty else None
                
                if strongest_sector and weakest_sector:
                    insights["overall"]["tonga_sector_comparison"] = {
                        "strongest_sector": strongest_sector,
                        "strongest_sector_rating": sector_ratings[strongest_sector],
                        "weakest_sector": weakest_sector,
                        "weakest_sector_rating": sector_ratings[weakest_sector]
                    }
        
        # Save insights to JSON
        with open(f"{self.output_dir}/overall/competitive_insights.json", 'w', encoding='utf-8') as f:
            json.dump(insights, f, indent=4)
        
        return insights
    
    def run_analysis(self):
        """Run the complete regional comparison analysis."""
        print("Starting regional comparison analysis...")
        
        # Step 1: Load data for all countries
        print("Loading data for all countries...")
        self.load_country_data()
        
        # Step 2: Apply sentiment analysis
        print("Applying sentiment analysis...")
        self.apply_sentiment_analysis()
        
        # Step 3: Generate detailed cross-regional sentiment analysis
        print("Generating detailed cross-regional sentiment analysis...")
        self.generate_cross_regional_sentiment_analysis()
        
        # Step 4: Run sector-specific analyses
        print("Analyzing accommodations...")
        self.analyze_accommodations()
        
        print("Analyzing attractions...")
        self.analyze_attractions()
        
        print("Analyzing restaurants...")
        self.analyze_restaurants()
        
        # Step 5: Create cross-sector comparisons
        print("Creating cross-sector comparisons...")
        self.create_cross_sector_comparisons()
        
        # Step 6: Generate competitive insights
        print("Generating competitive insights...")
        insights = self.generate_competitive_insights()
        
        print("Regional comparison analysis complete.")
        return self.results, insights