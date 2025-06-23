import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import json
import numpy as np
import os
import re
import seaborn as sns
from wordcloud import WordCloud
from tonga_analysis.visualization_styles import (
    REGION_COLORS, ISLAND_COLORS, set_visualization_style, 
    get_island_palette, apply_island_style
)

class AttractionAnalyzer:
    """
    Specialized analyzer for attraction and activity reviews in Tonga.
    Focuses on tours, activities, cultural experiences, and natural attractions.
    
    Note: This file replaces the original attractions_analyzer.py with the fixed version,
    incorporating improvements and new visualization styles.
    """
    
    def __init__(self, sentiment_analyzer=None, output_dir='outputs/attraction_analysis'):
        """
        Initialize the attraction analyzer.
        
        Parameters:
        - sentiment_analyzer: SentimentAnalyzer instance for text analysis
        - output_dir: Directory to save analysis outputs
        """
        self.sentiment_analyzer = sentiment_analyzer
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")
            
        # Create visualizations directory
        self.viz_dir = os.path.join(self.output_dir, 'visualizations')
        if not os.path.exists(self.viz_dir):
            os.makedirs(self.viz_dir)
            
        # Define attraction categories and keywords
        self.attraction_categories = {
            'nature': ['beach', 'island', 'waterfall', 'cave', 'park', 'mountain', 'garden', 'reef', 'lagoon', 'wildlife'],
            'culture': ['museum', 'palace', 'history', 'culture', 'art', 'tradition', 'heritage', 'monument', 'church'],
            'adventure': ['snorkel', 'dive', 'kayak', 'hike', 'swim', 'tour', 'boat', 'sail', 'fishing', 'surfing'],
            'leisure': ['relax', 'spa', 'lounge', 'walk', 'view', 'vista', 'sunset', 'photo', 'shopping'],
            'entertainment': ['show', 'performance', 'dance', 'music', 'concert', 'festival', 'event']
        }
            
    def filter_attraction_reviews(self, df):
        """
        Filter the DataFrame to include only attraction reviews.
        
        Parameters:
        - df: DataFrame with reviews
        
        Returns:
        - DataFrame with only attraction reviews
        """
        # Check if 'category' column exists
        if 'category' not in df.columns:
            print("Error: No 'category' column found in data")
            return df.head(0)  # Return empty DataFrame with same columns
            
        # Filter to attraction categories
        filtered_df = df[df['category'].str.lower() == 'attraction'].copy()
        
        if len(filtered_df) == 0:
            # Try alternative method if no attractions found
            keywords = ['tour', 'attraction', 'activity', 'sight', 'beach',
                      'island', 'park', 'museum', 'cave']
            pattern = '|'.join(keywords)
            mask = df['business_name'].str.lower().str.contains(pattern, na=False)
            filtered_df = df[mask].copy()
        
        print(f"Filtered to {len(filtered_df)} attraction reviews")
        return filtered_df
        
    def run_analysis(self, df):
        """
        Run a complete attraction analysis.
        
        Parameters:
        - df: DataFrame with review data
        
        Returns:
        - Dictionary with analysis results
        """
        print("\nRunning attraction analysis...")
        attraction_df = self.filter_attraction_reviews(df)
        print(f"Analyzing {len(attraction_df)} attraction reviews")

        # Run analyses
        results = {
            'activity_types': self.analyze_activity_types(attraction_df),
            'experience_aspects': self.analyze_experience_aspects(attraction_df),
            'unique_features': self.analyze_unique_features(attraction_df)
        }

        # Generate visualizations
        self.generate_visualizations(attraction_df)

        # Save detailed results
        self.save_results(results)

        return results
            'adventure': ['snorkel', 'dive', 'kayak', 'hike', 'swim', 'tour', 'boat', 'sail', 'fishing', 'surfing'],
            'entertainment': ['show', 'performance', 'dance', 'music', 'festival', 'concert', 'party', 'nightlife'],
            'relaxation': ['spa', 'massage', 'retreat', 'resort', 'relax']
        }
        
        # Define activity types
        self.activity_types = {
            'water_activities': ['swim', 'snorkel', 'dive', 'kayak', 'paddle', 'boat', 'sail', 'surf', 'fishing'],
            'land_activities': ['hike', 'walk', 'bike', 'trek', 'climb', 'tour', 'drive', 'ride'],
            'cultural_activities': ['dance', 'music', 'craft', 'cooking', 'ceremony', 'tradition', 'festival'],
            'sightseeing': ['view', 'sight', 'photo', 'landscape', 'panorama', 'lookout', 'overlook'],
            'educational': ['learn', 'guide', 'history', 'information', 'museum', 'exhibit']
        }
        
        # Define aspects to analyze
        self.aspects = {
            'staff': ['staff', 'guide', 'instructor', 'employee', 'service', 'friendly', 'helpful', 'knowledgeable'],
            'value': ['value', 'price', 'worth', 'expensive', 'cheap', 'cost', 'fee', 'money'],
            'experience': ['experience', 'memorable', 'amazing', 'awesome', 'excellent', 'great', 'fantastic', 'fun'],
            'safety': ['safety', 'safe', 'secure', 'danger', 'risk', 'equipment', 'instruction', 'precaution'],
            'amenities': ['facility', 'amenity', 'clean', 'bathroom', 'toilet', 'shower', 'change', 'parking']
        }
    
    def categorize_attraction(self, text):
        """
        Categorize attraction based on review text.
        
        Parameters:
        - text: Review text to analyze
        
        Returns:
        - Dict with category scores
        """
        if not isinstance(text, str):
            return {}
        
        text = text.lower()
        scores = {}
        
        for category, keywords in self.attraction_categories.items():
            score = sum(1 for keyword in keywords if keyword in text)
            scores[category] = score
        
        return scores
    
    def identify_activities(self, text):
        """
        Identify activities mentioned in review text.
        
        Parameters:
        - text: Review text to analyze
        
        Returns:
        - Dict with activity type scores
        """
        if not isinstance(text, str):
            return {}
        
        text = text.lower()
        scores = {}
        
        for activity_type, keywords in self.activity_types.items():
            score = sum(1 for keyword in keywords if keyword in text)
            scores[activity_type] = score
        
        return scores
    
    def analyze_aspects(self, text):
        """
        Analyze specific aspects mentioned in review text.
        
        Parameters:
        - text: Review text to analyze
        
        Returns:
        - Dict with aspect scores and sentiment
        """
        if not isinstance(text, str) or self.sentiment_analyzer is None:
            return {}
        
        text = text.lower()
        aspect_analysis = {}
        
        for aspect, keywords in self.aspects.items():
            # Check if any of the keywords are mentioned
            mentioned = any(keyword in text for keyword in keywords)
            
            if mentioned:
                # Extract sentences containing the aspect keywords
                sentences = re.split(r'[.!?]+', text)
                relevant_sentences = []
                
                for sentence in sentences:
                    if any(keyword in sentence for keyword in keywords):
                        relevant_sentences.append(sentence)
                
                if relevant_sentences:
                    # Analyze sentiment of the relevant sentences
                    combined_text = '. '.join(relevant_sentences)
                    sentiment = self.sentiment_analyzer.analyze_text_sentiment(combined_text)
                    
                    aspect_analysis[aspect] = {
                        'mentioned': True,
                        'sentiment': sentiment,
                        'sentences': relevant_sentences
                    }
                else:
                    aspect_analysis[aspect] = {'mentioned': True, 'sentiment': 0}
            else:
                aspect_analysis[aspect] = {'mentioned': False}
        
        return aspect_analysis
    
    def extract_seasonal_patterns(self, df):
        """
        Extract seasonal patterns from reviews.
        
        Parameters:
        - df: DataFrame of reviews with dates
        
        Returns:
        - Dict with seasonal patterns
        """
        if 'published_date' not in df.columns:
            return {}
        
        # Convert to datetime and extract season
        df['date'] = pd.to_datetime(df['published_date'], errors='coerce')
        
        # Filter out rows with invalid dates
        df = df.dropna(subset=['date'])
        
        # Define seasons based on month
        def get_season(month):
            if month in [12, 1, 2]:  # Summer in Southern Hemisphere
                return 'Summer'
            elif month in [3, 4, 5]:  # Fall in Southern Hemisphere
                return 'Fall'
            elif month in [6, 7, 8]:  # Winter in Southern Hemisphere
                return 'Winter'
            else:  # Spring in Southern Hemisphere
                return 'Spring'
        
        df['season'] = df['date'].dt.month.apply(get_season)
        
        # Analyze by season
        seasonal_stats = {}
        
        for season, season_df in df.groupby('season'):
            # Skip if too few reviews
            if len(season_df) < 5:
                continue
                
            # Get average ratings and sentiment
            avg_rating = season_df['rating'].mean() if 'rating' in season_df.columns else None
            
            if 'sentiment_score' in season_df.columns:
                avg_sentiment = season_df['sentiment_score'].mean()
                sentiment_std = season_df['sentiment_score'].std()
            else:
                avg_sentiment = None
                sentiment_std = None
            
            # Get popular activities in this season
            season_activities = {}
            for activity_type in self.activity_types:
                if activity_type in season_df.columns:
                    season_activities[activity_type] = season_df[activity_type].sum()
            
            seasonal_stats[season] = {
                'review_count': len(season_df),
                'avg_rating': avg_rating,
                'avg_sentiment': avg_sentiment,
                'sentiment_std': sentiment_std,
                'popular_activities': season_activities,
                'top_attractions': season_df['name'].value_counts().head(5).to_dict()
            }
        
        return seasonal_stats
    
    def analyze_unique_features(self, df):
        """
        Analyze unique or distinctive features of Tongan attractions.
        
        Parameters:
        - df: DataFrame of reviews
        
        Returns:
        - Dict with unique features analysis
        """
        if 'text' not in df.columns:
            return {}
        
        # Define Tongan-specific cultural and natural features
        tongan_features = {
            'cultural': ['kava', 'tapa', 'ceremony', 'tongan culture', 'traditional', 'dance', 'feast', 'umu'],
            'natural': ['whale', 'coral', 'beach', 'island', 'volcano', 'cave', 'flying fox', 'tropical'],
            'activities': ['swimming with whales', 'volcano hiking', 'cave swimming', 'royal palace', 'blowholes']
        }
        
        # Analyze mention frequency and sentiment
        feature_analysis = {}
        
        for category, features in tongan_features.items():
            feature_mentions = {}
            
            for feature in features:
                # Count mentions
                mentions = df['text'].str.lower().str.contains(feature, na=False).sum()
                
                if mentions > 0:
                    # Get reviews mentioning this feature
                    feature_reviews = df[df['text'].str.lower().str.contains(feature, na=False)]
                    
                    # Get average sentiment if available
                    if 'sentiment_score' in feature_reviews.columns:
                        avg_sentiment = feature_reviews['sentiment_score'].mean()
                    else:
                        avg_sentiment = None
                    
                    # Get average rating if available
                    if 'rating' in feature_reviews.columns:
                        avg_rating = feature_reviews['rating'].mean()
                    else:
                        avg_rating = None
                    
                    feature_mentions[feature] = {
                        'mentions': mentions,
                        'percentage': (mentions / len(df)) * 100,
                        'avg_sentiment': avg_sentiment,
                        'avg_rating': avg_rating
                    }
            
            feature_analysis[category] = feature_mentions
        
        return feature_analysis
    
    def run_analysis(self, df):
        """
        Run complete attraction analysis.
        
        Parameters:
        - df: DataFrame of attraction reviews
        
        Returns:
        - Dict with analysis results
        """
        print("Running attraction analysis...")
        
        if df is None or df.empty:
            print("Error: No data provided for analysis")
            return {}
            
        # Check required columns
        required_columns = ['text', 'rating']
        for col in required_columns:
            if col not in df.columns:
                print(f"Warning: Required column '{col}' not found")
        
        # Make a copy to avoid modifying the original
        analysis_df = df.copy()
        
        # Apply sentiment analysis if not already done and sentiment analyzer is available
        if 'sentiment_score' not in analysis_df.columns and self.sentiment_analyzer is not None:
            print("Applying sentiment analysis...")
            analysis_df = self.sentiment_analyzer.analyze_sentiment(analysis_df)
        
        # Categorize attractions
        print("Categorizing attractions...")
        for category in self.attraction_categories:
            analysis_df[category] = 0
        
        for i, row in analysis_df.iterrows():
            if isinstance(row.get('text'), str):
                category_scores = self.categorize_attraction(row['text'])
                for category, score in category_scores.items():
                    analysis_df.at[i, category] = score
        
        # Identify activities
        print("Identifying activities...")
        for activity_type in self.activity_types:
            analysis_df[activity_type] = 0
        
        for i, row in analysis_df.iterrows():
            if isinstance(row.get('text'), str):
                activity_scores = self.identify_activities(row['text'])
                for activity_type, score in activity_scores.items():
                    analysis_df.at[i, activity_type] = score
        
        # Compile results
        results = {
            'review_count': len(analysis_df),
            'avg_rating': analysis_df['rating'].mean() if 'rating' in analysis_df.columns else None,
            'category_counts': {},
            'attraction_categories': {},
            'popular_activities': {},
            'aspect_analysis': {},
            'seasonal_patterns': self.extract_seasonal_patterns(analysis_df),
            'unique_features': self.analyze_unique_features(analysis_df)
        }
        
        # Attraction categories summary
        for category in self.attraction_categories:
            # Count reviews mentioning this category
            count = (analysis_df[category] > 0).sum()
            results['category_counts'][category] = count
            
            if count > 0:
                # Get average rating for this category
                cat_reviews = analysis_df[analysis_df[category] > 0]
                avg_rating = cat_reviews['rating'].mean() if 'rating' in cat_reviews.columns else None
                
                # Get average sentiment for this category
                avg_sentiment = cat_reviews['sentiment_score'].mean() if 'sentiment_score' in cat_reviews.columns else None
                
                results['attraction_categories'][category] = {
                    'count': count,
                    'percentage': (count / len(analysis_df)) * 100,
                    'avg_rating': avg_rating,
                    'avg_sentiment': avg_sentiment,
                    'top_attractions': cat_reviews['name'].value_counts().head(5).to_dict() if 'name' in cat_reviews.columns else {}
                }
        
        # Popular activities summary
        for activity_type in self.activity_types:
            # Count reviews mentioning this activity
            count = (analysis_df[activity_type] > 0).sum()
            
            if count > 0:
                # Get average rating for this activity
                act_reviews = analysis_df[analysis_df[activity_type] > 0]
                avg_rating = act_reviews['rating'].mean() if 'rating' in act_reviews.columns else None
                
                # Get average sentiment for this activity
                avg_sentiment = act_reviews['sentiment_score'].mean() if 'sentiment_score' in act_reviews.columns else None
                
                results['popular_activities'][activity_type] = {
                    'count': count,
                    'percentage': (count / len(analysis_df)) * 100,
                    'avg_rating': avg_rating,
                    'avg_sentiment': avg_sentiment,
                    'top_attractions': act_reviews['name'].value_counts().head(5).to_dict() if 'name' in act_reviews.columns else {}
                }
        
        # Aspect analysis (if sentiment analyzer is available)
        if self.sentiment_analyzer is not None:
            aspect_mentions = {aspect: 0 for aspect in self.aspects}
            aspect_sentiments = {aspect: [] for aspect in self.aspects}
            
            # Analyze each review for aspects
            for i, row in analysis_df.iterrows():
                if isinstance(row.get('text'), str):
                    aspect_analysis = self.analyze_aspects(row['text'])
                    
                    for aspect, data in aspect_analysis.items():
                        if data.get('mentioned', False):
                            aspect_mentions[aspect] += 1
                            
                            if 'sentiment' in data:
                                aspect_sentiments[aspect].append(data['sentiment'])
            
            # Compile aspect analysis results
            for aspect in self.aspects:
                count = aspect_mentions[aspect]
                
                if count > 0:
                    avg_sentiment = sum(aspect_sentiments[aspect]) / len(aspect_sentiments[aspect]) if aspect_sentiments[aspect] else None
                    
                    results['aspect_analysis'][aspect] = {
                        'count': count,
                        'percentage': (count / len(analysis_df)) * 100,
                        'avg_sentiment': avg_sentiment
                    }
        
        # Generate visualizations
        self.generate_visualizations(analysis_df, results)
        
        # Convert numpy values to Python natives for JSON serialization
        def convert_to_serializable(obj):
            import numpy as np
            import pandas as pd
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.core.series.Series):
                return obj.to_list()
            return obj
            
        # Save results to JSON (convert numpy types to Python natives first)
        results_serializable = json.loads(
            json.dumps(results, default=convert_to_serializable)
        )
        
        with open(os.path.join(self.output_dir, 'attraction_analysis_results.json'), 'w') as f:
            json.dump(results_serializable, f, indent=4)
        
        print("Attraction analysis complete!")
        return results
    
    def generate_visualizations(self, df, results):
        """
        Generate visualizations from analysis results.
        
        Parameters:
        - df: DataFrame with analyzed data
        - results: Dict of analysis results
        """
        print("Generating attraction analysis visualizations...")
        
        # Apply consistent visualization style
        set_visualization_style()
        
        # 1. Activity Analysis
        if results['popular_activities']:
            plt.figure(figsize=(12, 8))
            
            # Prepare data for plotting
            activities = []
            counts = []
            sentiments = []
            
            for activity, data in results['popular_activities'].items():
                activities.append(activity.replace('_', ' ').title())
                counts.append(data['count'])
                if data.get('avg_sentiment') is not None:
                    sentiments.append(data['avg_sentiment'])
                else:
                    sentiments.append(0)
            
            # Create color gradient based on sentiment
            colors = plt.cm.RdYlGn(np.array(sentiments) * 0.5 + 0.5)  # Scale from -1:1 to 0:1
            
            # Create horizontal bar chart
            y_pos = np.arange(len(activities))
            plt.barh(y_pos, counts, color=colors)
            plt.yticks(y_pos, activities)
            plt.xlabel('Number of Reviews Mentioning')
            plt.title('Popular Activities in Tonga')
            
            # Add sentiment labels
            for i, v in enumerate(counts):
                sentiment_text = f"{sentiments[i]:.2f}"
                plt.text(v + 0.5, i, sentiment_text, color='black', va='center')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, 'activity_analysis.png'), dpi=300)
            plt.close()
        
        # 2. Aspect Analysis
        if results['aspect_analysis']:
            plt.figure(figsize=(12, 8))
            
            # Prepare data for plotting
            aspects = []
            percentages = []
            sentiments = []
            
            for aspect, data in results['aspect_analysis'].items():
                aspects.append(aspect.title())
                percentages.append(data['percentage'])
                if data.get('avg_sentiment') is not None:
                    sentiments.append(data['avg_sentiment'])
                else:
                    sentiments.append(0)
            
            # Create color gradient based on sentiment
            colors = plt.cm.RdYlGn(np.array(sentiments) * 0.5 + 0.5)  # Scale from -1:1 to 0:1
            
            # Create horizontal bar chart
            y_pos = np.arange(len(aspects))
            plt.barh(y_pos, percentages, color=colors)
            plt.yticks(y_pos, aspects)
            plt.xlabel('Percentage of Reviews Mentioning')
            plt.title('Important Aspects of Attraction Experiences')
            
            # Add sentiment labels
            for i, v in enumerate(percentages):
                sentiment_text = f"{sentiments[i]:.2f}"
                plt.text(v + 0.5, i, sentiment_text, color='black', va='center')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, 'aspect_analysis.png'), dpi=300)
            plt.close()
        
        # 3. Seasonal Patterns
        if results['seasonal_patterns']:
            plt.figure(figsize=(14, 8))
            
            # Prepare data for plotting
            seasons = []
            counts = []
            sentiments = []
            
            # Define the seasonal order
            season_order = ['Summer', 'Fall', 'Winter', 'Spring']
            
            # Collect data for sorted seasons
            for season in season_order:
                if season in results['seasonal_patterns']:
                    data = results['seasonal_patterns'][season]
                    seasons.append(season)
                    counts.append(data['review_count'])
                    if data.get('avg_sentiment') is not None:
                        sentiments.append(data['avg_sentiment'])
                    else:
                        sentiments.append(0)
            
            # Create a DataFrame for easier plotting
            seasonal_df = pd.DataFrame({
                'Season': seasons,
                'Reviews': counts,
                'Sentiment': sentiments
            })
            
            # Create subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Plot 1: Review counts by season
            bar_colors = plt.cm.viridis(np.linspace(0, 1, len(seasons)))
            ax1.bar(seasonal_df['Season'], seasonal_df['Reviews'], color=bar_colors)
            ax1.set_title('Number of Reviews by Season')
            ax1.set_ylabel('Number of Reviews')
            
            # Plot 2: Sentiment by season
            sentiment_colors = plt.cm.RdYlGn(seasonal_df['Sentiment'] * 0.5 + 0.5)
            ax2.bar(seasonal_df['Season'], seasonal_df['Sentiment'], color=sentiment_colors)
            ax2.set_title('Average Sentiment by Season')
            ax2.set_ylabel('Sentiment Score')
            
            # Add sentiment labels to the second plot
            for i, v in enumerate(seasonal_df['Sentiment']):
                sentiment_text = f"{v:.2f}"
                ax2.text(i, v + 0.02, sentiment_text, ha='center')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, 'seasonal_patterns.png'), dpi=300)
            plt.close()
        
        # 4. Unique Features Analysis
        if results['unique_features']:
            plt.figure(figsize=(14, 10))
            
            # Create subplots for each category
            num_cats = len(results['unique_features'])
            fig, axes = plt.subplots(num_cats, 1, figsize=(12, 4 * num_cats))
            
            if num_cats == 1:
                axes = [axes]  # Ensure axes is a list for single category
            
            for i, (category, features) in enumerate(results['unique_features'].items()):
                # Prepare data for plotting
                feature_names = []
                feature_pcts = []
                sentiments = []
                
                for feature, data in features.items():
                    feature_names.append(feature.title())
                    feature_pcts.append(data['percentage'])
                    if data.get('avg_sentiment') is not None:
                        sentiments.append(data['avg_sentiment'])
                    else:
                        sentiments.append(0)
                
                # Skip if empty
                if not feature_names:
                    continue
                
                # Sort by percentage
                sorted_indices = np.argsort(feature_pcts)[::-1]
                feature_names = [feature_names[i] for i in sorted_indices]
                feature_pcts = [feature_pcts[i] for i in sorted_indices]
                sentiments = [sentiments[i] for i in sorted_indices]
                
                # Create color gradient based on sentiment
                colors = plt.cm.RdYlGn(np.array(sentiments) * 0.5 + 0.5)
                
                # Create horizontal bar chart
                y_pos = np.arange(len(feature_names))
                axes[i].barh(y_pos, feature_pcts, color=colors)
                axes[i].set_yticks(y_pos)
                axes[i].set_yticklabels(feature_names)
                axes[i].set_xlabel('Percentage of Reviews Mentioning')
                axes[i].set_title(f'Unique {category.title()} Features')
                
                # Add sentiment labels
                for j, v in enumerate(feature_pcts):
                    sentiment_text = f"{sentiments[j]:.2f}"
                    axes[i].text(v + 0.5, j, sentiment_text, va='center')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, 'unique_features_analysis.png'), dpi=300)
            plt.close()
        
        print("Attraction visualizations complete!")
    
    def analyze_text_sentiment(self, text):
        """
        Analyze sentiment of text using the sentiment analyzer.
        
        Parameters:
        - text: Text to analyze
        
        Returns:
        - Sentiment score (-1 to 1)
        """
        if self.sentiment_analyzer is None or not isinstance(text, str):
            return 0
        
        return self.sentiment_analyzer.analyze_text_sentiment(text)
    
    def create_wordcloud(self, text, mask=None):
        """
        Create a word cloud visualization from text.
        
        Parameters:
        - text: Text to visualize
        - mask: Optional shape mask for the word cloud
        
        Returns:
        - WordCloud object
        """
        if not isinstance(text, str) or not text.strip():
            return None
        
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=200,
            contour_width=1,
            mask=mask
        ).generate(text)
        
        return wordcloud
        
    def run_island_analysis(self, df=None, island_col='island_category'):
        """
        Run attraction analysis segmented by island groups.
        
        Parameters:
        - df: DataFrame with all reviews that includes island classification
        - island_col: Column name containing island categorization
        
        Returns:
        - Dictionary with analysis results by island
        """
        print("\n=== Running Island-Based Attraction Analysis ===")
        
        # Check if we have data
        if df is None:
            if self.sentiment_analyzer is not None and hasattr(self.sentiment_analyzer, 'reviews_df'):
                df = self.sentiment_analyzer.reviews_df
            else:
                print("No data available for attraction analysis")
                return None
        
        if df is None or len(df) == 0:
            print("No data available for attraction analysis")
            return None
        
        # Check if we have the island column
        if island_col not in df.columns:
            print(f"Island column '{island_col}' not found in the data. Please run island analysis first.")
            return None
        
        # Filter to attraction reviews
        all_attraction_df = self.filter_attraction_reviews(df)
        
        if len(all_attraction_df) == 0:
            print("No attraction reviews found")
            return None
        
        print(f"Found {len(all_attraction_df)} attraction reviews across all islands")
        
        # Get list of islands with enough attraction reviews
        island_counts = all_attraction_df[island_col].value_counts()
        valid_islands = island_counts[island_counts >= 10].index.tolist()
        
        if not valid_islands:
            print("No islands with sufficient attraction reviews for analysis")
            return self.run_analysis(all_attraction_df)  # Fall back to regular analysis
        
        print(f"Islands with sufficient attraction reviews: {', '.join(valid_islands)}")
        
        # Create a directory for island-specific outputs
        island_output_dir = os.path.join(self.output_dir, 'island_analysis')
        if not os.path.exists(island_output_dir):
            os.makedirs(island_output_dir)
        
        # Store results for each island
        island_results = {}
        island_stats_comparison = {
            'activity_types': {},
            'experience_aspects': {},
            'unique_features': {},
            'avg_ratings': {}
        }
        
        # Run analysis for each island
        for island in valid_islands:
            print(f"\n--- Analyzing attractions on {island} ---")
            
            # Create island-specific output directory
            island_dir = os.path.join(island_output_dir, island.replace(' ', '_').lower())
            if not os.path.exists(island_dir):
                os.makedirs(island_dir)
            
            # Save original output dir and temporarily set to island dir
            original_output_dir = self.output_dir
            self.output_dir = island_dir
            
            # Filter to this island's attraction reviews
            island_df = all_attraction_df[all_attraction_df[island_col] == island].copy()
            
            print(f"Analyzing {len(island_df)} attraction reviews for {island}")
            
            # Run analyses for this island
            activity_stats = self.analyze_activity_types(island_df)
            aspect_stats = self.analyze_experience_aspects(island_df)
            feature_stats = self.analyze_unique_features(island_df)
            
            # Generate island-specific visualizations
            viz_dir = os.path.join(island_dir, 'visualizations')
            if not os.path.exists(viz_dir):
                os.makedirs(viz_dir)
            
            self.generate_visualizations(island_df)
            
            # Store results for this island
            island_results[island] = {
                'review_count': len(island_df),
                'avg_rating': island_df['rating'].mean() if 'rating' in island_df.columns else None,
                'activity_types': activity_stats,
                'experience_aspects': aspect_stats,
                'unique_features': feature_stats
            }
            
            # Restore original output directory
            self.output_dir = original_output_dir
        
        # Save combined results
        results_file = os.path.join(island_output_dir, 'island_attraction_summary.json')
        with open(results_file, 'w') as f:
            json.dump(island_results, f, indent=2)
        
        # Create island comparison Excel report
        self._save_island_results(island_results, island_output_dir)
        
        # Generate cross-island visualizations
        self.visualize_island_attraction_comparisons(island_results, island_output_dir)
        
        return island_results
    
    def _save_island_results(self, island_results, output_dir):
        """
        Save island analysis results to Excel for easy comparison.
        
        Parameters:
        - island_results: Dictionary with results by island
        - output_dir: Directory to save the comparison file
        """
        try:
            # Create summary of island results
            summary = {}
            for island, data in island_results.items():
                summary[island] = {
                    'review_count': data['review_count'],
                    'avg_rating': data['avg_rating'],
                    'top_activities': self._get_top_items(data['activity_types'], 'review_count', 3),
                    'top_aspects': self._get_top_items(data['experience_aspects'], 'avg_sentiment', 3),
                    'unique_features': self._get_top_items(data['unique_features'], 'review_count', 3)
                }
            
            # Create Excel comparison file
            excel_path = os.path.join(output_dir, 'island_attraction_comparison.xlsx')
            writer = pd.ExcelWriter(excel_path, engine='xlsxwriter')
            
            # Sheet 1: Overview
            overview_data = []
            for island, data in summary.items():
                row = {
                    'Island': island,
                    'Review Count': data['review_count'],
                    'Average Rating': round(data['avg_rating'], 2) if data['avg_rating'] else 'N/A',
                    'Top Activities': ', '.join(data['top_activities']),
                    'Top Experience Aspects': ', '.join(data['top_aspects']),
                    'Top Unique Features': ', '.join(data['unique_features'])
                }
                overview_data.append(row)
            
            overview_df = pd.DataFrame(overview_data)
            overview_df.to_excel(writer, sheet_name='Overview', index=False)
            
            # Sheet 2: Activity Types
            activity_data = []
            for island, results in island_results.items():
                for activity, stats in results['activity_types'].items():
                    row = {
                        'Island': island,
                        'Activity Type': activity,
                        'Mentions': stats.get('review_count', 0),
                        'Sentiment': stats.get('avg_sentiment', 'N/A'),
                        'Rating': stats.get('avg_rating', 'N/A')
                    }
                    activity_data.append(row)
            
            if activity_data:
                activity_df = pd.DataFrame(activity_data)
                activity_df.to_excel(writer, sheet_name='Activity Types', index=False)
            
            # Sheet 3: Experience Aspects
            aspect_data = []
            for island, results in island_results.items():
                for aspect, stats in results['experience_aspects'].items():
                    row = {
                        'Island': island,
                        'Experience Aspect': aspect,
                        'Mentions': stats.get('review_count', 0),
                        'Sentiment': stats.get('avg_sentiment', 'N/A'),
                        'Rating': stats.get('avg_rating', 'N/A')
                    }
                    aspect_data.append(row)
            
            if aspect_data:
                aspect_df = pd.DataFrame(aspect_data)
                aspect_df.to_excel(writer, sheet_name='Experience Aspects', index=False)
            
            # Sheet 4: Unique Features
            feature_data = []
            for island, results in island_results.items():
                for feature, stats in results['unique_features'].items():
                    row = {
                        'Island': island,
                        'Unique Feature': feature,
                        'Mentions': stats.get('review_count', 0),
                        'Sentiment': stats.get('avg_sentiment', 'N/A')
                    }
                    feature_data.append(row)
            
            if feature_data:
                feature_df = pd.DataFrame(feature_data)
                feature_df.to_excel(writer, sheet_name='Unique Features', index=False)
            
            # Save and close
            writer.close()
            print(f"Island comparison Excel report saved to {excel_path}")
        
        except Exception as e:
            print(f"Error creating Excel report: {e}")
            
            # Create CSV files instead
            csv_dir = os.path.join(output_dir, 'csv_reports')
            if not os.path.exists(csv_dir):
                os.makedirs(csv_dir)
            
            # Save summary as CSV
            summary_rows = []
            for island, data in summary.items():
                row = {'Island': island}
                row.update({k: v for k, v in data.items() if not isinstance(v, list)})
                summary_rows.append(row)
            
            if summary_rows:
                pd.DataFrame(summary_rows).to_csv(os.path.join(csv_dir, 'island_summary.csv'), index=False)
                print(f"Island summary CSV saved to {csv_dir}")
    
    def _get_top_items(self, data_dict, sort_key, limit=3):
        """Helper to get top items from a dictionary based on a sort key."""
        items = [(k, v.get(sort_key, 0)) for k, v in data_dict.items()]
        items.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in items[:limit]]
    
    def visualize_island_attraction_comparisons(self, island_results, output_dir):
        """
        Create side-by-side visualizations comparing attraction features across islands.
        
        Parameters:
        - island_results: Dictionary with results by island from run_island_analysis
        - output_dir: Directory to save the comparison visualizations
        """
        print("\nGenerating cross-island attraction comparison visualizations...")
        
        # Create visualizations directory
        viz_dir = os.path.join(output_dir, 'visualizations')
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        
        # Get islands with enough data
        valid_islands = [island for island, data in island_results.items() 
                        if data['review_count'] >= 10]
        
        if len(valid_islands) < 2:
            print("Not enough islands with sufficient data for comparison")
            return
        
        # Set visualization style
        set_visualization_style()
        
        # Get color palette for islands
        island_colors = {island: ISLAND_COLORS.get(island.lower(), '#AAAAAA') for island in valid_islands}
        
        # 1. Activity type comparison across islands
        plt.figure(figsize=(12, 8))
        
        # Collect activity data across islands
        activity_data = {}
        for island, data in island_results.items():
            if island not in valid_islands:
                continue
                
            for activity, stats in data['activity_types'].items():
                if activity not in activity_data:
                    activity_data[activity] = {}
                activity_data[activity][island] = stats.get('review_count', 0)
        
        # Create a bar chart for activities
        activities = sorted(activity_data.keys())
        island_list = sorted(valid_islands)
        
        if activities and island_list:
            # Create DataFrame for easier plotting
            activity_df = pd.DataFrame({activity: [activity_data[activity].get(island, 0) 
                                                for island in island_list] 
                                        for activity in activities}, 
                                    index=island_list)
            
            # Stacked bar chart
            ax = activity_df.plot(kind='bar', stacked=True, figsize=(12, 6), 
                                color=sns.color_palette('tab10', len(activities)))
            plt.title('Activity Types by Island', fontsize=14, pad=20)
            plt.xlabel('Island', fontsize=12)
            plt.ylabel('Number of Mentions', fontsize=12)
            plt.legend(title='Activity Type', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'activity_types_by_island.png'), dpi=300)
            plt.close()
            
            # Proportional stacked bar chart
            activity_prop_df = activity_df.div(activity_df.sum(axis=1), axis=0) * 100
            ax = activity_prop_df.plot(kind='bar', stacked=True, figsize=(12, 6), 
                                     color=sns.color_palette('tab10', len(activities)))
            plt.title('Activity Types Proportion by Island', fontsize=14, pad=20)
            plt.xlabel('Island', fontsize=12)
            plt.ylabel('Percentage of Mentions', fontsize=12)
            plt.legend(title='Activity Type', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'activity_types_proportion_by_island.png'), dpi=300)
            plt.close()
        
        # 2. Experience aspects comparison across islands
        aspect_data = {}
        for island, data in island_results.items():
            if island not in valid_islands:
                continue
                
            for aspect, stats in data['experience_aspects'].items():
                if aspect not in aspect_data:
                    aspect_data[aspect] = {}
                aspect_data[aspect][island] = stats.get('review_count', 0)
        
        # Create bar charts for aspects
        aspects = sorted(aspect_data.keys())
        
        if aspects and island_list:
            # Create DataFrame for easier plotting
            aspect_df = pd.DataFrame({aspect: [aspect_data[aspect].get(island, 0) 
                                            for island in island_list] 
                                    for aspect in aspects}, 
                                   index=island_list)
            
            # Stacked bar chart
            ax = aspect_df.plot(kind='bar', stacked=True, figsize=(12, 6), 
                              color=sns.color_palette('Set2', len(aspects)))
            plt.title('Experience Aspects by Island', fontsize=14, pad=20)
            plt.xlabel('Island', fontsize=12)
            plt.ylabel('Number of Mentions', fontsize=12)
            plt.legend(title='Experience Aspect', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'experience_aspects_by_island.png'), dpi=300)
            plt.close()
            
            # Proportional stacked bar chart
            aspect_prop_df = aspect_df.div(aspect_df.sum(axis=1), axis=0) * 100
            ax = aspect_prop_df.plot(kind='bar', stacked=True, figsize=(12, 6), 
                                   color=sns.color_palette('Set2', len(aspects)))
            plt.title('Experience Aspects Proportion by Island', fontsize=14, pad=20)
            plt.xlabel('Island', fontsize=12)
            plt.ylabel('Percentage of Mentions', fontsize=12)
            plt.legend(title='Experience Aspect', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'experience_aspects_proportion_by_island.png'), dpi=300)
            plt.close()
        
        # 3. Rating comparison by island and activity type
        rating_data = []
        for island, data in island_results.items():
            if island not in valid_islands:
                continue
                
            for activity, stats in data['activity_types'].items():
                if 'avg_rating' in stats and stats['avg_rating'] is not None:
                    rating_data.append({
                        'Island': island,
                        'Activity Type': activity,
                        'Average Rating': stats['avg_rating'],
                        'Review Count': stats['review_count']
                    })
        
        if rating_data:
            rating_df = pd.DataFrame(rating_data)
            
            # Bar chart
            plt.figure(figsize=(14, 8))
            ax = sns.barplot(x='Island', y='Average Rating', hue='Activity Type', data=rating_df,
                           palette='Set3')
            plt.title('Average Rating by Island and Activity Type', fontsize=14, pad=20)
            plt.xlabel('Island', fontsize=12)
            plt.ylabel('Average Rating', fontsize=12)
            plt.legend(title='Activity Type', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'rating_by_island_and_type_bars.png'), dpi=300)
            plt.close()
            
            # Heatmap
            rating_pivot = rating_df.pivot_table(index='Activity Type', columns='Island', 
                                              values='Average Rating', aggfunc='mean')
            plt.figure(figsize=(12, 8))
            sns.heatmap(rating_pivot, annot=True, cmap='RdYlGn', center=3.5, vmin=1, vmax=5,
                       linewidths=0.5, fmt='.2f')
            plt.title('Average Rating by Island and Activity Type', fontsize=14, pad=20)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'rating_by_island_and_type_heatmap.png'), dpi=300)
            plt.close()
        
        # 4. Top attractions by island visualization
        for island in valid_islands:
            island_df = pd.DataFrame([{'name': k, 'count': v['review_count']} 
                                   for k, v in island_results[island]['unique_features'].items()])
            
            if len(island_df) < 3:
                continue
                
            # Sort by count
            island_df = island_df.sort_values('count', ascending=False).head(10)
            
            # Create bar chart
            plt.figure(figsize=(12, 6))
            bars = plt.barh(island_df['name'], island_df['count'], 
                          color=island_colors.get(island, '#AAAAAA'))
            
            # Add count labels
            for bar in bars:
                width = bar.get_width()
                plt.text(width + 0.3, bar.get_y() + bar.get_height()/2,
                        f'{int(width)}', ha='left', va='center')
            
            plt.title(f'Top Attractions on {island}', fontsize=14, pad=20)
            plt.xlabel('Number of Mentions', fontsize=12)
            plt.gca().invert_yaxis()  # Show highest count at top
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            
            # Save with island name sanitized for filename
            safe_name = island.lower().replace("'", "").replace(" ", "_")
            plt.savefig(os.path.join(viz_dir, f'top_attractions_{safe_name}.png'), dpi=300)
            plt.close()
            
        print(f"Island comparison visualizations saved to {viz_dir}")