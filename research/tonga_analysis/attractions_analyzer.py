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
        
    def save_results(self, results):
        """
        Save analysis results to file, handling complex objects.
        """
        output_file = os.path.join(self.output_dir, 'attraction_analysis_results.json')
        
        def convert_to_serializable(obj):
            """
            Recursively convert objects to a JSON-serializable format.
            """
            if isinstance(obj, np.ndarray):
                return obj.tolist()  # Convert numpy arrays to lists
            elif isinstance(obj, np.generic):
                return obj.item()  # Convert numpy datatypes to native Python types
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict()  # Convert DataFrames to dictionaries
            elif isinstance(obj, pd.Series):
                return obj.to_dict()  # Convert Series to dictionaries
            return obj  # Fallback for all other data types

        # Use a custom function to ensure all objects are serializable
        clean_results = json.loads(json.dumps(results, default=convert_to_serializable))

        # Save the cleaned results to a JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, indent=2)
        
        print(f"Analysis results saved to {output_file}")
    
    def analyze_activity_types(self, df):
        """
        Analyze attraction reviews by activity type.
        
        Parameters:
        - df: DataFrame with attraction reviews
        
        Returns:
        - Dictionary with activity type statistics
        """
        # Check if we have data
        if len(df) == 0:
            return {}
            
        # Define activity types and keywords
        activity_types = {
            'water_activities': ['swim', 'snorkel', 'dive', 'kayak', 'paddle', 'boat', 'cruise', 'sail', 'surf', 'fishing'],
            'land_activities': ['hike', 'walk', 'trek', 'climb', 'tour', 'drive', 'ride', 'bike', 'cycling'],
            'cultural_activities': ['museum', 'history', 'tradition', 'culture', 'ceremony', 'dance', 'music', 'craft'],
            'nature_viewing': ['wildlife', 'bird', 'whale', 'dolphin', 'turtle', 'fish', 'coral', 'reef', 'forest'],
            'relaxation': ['beach', 'relax', 'lounge', 'sunset', 'view', 'scenic', 'photo', 'sightseeing']
        }
        
        # Analyze mentions of each activity type
        activity_stats = {}
        
        for activity, keywords in activity_types.items():
            pattern = '|'.join(keywords)
            # Find reviews mentioning this activity
            mask = df['text'].str.lower().str.contains(pattern, na=False)
            activity_reviews = df[mask]
            
            # Skip if not enough mentions
            if len(activity_reviews) < 5:
                continue
                
            # Calculate statistics
            activity_stats[activity] = {
                'review_count': len(activity_reviews),
                'percentage': len(activity_reviews) / len(df) * 100,
                'avg_sentiment': activity_reviews['sentiment_score'].mean() if 'sentiment_score' in activity_reviews.columns else None,
                'avg_rating': activity_reviews['rating'].mean() if 'rating' in activity_reviews.columns else None,
                'top_keywords': self._extract_top_keywords(activity_reviews, keywords)
            }
            
        return activity_stats
    
    def _extract_top_keywords(self, df, keywords, limit=5):
        """Extract most frequently mentioned keywords from a set."""
        if len(df) == 0:
            return []
            
        text = ' '.join(df['text'].fillna('').str.lower())
        counts = {}
        
        for keyword in keywords:
            counts[keyword] = text.count(keyword)
            
        # Sort by frequency
        sorted_keywords = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return [k for k, v in sorted_keywords[:limit] if v > 0]
    
    def analyze_experience_aspects(self, df):
        """
        Analyze different aspects of visitor experiences.
        
        Parameters:
        - df: DataFrame with attraction reviews
        
        Returns:
        - Dictionary with aspect statistics
        """
        # Check if we have data
        if len(df) == 0:
            return {}
            
        # Define experience aspects and keywords
        experience_aspects = {
            'guide_quality': ['guide', 'tour guide', 'leader', 'instructor', 'captain', 'staff'],
            'safety': ['safe', 'safety', 'equipment', 'secure', 'comfort', 'comfortable'],
            'value': ['price', 'value', 'worth', 'money', 'expensive', 'cheap', 'cost'],
            'scenery': ['beautiful', 'view', 'scenery', 'landscape', 'picture', 'photo'],
            'learning': ['learn', 'educational', 'interesting', 'information', 'history', 'knowledge']
        }
        
        # Analyze mentions of each aspect
        aspect_stats = {}
        
        for aspect, keywords in experience_aspects.items():
            pattern = '|'.join(keywords)
            # Find reviews mentioning this aspect
            mask = df['text'].str.lower().str.contains(pattern, na=False)
            aspect_reviews = df[mask]
            
            # Skip if not enough mentions
            if len(aspect_reviews) < 5:
                continue
                
            # Calculate statistics
            aspect_stats[aspect] = {
                'review_count': len(aspect_reviews),
                'percentage': len(aspect_reviews) / len(df) * 100,
                'avg_sentiment': aspect_reviews['sentiment_score'].mean() if 'sentiment_score' in aspect_reviews.columns else None,
                'avg_rating': aspect_reviews['rating'].mean() if 'rating' in aspect_reviews.columns else None,
                'top_keywords': self._extract_top_keywords(aspect_reviews, keywords)
            }
            
        return aspect_stats
    
    def analyze_unique_features(self, df):
        """
        Identify and analyze unique attraction features.
        
        Parameters:
        - df: DataFrame with attraction reviews
        
        Returns:
        - Dictionary with feature statistics
        """
        # Check if we have data
        if len(df) == 0:
            return {}
            
        # Find unique attractions by business ID or name
        if 'business_id' in df.columns:
            grouped = df.groupby('business_id')
        else:
            grouped = df.groupby('business_name')
            
        # Analyze each unique attraction
        feature_stats = {}
        
        for name, group in grouped:
            # Skip if not enough reviews
            if len(group) < 5:
                continue
                
            # Use business name for the key
            business_name = group['business_name'].iloc[0] if 'business_name' in group.columns else name
            
            # Calculate statistics
            feature_stats[business_name] = {
                'review_count': len(group),
                'avg_rating': group['rating'].mean() if 'rating' in group.columns else None,
                'avg_sentiment': group['sentiment_score'].mean() if 'sentiment_score' in group.columns else None,
                'activity_types': self._identify_activity_types(group)
            }
            
        return feature_stats
    
    def _identify_activity_types(self, df):
        """Identify main activity types for an attraction."""
        if len(df) == 0:
            return []
            
        # Define activity types and keywords again (simplified)
        activity_types = {
            'water_activities': ['swim', 'snorkel', 'dive', 'kayak', 'boat', 'sail'],
            'land_activities': ['hike', 'walk', 'tour', 'bike'],
            'cultural': ['museum', 'history', 'culture', 'tradition'],
            'nature': ['wildlife', 'bird', 'whale', 'turtle', 'forest', 'beach']
        }
        
        text = ' '.join(df['text'].fillna('').str.lower())
        counts = {}
        
        for activity, keywords in activity_types.items():
            pattern = '|'.join(keywords)
            counts[activity] = len(re.findall(pattern, text))
            
        # Return top 2 activity types
        sorted_activities = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return [a for a, c in sorted_activities[:2] if c > 0]
    
    def generate_visualizations(self, df):
        """
        Generate visualizations for attraction analysis.
        
        Parameters:
        - df: DataFrame with attraction reviews
        """
        if len(df) == 0:
            print("No data available for visualizations")
            return
            
        # Set visualization style
        set_visualization_style()
        
        # Create empty visualizations directory if it doesn't exist
        if not os.path.exists(self.viz_dir):
            os.makedirs(self.viz_dir)
        
        # 1. Activity Type Analysis Visualization
        activity_stats = self.analyze_activity_types(df)
        
        if activity_stats:
            plt.figure(figsize=(10, 6))
            
            # Sort activities by review count
            activities = sorted(activity_stats.items(), key=lambda x: x[1]['review_count'], reverse=True)
            
            # Create bar chart for activity mentions
            x = range(len(activities))
            counts = [a[1]['review_count'] for a in activities]
            labels = [a[0].replace('_', ' ').title() for a in activities]
            
            # Set up colors
            colors = sns.color_palette('tab10', len(activities))
            
            bars = plt.bar(x, counts, color=colors)
            
            # Add count labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')
            
            plt.xticks(x, labels, rotation=45, ha='right')
            plt.xlabel('Activity Type')
            plt.ylabel('Number of Mentions')
            plt.title('Popular Activities Mentioned in Reviews')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            plt.savefig(os.path.join(self.viz_dir, 'activity_analysis.png'), dpi=300)
            plt.close()
        
        # 2. Experience Aspects Analysis Visualization
        aspect_stats = self.analyze_experience_aspects(df)
        
        if aspect_stats:
            plt.figure(figsize=(10, 6))
            
            # Sort aspects by review count
            aspects = sorted(aspect_stats.items(), key=lambda x: x[1]['review_count'], reverse=True)
            
            # Create bar chart for aspect mentions
            x = range(len(aspects))
            counts = [a[1]['review_count'] for a in aspects]
            sentiment = [a[1]['avg_sentiment'] for a in aspects]
            labels = [a[0].replace('_', ' ').title() for a in aspects]
            
            # Set up colors based on sentiment
            colors = []
            for s in sentiment:
                if s > 0.2:
                    colors.append('#57A773')  # Green for positive
                elif s < -0.2:
                    colors.append('#C33C54')  # Red for negative
                else:
                    colors.append('#4A6FA5')  # Blue for neutral
            
            bars = plt.bar(x, counts, color=colors)
            
            # Add count labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')
            
            plt.xticks(x, labels, rotation=45, ha='right')
            plt.xlabel('Experience Aspect')
            plt.ylabel('Number of Mentions')
            plt.title('Experience Aspects Mentioned in Reviews')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            plt.savefig(os.path.join(self.viz_dir, 'aspect_analysis.png'), dpi=300)
            plt.close()
            
        # 3. Unique Features Analysis
        feature_stats = self.analyze_unique_features(df)
        
        if feature_stats and len(feature_stats) >= 3:
            plt.figure(figsize=(12, 8))
            
            # Sort features by review count
            features = sorted(feature_stats.items(), key=lambda x: x[1]['review_count'], reverse=True)[:10]
            
            # Create bar chart for feature mentions
            x = range(len(features))
            counts = [f[1]['review_count'] for f in features]
            labels = [f[0] if len(f[0]) < 20 else f[0][:17] + '...' for f in features]
            
            # Set up colors
            colors = sns.color_palette('tab10', len(features))
            
            bars = plt.barh(x, counts, color=colors)
            
            # Add count labels
            for bar in bars:
                width = bar.get_width()
                plt.text(width + 0.3, bar.get_y() + bar.get_height()/2,
                        f'{int(width)}', ha='left', va='center')
            
            plt.yticks(x, labels)
            plt.xlabel('Number of Reviews')
            plt.title('Top Attractions by Review Count')
            plt.grid(axis='x', alpha=0.3)
            plt.gca().invert_yaxis()  # To show highest count at top
            plt.tight_layout()
            
            plt.savefig(os.path.join(self.viz_dir, 'unique_features_analysis.png'), dpi=300)
            plt.close()
