"""
ARCHIVED VERSION: This is the old version of the attractions analyzer preserved for reference.
DO NOT USE IN PRODUCTION. Use the new version in attractions_analyzer.py instead.
"""

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

class AttractionAnalyzer:
    """
    Specialized analyzer for attraction and activity reviews in Tonga.
    Focuses on tours, activities, cultural experiences, and natural attractions.
    """
    
    def __init__(self, sentiment_analyzer=None, output_dir='outputs/attraction_analysis'):
        """
        Initialize the attraction analyzer.
        
        Parameters:
        - sentiment_analyzer: Instance of SentimentAnalyzer for text processing
        - output_dir: Directory for attraction-specific outputs
        """
        self.sentiment_analyzer = sentiment_analyzer
        self.output_dir = output_dir
        
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created attraction analysis directory: {output_dir}")
            
        # Define attraction-specific categories
        self.activity_types = {
            'water_activities': [
                'swim', 'snorkel', 'dive', 'scuba', 'kayak', 'boat', 'fishing',
                'whale', 'beach', 'coral', 'reef', 'ocean', 'sea', 'underwater'
            ],
            'cultural_experiences': [
                'culture', 'traditional', 'history', 'village', 'ceremony',
                'dance', 'music', 'craft', 'art', 'local', 'heritage', 'kava'
            ],
            'nature_activities': [
                'hike', 'walk', 'trek', 'trail', 'cave', 'forest', 'bird',
                'wildlife', 'nature', 'island', 'botanical', 'garden', 'volcano'
            ],
            'tours_excursions': [
                'tour', 'guide', 'excursion', 'trip', 'sightseeing', 'cruise',
                'charter', 'adventure', 'expedition', 'journey', 'exploration'
            ]
        }
        
        self.experience_aspects = {
            'guide_quality': [
                'guide', 'instructor', 'leader', 'staff', 'teacher', 'expert',
                'professional', 'knowledgeable', 'friendly', 'helpful'
            ],
            'safety_comfort': [
                'safe', 'safety', 'equipment', 'comfortable', 'clean', 'secure',
                'protection', 'condition', 'maintained', 'reliable'
            ],
            'organization': [
                'organized', 'timing', 'schedule', 'punctual', 'efficient',
                'preparation', 'planned', 'arrangement', 'coordination'
            ],
            'education_info': [
                'learn', 'information', 'history', 'fact', 'story', 'education',
                'knowledge', 'explanation', 'detail', 'understanding'
            ],
            'value': [
                'price', 'value', 'worth', 'expensive', 'cheap', 'cost',
                'reasonable', 'fee', 'payment', 'money'
            ]
        }
        
        self.unique_features = [
            'whale', 'coral', 'beach', 'village', 'cave', 'culture',
            'tradition', 'kava', 'island', 'people', 'history'
        ]

    def filter_attraction_reviews(self, df):
        """
        Filter to only attraction reviews and add attraction-specific features.
        """
        attraction_df = df[df['category'] == 'attraction'].copy()
        
        # Add activity type flags
        for activity, keywords in self.activity_types.items():
            pattern = '|'.join(keywords)
            attraction_df[f'activity_{activity}'] = attraction_df['text'].str.lower().str.contains(
                pattern, na=False).astype(int)
        
        # Add experience aspect flags
        for aspect, keywords in self.experience_aspects.items():
            pattern = '|'.join(keywords)
            attraction_df[f'aspect_{aspect}'] = attraction_df['text'].str.lower().str.contains(
                pattern, na=False).astype(int)
        
        # Add unique feature flags
        for feature in self.unique_features:
            attraction_df[f'unique_{feature}'] = attraction_df['text'].str.lower().str.contains(
                feature, na=False).astype(int)
        
        return attraction_df

    def analyze_activity_types(self, df):
        """
        Analyze different types of activities and their reception.
        """
        activity_stats = {}
        
        # Check if sentiment columns exist
        has_sentiment = 'sentiment_score' in df.columns and 'sentiment_category' in df.columns
        
        for activity in self.activity_types.keys():
            activity_reviews = df[df[f'activity_{activity}'] == 1]
            
            if len(activity_reviews) > 0:
                stats = {
                    'review_count': len(activity_reviews)
                }
                
                # Add sentiment metrics if available
                if has_sentiment:
                    stats['avg_sentiment'] = activity_reviews['sentiment_score'].mean()
                    stats['positive_reviews'] = len(activity_reviews[activity_reviews['sentiment_category'] == 'positive'])
                    stats['negative_reviews'] = len(activity_reviews[activity_reviews['sentiment_category'] == 'negative'])
                
                # Add rating if available
                if 'rating' in activity_reviews.columns:
                    stats['avg_rating'] = activity_reviews['rating'].mean()
                
                # Add common phrases
                if 'processed_text' in activity_reviews.columns:
                    stats['common_phrases'] = self.extract_common_phrases(activity_reviews)
                
                activity_stats[activity] = stats
        
        return activity_stats

    def analyze_experience_aspects(self, df):
        """
        Analyze different aspects of the experience and their quality.
        """
        aspect_stats = {}
        
        # Check if sentiment columns exist
        has_sentiment = 'sentiment_score' in df.columns and 'sentiment_category' in df.columns
        
        for aspect in self.experience_aspects.keys():
            aspect_reviews = df[df[f'aspect_{aspect}'] == 1]
            
            if len(aspect_reviews) > 0:
                stats = {
                    'review_count': len(aspect_reviews)
                }
                
                # Add sentiment metrics if available
                if has_sentiment:
                    stats['avg_sentiment'] = aspect_reviews['sentiment_score'].mean()
                    stats['positive_reviews'] = len(aspect_reviews[aspect_reviews['sentiment_category'] == 'positive'])
                    stats['negative_reviews'] = len(aspect_reviews[aspect_reviews['sentiment_category'] == 'negative'])
                
                # Add rating if available
                if 'rating' in aspect_reviews.columns:
                    stats['avg_rating'] = aspect_reviews['rating'].mean()
                
                # Add common phrases
                if 'processed_text' in aspect_reviews.columns:
                    stats['common_phrases'] = self.extract_common_phrases(aspect_reviews)
                
                aspect_stats[aspect] = stats
        
        return aspect_stats

    def analyze_unique_features(self, df):
        """
        Analyze mentions and sentiment around unique Tongan features.
        """
        feature_stats = {}
        
        # Check if sentiment columns exist
        has_sentiment = 'sentiment_score' in df.columns and 'sentiment_category' in df.columns
        
        for feature in self.unique_features:
            feature_reviews = df[df[f'unique_{feature}'] == 1]
            
            if len(feature_reviews) > 0:
                stats = {
                    'review_count': len(feature_reviews)
                }
                
                # Add sentiment metrics if available
                if has_sentiment:
                    stats['avg_sentiment'] = feature_reviews['sentiment_score'].mean()
                    stats['positive_reviews'] = len(feature_reviews[feature_reviews['sentiment_category'] == 'positive'])
                    stats['negative_reviews'] = len(feature_reviews[feature_reviews['sentiment_category'] == 'negative'])
                
                # Add common phrases
                if 'processed_text' in feature_reviews.columns:
                    stats['common_phrases'] = self.extract_common_phrases(feature_reviews)
                
                feature_stats[feature] = stats
        
        return feature_stats

    def analyze_seasonal_patterns(self, df):
        """
        Analyze seasonal patterns in activity reviews.
        """
        seasonal_stats = {}
        
        if 'published_date' in df.columns:
            df['month'] = pd.to_datetime(df['published_date']).dt.month
            
            # Define seasons (for Southern Hemisphere)
            df['season'] = df['month'].apply(lambda x: 
                'Summer' if x in [12, 1, 2] else
                'Autumn' if x in [3, 4, 5] else
                'Winter' if x in [6, 7, 8] else
                'Spring'
            )
            
            # Check if sentiment columns exist
            has_sentiment = 'sentiment_score' in df.columns
            
            # Analyze by season
            for season in ['Summer', 'Autumn', 'Winter', 'Spring']:
                season_reviews = df[df['season'] == season]
                
                if len(season_reviews) > 0:
                    # Get popular activities this season
                    activity_cols = [col for col in df.columns if col.startswith('activity_')]
                    season_activities = season_reviews[activity_cols].sum()
                    top_activities = season_activities.nlargest(3)
                    
                    stats = {
                        'review_count': len(season_reviews),
                        'top_activities': dict(top_activities)
                    }
                    
                    # Add sentiment metrics if available
                    if has_sentiment:
                        stats['avg_sentiment'] = season_reviews['sentiment_score'].mean()
                    
                    # Add common phrases
                    if 'processed_text' in season_reviews.columns:
                        stats['common_phrases'] = self.extract_common_phrases(season_reviews)
                    
                    seasonal_stats[season] = stats
        
        return seasonal_stats

    def plot_seasonal_patterns(self, df, viz_dir, colors):
        """Helper function to plot seasonal patterns analysis."""
        if 'season' in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            season_counts = df['season'].value_counts().sort_index()
            ax.bar(season_counts.index, season_counts.values, color=colors[:len(season_counts)])
            ax.set_title('Seasonal Distribution of Reviews')
            ax.set_xlabel('Season')
            ax.set_ylabel('Number of Reviews')

            # Add labels to bars
            for i, v in enumerate(season_counts.values):
                ax.text(i, v + 5, str(v), color='black', ha='center')

            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'seasonal_patterns.png'), dpi=300)
            plt.close()
            print(f"Seasonal patterns visualization saved to {viz_dir}/seasonal_patterns.png")
        else:
            print("No 'season' column found in DataFrame. Skipping seasonal patterns visualization.")
            
    def analyze_by_traveler_type(self, df):
        """
        Analyze activity preferences by traveler type.
        """
        traveler_stats = {}
        
        if 'trip_type' in df.columns:
            # Check if sentiment columns exist
            has_sentiment = 'sentiment_score' in df.columns
            
            for trip_type in df['trip_type'].unique():
                if pd.isna(trip_type):
                    continue
                    
                traveler_reviews = df[df['trip_type'] == trip_type]
                
                if len(traveler_reviews) >= 5:  # Minimum threshold for analysis
                    # Get preferred activities
                    activity_cols = [col for col in df.columns if col.startswith('activity_')]
                    activity_prefs = traveler_reviews[activity_cols].sum()
                    top_activities = activity_prefs.nlargest(3)
                    
                    stats = {
                        'review_count': len(traveler_reviews),
                        'preferred_activities': dict(top_activities)
                    }
                    
                    # Add sentiment metrics if available
                    if has_sentiment:
                        stats['avg_sentiment'] = traveler_reviews['sentiment_score'].mean()
                    
                    # Add common phrases
                    if 'processed_text' in traveler_reviews.columns:
                        stats['common_phrases'] = self.extract_common_phrases(traveler_reviews)
                    
                    traveler_stats[trip_type] = stats
        
        return traveler_stats

    def extract_common_phrases(self, df, min_count=2):
        """Extract common phrases from a set of reviews."""
        if len(df) == 0 or 'processed_text' not in df.columns:
            return {}
            
        text = ' '.join(df['processed_text'].fillna(''))
        words = text.split()
        word_freq = Counter(words)
        
        return {word: count for word, count in word_freq.items() 
                if count >= min_count}

    def generate_visualizations(self, df):
        """
        Generate attraction-specific visualizations.
        """
        viz_dir = os.path.join(self.output_dir, 'visualizations')
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        
        # Set style
        plt.style.use('classic')
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f1c40f', '#9b59b6']
        
        # 1. Activity Types Analysis
        self.plot_feature_analysis(df, 'activity', self.activity_types.keys(),
                                 'Activity Types', viz_dir, colors)
        
        # 2. Experience Aspects Analysis
        self.plot_feature_analysis(df, 'aspect', self.experience_aspects.keys(),
                                 'Experience Aspects', viz_dir, colors)
        
        # 3. Unique Features Analysis
        self.plot_unique_features(df, viz_dir, colors)
        
        # 4. Seasonal Patterns
        if 'season' in df.columns:
            self.plot_seasonal_patterns(df, viz_dir, colors)
        
        print(f"Attraction visualizations saved to {viz_dir}")

    def plot_feature_analysis(self, df, feature_type, features, title, viz_dir, colors):
        """Helper function to plot feature analysis."""
        # Calculate metrics
        feature_metrics = []
        has_sentiment = 'sentiment_score' in df.columns
        
        for feature in features:
            col = f'{feature_type}_{feature}'
            if col in df.columns:
                feature_reviews = df[df[col] == 1]
                if len(feature_reviews) > 0:
                    metrics = {
                        'feature': feature,
                        'count': len(feature_reviews)
                    }
                    if has_sentiment:
                        metrics['sentiment'] = feature_reviews['sentiment_score'].mean()
                    feature_metrics.append(metrics)
        
        if feature_metrics:
            # Sort by count
            feature_metrics.sort(key=lambda x: x['count'], reverse=True)
            
            if has_sentiment:
                # Create plot with sentiment
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                
                # Mention counts
                x = range(len(feature_metrics))
                counts = [m['count'] for m in feature_metrics]
                bars1 = ax1.bar(x, counts, color=colors[:len(feature_metrics)])
                ax1.set_title(f'{title} Mention Counts')
                ax1.set_xticks(x)
                ax1.set_xticklabels([m['feature'].replace('_', ' ').title() 
                                    for m in feature_metrics], rotation=45)
                
                # Add count labels
                for bar in bars1:
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}', ha='center', va='bottom')
                
                # Sentiment scores
                sentiments = [m['sentiment'] for m in feature_metrics]
                bars2 = ax2.bar(x, sentiments, color=colors[:len(feature_metrics)])
                ax2.set_title(f'{title} Sentiment Analysis')
                ax2.set_xticks(x)
                ax2.set_xticklabels([m['feature'].replace('_', ' ').title() 
                                    for m in feature_metrics], rotation=45)
                ax2.axhline(y=0, color='black', linestyle='-', alpha=0.2)
                
                # Add sentiment labels
                for bar in bars2:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.2f}', ha='center', va='bottom')
            else:
                # Create plot without sentiment
                fig, ax1 = plt.subplots(figsize=(12, 6))
                
                # Mention counts
                x = range(len(feature_metrics))
                counts = [m['count'] for m in feature_metrics]
                bars1 = ax1.bar(x, counts, color=colors[:len(feature_metrics)])
                ax1.set_title(f'{title} Mention Counts')
                ax1.set_xticks(x)
                ax1.set_xticklabels([m['feature'].replace('_', ' ').title() 
                                    for m in feature_metrics], rotation=45)
                
                # Add count labels
                for bar in bars1:
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f'{feature_type}_analysis.png'), dpi=300)
            plt.close()

    def plot_unique_features(self, df, viz_dir, colors):
        """Helper function to plot unique features analysis."""
        feature_metrics = []
        has_sentiment = 'sentiment_score' in df.columns
        
        for feature in self.unique_features:
            col = f'unique_{feature}'
            if col in df.columns:
                feature_reviews = df[df[col] == 1]
                if len(feature_reviews) > 0:
                    metrics = {
                        'feature': feature,
                        'count': len(feature_reviews)
                    }
                    if has_sentiment:
                        metrics['sentiment'] = feature_reviews['sentiment_score'].mean()
                    feature_metrics.append(metrics)
        
        if feature_metrics:
            # Sort by count
            feature_metrics.sort(key=lambda x: x['count'], reverse=True)
            
            if has_sentiment:
                # Create plot with sentiment
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Counts
                x = range(len(feature_metrics))
                counts = [m['count'] for m in feature_metrics]
                bars1 = ax1.bar(x, counts, color=colors[:len(feature_metrics)])
                ax1.set_title('Unique Feature Mentions')
                ax1.set_xticks(x)
                ax1.set_xticklabels([m['feature'].title() for m in feature_metrics], rotation=45)

                # Add count labels
                for bar in bars1:
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}', ha='center', va='bottom')

                # Sentiment scores
                sentiments = [m['sentiment'] for m in feature_metrics]
                bars2 = ax2.bar(x, sentiments, color=colors[:len(feature_metrics)])
                ax2.set_title('Sentiment Scores for Unique Features')
                ax2.set_xticks(x)
                ax2.set_xticklabels([m['feature'].title() for m in feature_metrics], rotation=45)
                ax2.axhline(y=0, color='black', linestyle='-', alpha=0.2)

                # Add sentiment labels
                for bar in bars2:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.2f}', ha='center', va='bottom')
            else:
                # Create plot without sentiment
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Counts
                x = range(len(feature_metrics))
                counts = [m['count'] for m in feature_metrics]
                bars = ax.bar(x, counts, color=colors[:len(feature_metrics)])
                ax.set_title('Unique Feature Mentions')
                ax.set_xticks(x)
                ax.set_xticklabels([m['feature'].title() for m in feature_metrics], rotation=45)

                # Add count labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}', ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'unique_features_analysis.png'), dpi=300)
            plt.close()
    
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
    
    def run_analysis(self, df):
        print("\nRunning attraction analysis...")
        attraction_df = self.filter_attraction_reviews(df)
        print(f"Analyzing {len(attraction_df)} attraction reviews")

        # Run analyses
        results = {
            'activity_types': self.analyze_activity_types(attraction_df),
            'experience_aspects': self.analyze_experience_aspects(attraction_df),
            'unique_features': self.analyze_unique_features(attraction_df),
            'seasonal_patterns': self.analyze_seasonal_patterns(attraction_df),
            'traveler_preferences': self.analyze_by_traveler_type(attraction_df)
        }

        # Generate visualizations
        self.generate_visualizations(attraction_df)

        # Save detailed results
        self.save_results(results)

        return results
        
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
            seasonal_stats = self.analyze_seasonal_patterns(island_df)
            traveler_stats = self.analyze_by_traveler_type(island_df)
            
            # Generate visualizations for this island
            self.generate_visualizations(island_df)
            
            # Compile results for this island
            island_results[island] = {
                'activity_types': activity_stats,
                'experience_aspects': aspect_stats,
                'unique_features': feature_stats,
                'seasonal_patterns': seasonal_stats,
                'traveler_preferences': traveler_stats,
                'review_count': len(island_df),
                'avg_rating': float(island_df['rating'].mean()) if 'rating' in island_df.columns else None
            }
            
            # Collect data for comparison
            if 'rating' in island_df.columns:
                island_stats_comparison['avg_ratings'][island] = float(island_df['rating'].mean())
            
            # Collect top activity types with sentiment data
            for activity, stats in activity_stats.items():
                if 'avg_sentiment' in stats:
                    if activity not in island_stats_comparison['activity_types']:
                        island_stats_comparison['activity_types'][activity] = {}
                    island_stats_comparison['activity_types'][activity][island] = stats['avg_sentiment']
            
            # Collect top experience aspects with sentiment data
            for aspect, stats in aspect_stats.items():
                if 'avg_sentiment' in stats:
                    if aspect not in island_stats_comparison['experience_aspects']:
                        island_stats_comparison['experience_aspects'][aspect] = {}
                    island_stats_comparison['experience_aspects'][aspect][island] = stats['avg_sentiment']
            
            # Collect unique features with sentiment data
            for feature, stats in feature_stats.items():
                if 'avg_sentiment' in stats:
                    if feature not in island_stats_comparison['unique_features']:
                        island_stats_comparison['unique_features'][feature] = {}
                    island_stats_comparison['unique_features'][feature][island] = stats['avg_sentiment']
            
            # Restore original output directory
            self.output_dir = original_output_dir
        
        # Generate cross-island comparison visualizations
        self._visualize_island_comparisons(island_stats_comparison, island_output_dir)
        
        # Save overall island results
        self._save_island_results(island_results, island_output_dir)
        
        print("\nIsland-based attraction analysis complete.")
        
        # Generate cross-island visualizations
        self.visualize_island_attraction_comparisons(island_results, island_output_dir)
        
        return island_results
    
    def _visualize_island_comparisons(self, comparison_data, output_dir):
        """
        Create visualizations comparing attraction analysis across islands.
        
        Parameters:
        - comparison_data: Dictionary with cross-island comparisons
        - output_dir: Directory to save the visualizations
        """
        # Create visualizations directory
        viz_dir = os.path.join(output_dir, 'visualizations')
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        
        # 1. Compare average ratings by island
        if comparison_data['avg_ratings']:
            plt.figure(figsize=(12, 6))
            islands = list(comparison_data['avg_ratings'].keys())
            ratings = list(comparison_data['avg_ratings'].values())
            
            # Sort islands by average rating
            sorted_data = sorted(zip(islands, ratings), key=lambda x: x[1], reverse=True)
            sorted_islands, sorted_ratings = zip(*sorted_data)
            
            # Create bar chart
            bars = plt.bar(sorted_islands, sorted_ratings, color='skyblue')
            plt.title('Average Attraction Ratings by Island', fontsize=14)
            plt.xlabel('Island')
            plt.ylabel('Average Rating (1-5)')
            plt.ylim(min(ratings) - 0.5 if min(ratings) > 0.5 else 0, 5.5)
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            
            # Add rating labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'island_attraction_ratings.png'), dpi=300)
            plt.close()
        
        # 2. Compare activity types sentiment across islands
        if comparison_data['activity_types']:
            # Convert to DataFrame for easier heatmap creation
            activity_data = {}
            
            for activity, island_values in comparison_data['activity_types'].items():
                # Only include activities with data from multiple islands and no None values
                if len(island_values) >= 2 and all(v is not None for v in island_values.values()):
                    activity_data[activity] = island_values
            
            if activity_data:
                df_activities = pd.DataFrame(activity_data).T
                
                # Check if we have any valid data for the heatmap
                if not df_activities.empty and not df_activities.isna().all().all():
                    # Convert any remaining object dtypes to float
                    df_activities = df_activities.astype(float)
                    
                    # Create heatmap
                    plt.figure(figsize=(12, 8))
                    sns.heatmap(df_activities, annot=True, cmap='RdYlGn', center=0, fmt='.2f')
                    plt.title('Activity Type Sentiment by Island', fontsize=14)
                    plt.ylabel('Activity Type')
                    plt.xlabel('Island')
                    plt.tight_layout()
                    plt.savefig(os.path.join(viz_dir, 'island_activity_types_comparison.png'), dpi=300)
                    plt.close()
        
        # 3. Compare experience aspects sentiment across islands
        if comparison_data['experience_aspects']:
            # Convert to DataFrame for easier heatmap creation
            aspect_data = {}
            
            for aspect, island_values in comparison_data['experience_aspects'].items():
                # Only include aspects with data from multiple islands and no None values
                if len(island_values) >= 2 and all(v is not None for v in island_values.values()):
                    aspect_data[aspect] = island_values
            
            if aspect_data:
                df_aspects = pd.DataFrame(aspect_data).T
                
                # Check if we have any valid data for the heatmap
                if not df_aspects.empty and not df_aspects.isna().all().all():
                    # Convert any remaining object dtypes to float
                    df_aspects = df_aspects.astype(float)
                    
                    # Create heatmap
                    plt.figure(figsize=(12, 8))
                    sns.heatmap(df_aspects, annot=True, cmap='RdYlGn', center=0, fmt='.2f')
                    plt.title('Experience Aspect Sentiment by Island', fontsize=14)
                    plt.ylabel('Experience Aspect')
                    plt.xlabel('Island')
                    plt.tight_layout()
                    plt.savefig(os.path.join(viz_dir, 'island_experience_aspects_comparison.png'), dpi=300)
                    plt.close()
        
        # 4. Compare unique features sentiment across islands
        if comparison_data['unique_features']:
            # Convert to DataFrame for easier heatmap creation
            feature_data = {}
            
            for feature, island_values in comparison_data['unique_features'].items():
                # Only include features with data from multiple islands and no None values
                if len(island_values) >= 2 and all(v is not None for v in island_values.values()):
                    feature_data[feature] = island_values
            
            if feature_data:
                df_features = pd.DataFrame(feature_data).T
                
                # Check if we have any valid data for the heatmap
                if not df_features.empty and not df_features.isna().all().all():
                    # Convert any remaining object dtypes to float
                    df_features = df_features.astype(float)
                    
                    # Create heatmap
                    plt.figure(figsize=(12, 6))
                    sns.heatmap(df_features, annot=True, cmap='RdYlGn', center=0, fmt='.2f')
                    plt.title('Unique Feature Sentiment by Island', fontsize=14)
                    plt.ylabel('Unique Feature')
                    plt.xlabel('Island')
                    plt.tight_layout()
                    plt.savefig(os.path.join(viz_dir, 'island_unique_features_comparison.png'), dpi=300)
                    plt.close()
        
        print(f"Island comparison visualizations saved to {viz_dir}")
    
    def _save_island_results(self, island_results, output_dir):
        """
        Save island-based analysis results.
        
        Parameters:
        - island_results: Dictionary with results by island
        - output_dir: Directory to save results
        """
        # Create a summary of island analysis
        summary = {}
        
        for island, results in island_results.items():
            summary[island] = {
                'review_count': results['review_count'],
                'avg_rating': results['avg_rating'],
                'top_activities': sorted(results['activity_types'].items(), 
                                      key=lambda x: x[1].get('review_count', 0), 
                                      reverse=True)[:3] if results['activity_types'] else [],
                'top_experience_aspects': sorted(results['experience_aspects'].items(), 
                                             key=lambda x: x[1].get('review_count', 0), 
                                             reverse=True)[:3] if results['experience_aspects'] else [],
                'top_unique_features': sorted(results['unique_features'].items(), 
                                           key=lambda x: x[1].get('review_count', 0), 
                                           reverse=True)[:3] if results['unique_features'] else []
            }
        
        # Save summary to JSON
        def convert_to_serializable(obj):
            """Helper function to make objects JSON serializable"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()  # Convert numpy arrays to lists
            elif isinstance(obj, np.generic):
                return obj.item()  # Convert numpy datatypes to native Python types
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict()  # Convert DataFrames to dictionaries
            elif isinstance(obj, pd.Series):
                return obj.to_dict()  # Convert Series to dictionaries
            return obj  # Fallback for all other data types
        
        with open(os.path.join(output_dir, 'island_attraction_summary.json'), 'w', encoding='utf-8') as f:
            # Use a custom function to ensure all objects are serializable
            clean_summary = json.loads(json.dumps(summary, default=convert_to_serializable))
            json.dump(clean_summary, f, indent=2)
        
        # Create a more detailed Excel report
        try:
            import openpyxl
            
            # Create Excel writer
            excel_path = os.path.join(output_dir, 'island_attraction_comparison.xlsx')
            writer = pd.ExcelWriter(excel_path, engine='openpyxl')
            
            # Sheet 1: Overview
            overview_data = []
            for island, results in island_results.items():
                row = {
                    'Island': island,
                    'Reviews': results['review_count'],
                    'Avg Rating': results['avg_rating'] if results['avg_rating'] else 'N/A'
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
        
        except ImportError:
            print("openpyxl not found. Excel report not generated.")
            
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
    
    def visualize_island_attraction_comparisons(self, island_results, output_dir):
        """
        Create side-by-side visualizations comparing attraction features across islands.
        
        Parameters:
        - island_results: Dictionary with results by island from run_island_analysis
        - output_dir: Directory to save the comparison visualizations
        """
        print("\nGenerating cross-island attraction comparison visualizations...")
        
        # Create visualizations directory
        viz_dir = os.path.join(output_dir, 'island_comparisons')
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        
        # Get islands with enough data
        valid_islands = [island for island, data in island_results.items() 
                        if data['review_count'] >= 10]
        
        if len(valid_islands) < 2:
            print("Not enough islands with sufficient data for comparison")
            return
        
        # Set visualization style
        plt.style.use('seaborn-v0_8-colorblind')
        
        # 1. Activity Types Comparison
        activity_types = {}
        
        # Collect data for all islands
        for island in valid_islands:
            activity_stats = island_results[island]['activity_types']
            for activity, stats in activity_stats.items():
                if 'review_count' in stats:
                    if activity not in activity_types:
                        activity_types[activity] = {}
                    # Calculate percentage of reviews mentioning this activity
                    activity_types[activity][island] = (stats['review_count'] / 
                                                    island_results[island]['review_count'] * 100)
        
        # Create visualization if we have data
        if activity_types:
            # Convert to DataFrame
            df_activity_types = pd.DataFrame(activity_types)
            
            # Sort columns by overall mention frequency
            total_mentions = df_activity_types.sum()
            sorted_types = total_mentions.sort_values(ascending=False).index.tolist()
            
            # Limit to top 7 types if there are too many
            if len(sorted_types) > 7:
                sorted_types = sorted_types[:7]
            
            # Select and sort data
            plot_data = df_activity_types[sorted_types].copy()
            
            # Create plot
            plt.figure(figsize=(14, 8))
            plot_data.plot(kind='bar', figsize=(14, 8))
            plt.title('Activity Types by Island (% of Reviews)', fontsize=16)
            plt.xlabel('Island', fontsize=14)
            plt.ylabel('% of Reviews Mentioning', fontsize=14)
            plt.xticks(rotation=45, ha='right', fontsize=12)
            plt.legend(title='Activity Type', fontsize=12, title_fontsize=14)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'activity_types_by_island.png'), dpi=300)
            plt.close()
        
        # 2. Experience Aspects Comparison
        experience_aspects = {}
        
        # Collect data for all islands
        for island in valid_islands:
            aspect_stats = island_results[island]['experience_aspects']
            for aspect, stats in aspect_stats.items():
                if 'review_count' in stats:
                    if aspect not in experience_aspects:
                        experience_aspects[aspect] = {}
                    # Calculate percentage of reviews mentioning this aspect
                    experience_aspects[aspect][island] = (stats['review_count'] / 
                                                       island_results[island]['review_count'] * 100)
        
        # Create visualization if we have data
        if experience_aspects:
            # Convert to DataFrame
            df_aspects = pd.DataFrame(experience_aspects)
            
            # Sort columns by overall mention frequency
            total_mentions = df_aspects.sum()
            sorted_aspects = total_mentions.sort_values(ascending=False).index.tolist()
            
            # Limit to top 7 aspects if there are too many
            if len(sorted_aspects) > 7:
                sorted_aspects = sorted_aspects[:7]
            
            # Select and sort data
            plot_data = df_aspects[sorted_aspects].copy()
            
            # Create plot
            plt.figure(figsize=(14, 8))
            plot_data.plot(kind='bar', figsize=(14, 8))
            plt.title('Experience Aspects by Island (% of Reviews)', fontsize=16)
            plt.xlabel('Island', fontsize=14)
            plt.ylabel('% of Reviews Mentioning', fontsize=14)
            plt.xticks(rotation=45, ha='right', fontsize=12)
            plt.legend(title='Experience Aspect', fontsize=12, title_fontsize=14)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'experience_aspects_by_island.png'), dpi=300)
            plt.close()
        
        # 3. Unique Features Comparison
        unique_features = {}
        
        # Collect data for all islands
        for island in valid_islands:
            feature_stats = island_results[island]['unique_features']
            for feature, stats in feature_stats.items():
                if 'review_count' in stats:
                    if feature not in unique_features:
                        unique_features[feature] = {}
                    # Calculate percentage of reviews mentioning this feature
                    unique_features[feature][island] = (stats['review_count'] / 
                                                     island_results[island]['review_count'] * 100)
        
        # Create visualization if we have data
        if unique_features:
            # Convert to DataFrame
            df_features = pd.DataFrame(unique_features)
            
            # Sort columns by overall mention frequency
            total_mentions = df_features.sum()
            sorted_features = total_mentions.sort_values(ascending=False).index.tolist()
            
            # Limit to top 7 features if there are too many
            if len(sorted_features) > 7:
                sorted_features = sorted_features[:7]
            
            # Select and sort data
            plot_data = df_features[sorted_features].copy()
            
            # Create plot
            plt.figure(figsize=(14, 8))
            plot_data.plot(kind='bar', figsize=(14, 8))
            plt.title('Unique Features by Island (% of Reviews)', fontsize=16)
            plt.xlabel('Island', fontsize=14)
            plt.ylabel('% of Reviews Mentioning', fontsize=14)
            plt.xticks(rotation=45, ha='right', fontsize=12)
            plt.legend(title='Unique Feature', fontsize=12, title_fontsize=14)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'unique_features_by_island.png'), dpi=300)
            plt.close()
        
        # 4. Average Ratings Comparison
        if all('avg_rating' in island_results[island] and 
              island_results[island]['avg_rating'] is not None 
              for island in valid_islands):
            
            # Collect ratings for each island
            ratings = {island: island_results[island]['avg_rating'] for island in valid_islands}
            
            # Create ratings DataFrame
            df_ratings = pd.DataFrame(list(ratings.items()), columns=['Island', 'Rating'])
            df_ratings = df_ratings.sort_values('Rating', ascending=False)
            
            # Create bar chart
            plt.figure(figsize=(12, 6))
            bars = plt.bar(df_ratings['Island'], df_ratings['Rating'], color='skyblue')
            plt.title('Average Attraction Ratings by Island', fontsize=16)
            plt.xlabel('Island', fontsize=14)
            plt.ylabel('Average Rating (1-5)', fontsize=14)
            plt.ylim(min(df_ratings['Rating']) - 0.5 if min(df_ratings['Rating']) > 0.5 else 0, 5.5)
            plt.xticks(rotation=45, ha='right', fontsize=12)
            plt.grid(axis='y', alpha=0.3)
            
            # Add rating labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'average_ratings_by_island.png'), dpi=300)
            plt.close()
        
        print(f"Island comparison visualizations saved to {viz_dir}")


# EnhancedAttractionAnalyzer class implementation would follow here (unchanged)