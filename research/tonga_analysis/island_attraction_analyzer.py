import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from island_review_count import IslandBasedAnalyzer
from sentiment_analyzer import SentimentAnalyzer
import re
from wordcloud import WordCloud
from tonga_analysis.visualization_styles import (
    REGION_COLORS, ISLAND_COLORS, set_visualization_style, 
    get_island_palette
)

class IslandAttractionAnalyzer:
    """
    Analyzer for attractions across different islands in Tonga.
    This is a simplified version focused specifically on island-based analysis
    to fix the missing functionality in the attractions_analyzer.py file.
    """
    
    def __init__(self, sentiment_analyzer=None, output_dir=None):
        """
        Initialize the island attraction analyzer.
        
        Parameters:
        - sentiment_analyzer: SentimentAnalyzer instance for text analysis
        - output_dir: Directory to save outputs
        """
        self.sentiment_analyzer = sentiment_analyzer
        
        # Use standardized output directory by default
        if output_dir is None:
            # Use the parent directory's outputs folder
            parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            self.output_dir = os.path.join(parent_dir, "outputs", "attraction_analysis")
        else:
            self.output_dir = output_dir
            
        # Create output directory
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")
        
        # Set up island analysis directory
        self.island_dir = os.path.join(self.output_dir, 'island_analysis')
        if not os.path.exists(self.island_dir):
            os.makedirs(self.island_dir)
        
        # Create visualization directory
        self.viz_dir = os.path.join(self.output_dir, 'visualizations')
        if not os.path.exists(self.viz_dir):
            os.makedirs(self.viz_dir)
        
        # Define analysis categories
        self.activity_types = {
            'water_activities': ['swim', 'snorkel', 'dive', 'kayak', 'boat', 'cruise', 'sail', 'surf', 'fishing'],
            'land_activities': ['hike', 'walk', 'trek', 'climb', 'tour', 'drive', 'ride', 'bike', 'cycling'],
            'cultural_activities': ['museum', 'history', 'tradition', 'culture', 'ceremony', 'dance', 'music', 'craft'],
            'nature_viewing': ['wildlife', 'bird', 'whale', 'dolphin', 'turtle', 'fish', 'coral', 'reef', 'forest'],
            'relaxation': ['beach', 'relax', 'lounge', 'sunset', 'view', 'scenic', 'photo', 'sightseeing']
        }
        
        self.experience_aspects = {
            'guide_quality': ['guide', 'tour guide', 'leader', 'instructor', 'captain', 'staff'],
            'safety': ['safe', 'safety', 'equipment', 'secure', 'comfort', 'comfortable'],
            'value': ['price', 'value', 'worth', 'money', 'expensive', 'cheap', 'cost'],
            'scenery': ['beautiful', 'view', 'scenery', 'landscape', 'picture', 'photo'],
            'learning': ['learn', 'educational', 'interesting', 'information', 'history', 'knowledge']
        }
    
    def filter_attraction_reviews(self, df):
        """
        Filter to get only attraction reviews.
        
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
        
        # Analyze mentions of each activity type
        activity_stats = {}
        
        for activity, keywords in self.activity_types.items():
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
        
        # Analyze mentions of each aspect
        aspect_stats = {}
        
        for aspect, keywords in self.experience_aspects.items():
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
            
        # Check for possible business name column names
        business_col = None
        for col in ['business_name', 'name', 'business', 'place_name']:
            if col in df.columns:
                business_col = col
                break
        
        # If no valid column is found, use a generic grouper
        if business_col is None:
            print("Warning: No business name column found. Using generic attraction grouping.")
            # Create a dummy column for grouping
            df = df.copy()
            df['attraction_id'] = range(len(df))
            grouped = df.groupby('attraction_id')
        else:
            grouped = df.groupby(business_col)
            
        # Analyze each unique attraction
        feature_stats = {}
        
        for name, group in grouped:
            # Skip if not enough reviews
            if len(group) < 5:
                continue
                
            # Use business name for the key
            business_name = name
            # Try to use a better name if available
            for col in ['business_name', 'name', 'business', 'place_name']:
                if col in group.columns:
                    business_name = str(group[col].iloc[0])
                    break
            
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
            
        text = ' '.join(df['text'].fillna('').str.lower())
        counts = {}
        
        for activity, keywords in self.activity_types.items():
            pattern = '|'.join(keywords)
            counts[activity] = len(re.findall(pattern, text))
            
        # Return top 2 activity types
        sorted_activities = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return [a for a, c in sorted_activities[:2] if c > 0]
    
    def generate_visualizations(self, df, island=None):
        """
        Generate visualizations for attraction analysis.
        
        Parameters:
        - df: DataFrame with attraction reviews
        - island: Optional island name for island-specific visualizations
        
        Returns:
        - Dictionary with paths to generated visualizations
        """
        if len(df) == 0:
            print("No data available for visualizations")
            return {}
        
        # Set visualization style
        set_visualization_style()
        
        # Use island-specific output directory if provided
        if island:
            viz_dir = os.path.join(self.island_dir, island.replace(' ', '_').lower(), 'visualizations')
        else:
            viz_dir = self.viz_dir
            
        # Create directory if it doesn't exist
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        
        visualization_paths = {}
        
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
            plt.title(f'Popular Activities Mentioned in Reviews{" on " + island if island else ""}')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            activity_path = os.path.join(viz_dir, 'activity_analysis.png')
            plt.savefig(activity_path, dpi=300)
            plt.close()
            visualization_paths['activity_analysis'] = activity_path
        
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
            plt.title(f'Experience Aspects Mentioned in Reviews{" on " + island if island else ""}')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            aspect_path = os.path.join(viz_dir, 'aspect_analysis.png')
            plt.savefig(aspect_path, dpi=300)
            plt.close()
            visualization_paths['aspect_analysis'] = aspect_path
            
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
            plt.title(f'Top Attractions by Review Count{" on " + island if island else ""}')
            plt.grid(axis='x', alpha=0.3)
            plt.gca().invert_yaxis()  # To show highest count at top
            plt.tight_layout()
            
            feature_path = os.path.join(viz_dir, 'unique_features_analysis.png')
            plt.savefig(feature_path, dpi=300)
            plt.close()
            visualization_paths['unique_features_analysis'] = feature_path
            
        return visualization_paths
    
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
            return None
        
        print(f"Islands with sufficient attraction reviews: {', '.join(valid_islands)}")
        
        # Create a directory for island-specific outputs if it doesn't exist
        if not os.path.exists(self.island_dir):
            os.makedirs(self.island_dir)
            
        # Store results for each island
        island_results = {}
        
        # Run analysis for each island
        for island in valid_islands:
            print(f"\n--- Analyzing attractions on {island} ---")
            
            # Create island-specific output directory
            island_dir = os.path.join(self.island_dir, island.replace(' ', '_').lower())
            if not os.path.exists(island_dir):
                os.makedirs(island_dir)
            
            # Filter to this island's attraction reviews
            island_df = all_attraction_df[all_attraction_df[island_col] == island].copy()
            
            if len(island_df) < 10:
                print(f"Not enough reviews for {island}")
                continue
                
            print(f"Analyzing {len(island_df)} attraction reviews for {island}")
            
            # Run analyses for this island
            activity_stats = self.analyze_activity_types(island_df)
            aspect_stats = self.analyze_experience_aspects(island_df)
            feature_stats = self.analyze_unique_features(island_df)
            
            # Generate island-specific visualizations
            viz_paths = self.generate_visualizations(island_df, island)
            
            # Store results for this island
            island_results[island] = {
                'review_count': len(island_df),
                'avg_rating': island_df['rating'].mean() if 'rating' in island_df.columns else None,
                'activity_types': activity_stats,
                'experience_aspects': aspect_stats,
                'unique_features': feature_stats,
                'visualizations': viz_paths
            }
        
        # Save combined results
        if island_results:
            results_file = os.path.join(self.island_dir, 'island_attraction_summary.json')
            
            # Convert to serializable format
            clean_results = self._clean_for_json(island_results)
            
            with open(results_file, 'w') as f:
                json.dump(clean_results, f, indent=2)
                
            print(f"Island analysis summary saved to {results_file}")
            
            # Create cross-island comparison Excel file
            self._save_island_comparison(island_results)
            
            # Generate cross-island visualizations
            self.generate_island_comparisons(island_results)
            
        return island_results
    
    def _clean_for_json(self, data):
        """Clean data for JSON serialization."""
        if isinstance(data, dict):
            return {k: self._clean_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._clean_for_json(i) for i in data]
        elif isinstance(data, (np.int64, np.int32)):
            return int(data)
        elif isinstance(data, (np.float64, np.float32)):
            return float(data) if not np.isnan(data) else None
        elif isinstance(data, (pd.DataFrame, pd.Series)):
            return data.to_dict()
        elif isinstance(data, np.ndarray):
            return data.tolist()
        else:
            return data
    
    def _save_island_comparison(self, island_results):
        """Save island comparison data to Excel."""
        try:
            # Create Excel file
            excel_path = os.path.join(self.island_dir, 'island_attraction_comparison.xlsx')
            writer = pd.ExcelWriter(excel_path, engine='xlsxwriter')
            
            # Overview sheet
            overview_data = []
            for island, data in island_results.items():
                row = {
                    'Island': island,
                    'Review Count': data['review_count'],
                    'Average Rating': round(data['avg_rating'], 2) if data['avg_rating'] and not np.isnan(data['avg_rating']) else 'N/A',
                    'Top Activities': ', '.join(sorted(data['activity_types'].keys(), 
                                              key=lambda x: data['activity_types'][x]['review_count'], 
                                              reverse=True)[:3]) if data['activity_types'] else 'N/A',
                    'Top Experience Aspects': ', '.join(sorted(data['experience_aspects'].keys(), 
                                                      key=lambda x: data['experience_aspects'][x]['review_count'], 
                                                      reverse=True)[:3]) if data['experience_aspects'] else 'N/A'
                }
                overview_data.append(row)
            
            if overview_data:
                overview_df = pd.DataFrame(overview_data)
                overview_df.to_excel(writer, sheet_name='Overview', index=False)
            
            # Activity types sheet
            activity_data = []
            for island, data in island_results.items():
                for activity, stats in data['activity_types'].items():
                    row = {
                        'Island': island,
                        'Activity Type': activity.replace('_', ' ').title(),
                        'Mentions': stats['review_count'],
                        'Percentage': round(stats['percentage'], 1),
                        'Average Sentiment': round(stats['avg_sentiment'], 2) if stats['avg_sentiment'] and not np.isnan(stats['avg_sentiment']) else 'N/A',
                        'Average Rating': round(stats['avg_rating'], 2) if stats['avg_rating'] and not np.isnan(stats['avg_rating']) else 'N/A',
                        'Top Keywords': ', '.join(stats['top_keywords'])
                    }
                    activity_data.append(row)
            
            if activity_data:
                activity_df = pd.DataFrame(activity_data)
                activity_df.to_excel(writer, sheet_name='Activity Types', index=False)
            
            # Experience aspects sheet
            aspect_data = []
            for island, data in island_results.items():
                for aspect, stats in data['experience_aspects'].items():
                    row = {
                        'Island': island,
                        'Experience Aspect': aspect.replace('_', ' ').title(),
                        'Mentions': stats['review_count'],
                        'Percentage': round(stats['percentage'], 1),
                        'Average Sentiment': round(stats['avg_sentiment'], 2) if stats['avg_sentiment'] and not np.isnan(stats['avg_sentiment']) else 'N/A',
                        'Average Rating': round(stats['avg_rating'], 2) if stats['avg_rating'] and not np.isnan(stats['avg_rating']) else 'N/A',
                        'Top Keywords': ', '.join(stats['top_keywords'])
                    }
                    aspect_data.append(row)
            
            if aspect_data:
                aspect_df = pd.DataFrame(aspect_data)
                aspect_df.to_excel(writer, sheet_name='Experience Aspects', index=False)
            
            # Save and close
            writer.close()
            print(f"Island comparison Excel report saved to {excel_path}")
            return excel_path
            
        except Exception as e:
            print(f"Error creating Excel report: {e}")
            return None
    
    def generate_island_comparisons(self, island_results):
        """Generate cross-island comparison visualizations."""
        print("\nGenerating cross-island attraction comparison visualizations...")
        
        # Create visualizations directory
        viz_dir = os.path.join(self.island_dir, 'visualizations')
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        
        # Set visualization style
        set_visualization_style()
        
        # Extract valid islands
        valid_islands = list(island_results.keys())
        
        if len(valid_islands) < 2:
            print("Not enough islands for comparison visualizations")
            return
        
        # Get island colors
        island_colors = {island: ISLAND_COLORS.get(island.lower(), '#AAAAAA') for island in valid_islands}
        
        # 1. Activity type comparison across islands
        activity_data = {}
        
        # Collect data for all islands and activities
        for island, data in island_results.items():
            for activity, stats in data['activity_types'].items():
                if activity not in activity_data:
                    activity_data[activity] = {}
                activity_data[activity][island] = stats['review_count']
        
        if activity_data:
            # Create DataFrame for easier plotting
            activities = sorted(activity_data.keys())
            islands = sorted(valid_islands)
            
            # Fill in missing values with 0
            activity_df = pd.DataFrame({
                activity: [activity_data[activity].get(island, 0) for island in islands]
                for activity in activities
            }, index=islands)
            
            # Stacked bar chart
            plt.figure(figsize=(12, 8))
            ax = activity_df.plot(kind='bar', stacked=True, figsize=(12, 6), 
                               color=sns.color_palette('tab10', len(activities)))
            plt.title('Activity Types by Island', fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Island', fontsize=14)
            plt.ylabel('Number of Mentions', fontsize=14)
            plt.legend(title='Activity Type', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'activity_types_by_island.png'), dpi=300)
            plt.close()
            
            # Proportional stacked bar chart
            plt.figure(figsize=(12, 8))
            prop_df = activity_df.div(activity_df.sum(axis=1), axis=0) * 100
            prop_df.plot(kind='bar', stacked=True, figsize=(12, 6), 
                      color=sns.color_palette('tab10', len(activities)))
            plt.title('Activity Types Proportion by Island', fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Island', fontsize=14)
            plt.ylabel('Percentage of Mentions', fontsize=14)
            plt.legend(title='Activity Type', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'activity_types_proportion_by_island.png'), dpi=300)
            plt.close()
        
        # 2. Experience aspects comparison across islands
        aspect_data = {}
        
        # Collect data for all islands and aspects
        for island, data in island_results.items():
            for aspect, stats in data['experience_aspects'].items():
                if aspect not in aspect_data:
                    aspect_data[aspect] = {}
                aspect_data[aspect][island] = stats['review_count']
        
        if aspect_data:
            # Create DataFrame for easier plotting
            aspects = sorted(aspect_data.keys())
            
            # Fill in missing values with 0
            aspect_df = pd.DataFrame({
                aspect: [aspect_data[aspect].get(island, 0) for island in islands]
                for aspect in aspects
            }, index=islands)
            
            # Stacked bar chart
            plt.figure(figsize=(12, 8))
            ax = aspect_df.plot(kind='bar', stacked=True, figsize=(12, 6), 
                             color=sns.color_palette('Set2', len(aspects)))
            plt.title('Experience Aspects by Island', fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Island', fontsize=14)
            plt.ylabel('Number of Mentions', fontsize=14)
            plt.legend(title='Experience Aspect', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'experience_aspects_by_island.png'), dpi=300)
            plt.close()
            
            # Proportional stacked bar chart
            plt.figure(figsize=(12, 8))
            prop_df = aspect_df.div(aspect_df.sum(axis=1), axis=0) * 100
            prop_df.plot(kind='bar', stacked=True, figsize=(12, 6), 
                      color=sns.color_palette('Set2', len(aspects)))
            plt.title('Experience Aspects Proportion by Island', fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Island', fontsize=14)
            plt.ylabel('Percentage of Mentions', fontsize=14)
            plt.legend(title='Experience Aspect', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'experience_aspects_proportion_by_island.png'), dpi=300)
            plt.close()
        
        # 3. Rating by island and activity type
        rating_data = []
        
        # Collect rating data by island and activity
        for island, data in island_results.items():
            for activity, stats in data['activity_types'].items():
                if stats['avg_rating'] and not np.isnan(stats['avg_rating']):
                    rating_data.append({
                        'Island': island,
                        'Activity Type': activity.replace('_', ' ').title(),
                        'Average Rating': stats['avg_rating'],
                        'Reviews': stats['review_count']
                    })
        
        if rating_data:
            rating_df = pd.DataFrame(rating_data)
            
            # Bar chart by island and activity
            plt.figure(figsize=(14, 8))
            ax = sns.barplot(x='Island', y='Average Rating', hue='Activity Type', data=rating_df)
            plt.title('Average Rating by Island and Activity Type', fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Island', fontsize=14)
            plt.ylabel('Average Rating', fontsize=14)
            plt.legend(title='Activity Type', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'rating_by_island_and_type_bars.png'), dpi=300)
            plt.close()
            
            # Heatmap
            plt.figure(figsize=(12, 8))
            rating_pivot = rating_df.pivot_table(index='Activity Type', columns='Island', 
                                            values='Average Rating', aggfunc='mean')
            sns.heatmap(rating_pivot, annot=True, cmap='RdYlGn', center=3.5, vmin=1, vmax=5,
                     linewidths=0.5, fmt='.2f')
            plt.title('Average Rating by Island and Activity Type', fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'rating_by_island_and_type_heatmap.png'), dpi=300)
            plt.close()
        
        print(f"Cross-island comparison visualizations saved to {viz_dir}")