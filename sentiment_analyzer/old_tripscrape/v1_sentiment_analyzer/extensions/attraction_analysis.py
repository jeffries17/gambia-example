import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import re
import json

class AttractionAnalysis:
    """
    Extension class for analyzing attraction-specific reviews in Tonga tourism data.
    Focuses on tours, attractions, and recreational activities.
    """
    
    def __init__(self, base_analyzer, output_dir='attraction_insights'):
        """
        Initialize with reference to the base analyzer.
        
        Parameters:
        - base_analyzer: Base TongaTourismAnalysis instance
        - output_dir: Directory to save attraction-specific outputs
        """
        self.analyzer = base_analyzer
        
        # Create full path if output_dir is relative
        if not os.path.isabs(output_dir):
            self.output_dir = os.path.join(base_analyzer.output_dir, output_dir)
        else:
            self.output_dir = output_dir
            
        # Create output directory
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created attraction analysis directory: {self.output_dir}")
            
        # Attraction-specific categories setup
        self.setup_attraction_categories()
        
    def setup_attraction_categories(self):
        """
        Set up attraction-specific categories for analysis.
        """
        # Define attraction categories
        self.attraction_categories = {
            'marine': ['whale', 'swim', 'snorkel', 'dive', 'scuba', 'reef', 'coral', 'fish', 'ocean', 
                     'sea', 'underwater', 'marine', 'turtle', 'dolphin', 'shark', 'kayak', 'paddle'],
            'land_tours': ['hike', 'walk', 'trek', 'trail', 'volcano', 'mountain', 'forest', 'jungle', 
                          'cave', 'nature', 'tour', 'guide', 'excursion', 'sightseeing', 'journey'],
            'cultural': ['culture', 'tradition', 'history', 'museum', 'art', 'craft', 'dance', 'music', 
                       'ceremony', 'festival', 'local', 'village', 'performance', 'heritage', 'native'],
            'adventure': ['adventure', 'extreme', 'zip', 'line', 'horseback', 'horse', 'atv', 'quad', 
                        'bike', 'jeep', 'buggy', 'climb', 'rappel', 'safari', 'helicopter', 'skydive'],
            'relaxation': ['beach', 'relax', 'lounge', 'sun', 'sunbathe', 'spa', 'massage', 'yoga', 
                         'meditation', 'peaceful', 'tranquil', 'chill', 'hammock', 'sunset', 'sunrise'],
            'boat_tours': ['boat', 'cruise', 'sail', 'yacht', 'catamaran', 'ferry', 'fishing', 'charter', 
                         'island', 'hopping', 'trip', 'excursion', 'scenic', 'sunset', 'cruise'],
            'shopping': ['shop', 'market', 'souvenir', 'handicraft', 'craft', 'vendor', 'store', 
                       'mall', 'buy', 'purchase', 'artisan', 'gift', 'boutique', 'local', 'product']
        }
        
        # attraction experience aspects
        self.experience_aspects = {
            'guide_quality': ['guide', 'instructor', 'leader', 'host', 'staff', 'knowledgeable', 
                            'informative', 'friendly', 'professional', 'expert', 'experienced'],
            'safety': ['safety', 'safe', 'secure', 'danger', 'dangerous', 'equipment', 'life jacket', 
                      'precaution', 'warning', 'emergency', 'protocol', 'risk', 'protect'],
            'educational_value': ['learn', 'educational', 'informative', 'knowledge', 'fact', 
                               'interesting', 'insight', 'history', 'information', 'understand'],
            'authenticity': ['authentic', 'genuine', 'real', 'true', 'actual', 'truth', 'unique', 
                           'original', 'traditional', 'touristy', 'commercial', 'staged'],
            'group_size': ['group', 'crowd', 'crowded', 'small', 'large', 'intimate', 'private', 
                         'exclusive', 'people', 'tourist', 'other', 'size', 'number', 'many'],
            'time_management': ['time', 'duration', 'long', 'short', 'enough', 'rush', 'wait', 
                              'delay', 'punctual', 'on time', 'schedule', 'efficient', 'waste'],
            'scenery_and_wildlife': ['scenery', 'view', 'landscape', 'photo', 'beautiful', 
                                  'stunning', 'gorgeous', 'breathtaking', 'wildlife', 'animal', 
                                  'nature', 'coral', 'fish', 'whale', 'bird']
        }
    
    def verify_attraction_data(self, df):
        """
        Verify that we're dealing with attraction reviews by checking the URL path.
        TripAdvisor uses Attraction_Review in the URL for attractions.
        """
        # Check webUrl path for Attraction_Review
        if 'placeInfo.webUrl' in df.columns:
            has_attraction_url = df['placeInfo.webUrl'].str.contains('Attraction_Review', case=False, na=False)
            
            if has_attraction_url.any():
                # If any reviews have Attraction_Review in URL, verify how many
                attraction_count = has_attraction_url.sum()
                total_count = len(df)
                
                if attraction_count < total_count:
                    print(f"Warning: Only {attraction_count} out of {total_count} reviews are verified as attractions")
                else:
                    print(f"Verified all {total_count} reviews are for attractions")
                return True
                
        # If URL check fails, assume it's attraction data since we're in the attraction analyzer
        print("Warning: Could not verify attraction reviews through URL pattern")
        return True
    
    def analyze_attraction_types(self, df):
        """
        Analyze different types of activities mentioned in reviews.
        
        Parameters:
        - df: DataFrame with attraction reviews
        
        Returns:
        - DataFrame with attraction type analysis
        """
        print("Analyzing attraction types mentioned in reviews...")
        
        # For each attraction category, check if any keywords are present
        for category, keywords in self.attraction_categories.items():
            pattern = r'\b(' + '|'.join(keywords) + r')\b'
            df[f'attraction_{category}'] = df['processed_text'].astype(str).str.contains(
                pattern, case=False, regex=True).astype(int)
        
        # Calculate attraction category mention counts
        attraction_cols = [f'attraction_{category}' for category in self.attraction_categories.keys()]
        category_counts = df[attraction_cols].sum().sort_values(ascending=False)
        
        print("attraction category mentions:")
        for category, count in category_counts.items():
            category_name = category.replace('attraction_', '')
            print(f"  {category_name}: {count} mentions")
        
        # Create a bar chart of attraction category mentions
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x=category_counts.index, y=category_counts.values, palette='viridis')
        plt.title('attraction Types Mentioned in Reviews')
        plt.xlabel('attraction Type')
        plt.ylabel('Number of Mentions')
        plt.xticks(rotation=45, ha='right')
        
        # Add count labels
        for i, count in enumerate(category_counts.values):
            ax.text(i, count + 0.5, str(count), ha='center')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/attraction_type_mentions.png')
        print(f"Saved attraction type chart to {self.output_dir}/attraction_type_mentions.png")
        
        return df
    
    def analyze_experience_aspects(self, df):
        """
        Analyze different aspects of the attraction experience.
        
        Parameters:
        - df: DataFrame with attraction reviews
        
        Returns:
        - DataFrame with experience aspect analysis
        """
        print("Analyzing attraction experience aspects...")
        
        # For each experience aspect, check if any keywords are present
        for aspect, keywords in self.experience_aspects.items():
            pattern = r'\b(' + '|'.join(keywords) + r')\b'
            df[f'aspect_{aspect}'] = df['processed_text'].astype(str).str.contains(
                pattern, case=False, regex=True).astype(int)
        
        # Calculate aspect mention counts
        aspect_cols = [f'aspect_{aspect}' for aspect in self.experience_aspects.keys()]
        aspect_counts = df[aspect_cols].sum().sort_values(ascending=False)
        
        print("Experience aspect mentions:")
        for aspect, count in aspect_counts.items():
            aspect_name = aspect.replace('aspect_', '')
            print(f"  {aspect_name}: {count} mentions")
        
        # Calculate average sentiment for each aspect
        aspect_sentiment = {}
        for aspect in self.experience_aspects.keys():
            col = f'aspect_{aspect}'
            # Only consider reviews that mention this aspect
            aspect_df = df[df[col] == 1]
            if len(aspect_df) > 0:
                avg_sentiment = aspect_df['sentiment_score'].mean()
                aspect_sentiment[aspect] = avg_sentiment
        
        if aspect_sentiment:
            # Create a bar chart of aspect sentiment
            plt.figure(figsize=(12, 6))
            aspects = list(aspect_sentiment.keys())
            sentiment_values = list(aspect_sentiment.values())
            colors = ['green' if v > 0 else 'red' for v in sentiment_values]
            
            bars = plt.bar(aspects, sentiment_values, color=colors)
            plt.title('Average Sentiment for attraction Experience Aspects')
            plt.xlabel('Experience Aspect')
            plt.ylabel('Average Sentiment Score (-1 to 1)')
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            
            # Add sentiment value labels
            for bar, value in zip(bars, sentiment_values):
                label_position = value + 0.02 if value > 0 else value - 0.08
                plt.text(bar.get_x() + bar.get_width()/2., label_position,
                       f'{value:.2f}', ha='center', va='bottom', color='black', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/experience_aspect_sentiment.png')
            print(f"Saved experience aspect sentiment chart to {self.output_dir}/experience_aspect_sentiment.png")
        
        return df
    
    def analyze_by_visitor_segment(self, df):
        """
        Analyze attraction preferences by visitor segment (trip type).
        
        Parameters:
        - df: DataFrame with attraction reviews
        
        Returns:
        - DataFrame with visitor segment analysis
        """
        if 'trip_type_standard' not in df.columns:
            print("Trip type information not available for visitor segment analysis")
            return df
        
        print("Analyzing attraction preferences by visitor segment...")
        
        # Get valid trip types with enough data
        trip_type_counts = df['trip_type_standard'].value_counts()
        valid_trip_types = trip_type_counts[trip_type_counts >= 3].index.tolist()
        
        if not valid_trip_types:
            print("Not enough data for visitor segment analysis")
            return df
            
        # Filter to segments with enough data
        segment_df = df[df['trip_type_standard'].isin(valid_trip_types)]
        
        # attraction preference by segment
        attraction_cols = [f'attraction_{category}' for category in self.attraction_categories.keys()]
        
        # Calculate the percentage of each segment that mentions each attraction type
        segment_preferences = segment_df.groupby('trip_type_standard')[attraction_cols].mean()
        
        # Clean up column names for display
        segment_preferences.columns = [col.replace('attraction_', '') for col in segment_preferences.columns]
        
        print("attraction preferences by visitor segment:")
        print(segment_preferences)
        
        # Create a heatmap of attraction preferences by segment
        plt.figure(figsize=(14, 8))
        sns.heatmap(segment_preferences, annot=True, cmap='YlGnBu', fmt='.2f')
        plt.title('attraction Preferences by Visitor Segment')
        plt.ylabel('Visitor Segment')
        plt.xlabel('attraction Type')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/segment_attraction_preferences.png')
        print(f"Saved segment preferences chart to {self.output_dir}/segment_attraction_preferences.png")
        
        # Get top attraction for each segment
        segment_top_activities = {}
        for segment in segment_preferences.index:
            segment_data = segment_preferences.loc[segment]
            top_attraction = segment_data.idxmax()
            top_value = segment_data.max()
            if top_value > 0:
                segment_top_activities[segment] = (top_attraction, top_value)
                
        print("\nTop attraction preference by segment:")
        for segment, (attraction, value) in segment_top_activities.items():
            print(f"  {segment}: {attraction} ({value:.2f})")
        
        # Save to file
        segment_preferences.to_csv(f'{self.output_dir}/segment_attraction_preferences.csv')
        
        return df
    
    def analyze_seasonal_attraction_patterns(self, df):
        """
        Analyze seasonal patterns in attraction reviews.
        
        Parameters:
        - df: DataFrame with attraction reviews
        
        Returns:
        - DataFrame with seasonal analysis
        """
        if 'publishedDate' not in df.columns:
            print("Published date information not available for seasonal analysis")
            return df
        
        print("Analyzing seasonal patterns in attraction reviews...")
        
        try:
            # Convert date to datetime
            df['publishedDate'] = pd.to_datetime(df['publishedDate'], errors='coerce')
            
            # Extract month
            df['month'] = df['publishedDate'].dt.month
            df['month_name'] = df['publishedDate'].dt.strftime('%b')
            
            # Group data by month
            monthly_counts = df.groupby('month').size()
            
            # Define seasons (for Southern Hemisphere)
            def get_season(month):
                if 3 <= month <= 5:
                    return 'Autumn'
                elif 6 <= month <= 8:
                    return 'Winter'
                elif 9 <= month <= 11:
                    return 'Spring'
                else:
                    return 'Summer'
            
            df['season'] = df['month'].apply(get_season)
            
            # attraction types by season
            attraction_cols = [f'attraction_{category}' for category in self.attraction_categories.keys()]
            seasonal_attraction = df.groupby('season')[attraction_cols].mean()
            
            # Clean column names
            seasonal_attraction.columns = [col.replace('attraction_', '') for col in seasonal_attraction.columns]
            
            print("attraction type prevalence by season:")
            print(seasonal_attraction)
            
            # Create heatmap of seasonal attraction patterns
            plt.figure(figsize=(14, 8))
            sns.heatmap(seasonal_attraction, annot=True, cmap='YlGnBu', fmt='.2f')
            plt.title('attraction Type Prevalence by Season')
            plt.ylabel('Season')
            plt.xlabel('attraction Type')
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/seasonal_attraction_patterns.png')
            print(f"Saved seasonal attraction chart to {self.output_dir}/seasonal_attraction_patterns.png")
            
            # Calculate average sentiment by season
            seasonal_sentiment = df.groupby('season')['sentiment_score'].mean().reset_index()
            seasonal_sentiment = seasonal_sentiment.sort_values('sentiment_score', ascending=False)
            
            print("\nAverage sentiment by season:")
            print(seasonal_sentiment)
            
            # Create bar chart of seasonal sentiment
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x='season', y='sentiment_score', data=seasonal_sentiment, palette='viridis')
            plt.title('Average Sentiment by Season')
            plt.xlabel('Season')
            plt.ylabel('Average Sentiment Score (-1 to 1)')
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            
            # Add sentiment value labels
            for i, row in enumerate(seasonal_sentiment.itertuples()):
                ax.text(i, row.sentiment_score + 0.01, f'{row.sentiment_score:.2f}', 
                      ha='center', va='bottom', color='black', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/seasonal_sentiment.png')
            print(f"Saved seasonal sentiment chart to {self.output_dir}/seasonal_sentiment.png")
            
            # Save seasonal data
            seasonal_attraction.to_csv(f'{self.output_dir}/seasonal_attraction_patterns.csv')
            seasonal_sentiment.to_csv(f'{self.output_dir}/seasonal_sentiment.csv', index=False)
            
        except Exception as e:
            print(f"Error in seasonal analysis: {str(e)}")
        
        return df
    
    def analyze_attraction_satisfaction(self, df):
        """
        Analyze satisfaction with different attraction types.
        
        Parameters:
        - df: DataFrame with attraction reviews
        
        Returns:
        - DataFrame with attraction satisfaction analysis
        """
        print("Analyzing satisfaction with different attraction types...")
        
        attraction_cols = [f'attraction_{category}' for category in self.attraction_categories.keys()]
        
        # Calculate sentiment and rating (if available) for each attraction type
        attraction_metrics = []
        
        for col in attraction_cols:
            attraction_type = col.replace('attraction_', '')
            attraction_reviews = df[df[col] == 1]
            
            if len(attraction_reviews) >= 3:  # Only analyze if we have enough data
                metrics = {
                    'attraction_type': attraction_type,
                    'mentions': len(attraction_reviews),
                    'sentiment': attraction_reviews['sentiment_score'].mean()
                }
                
                # Add rating if available
                if 'rating' in attraction_reviews.columns:
                    metrics['rating'] = attraction_reviews['rating'].mean()
                
                attraction_metrics.append(metrics)
        
        if not attraction_metrics:
            print("Not enough data for attraction satisfaction analysis")
            return df
        
        # Convert to DataFrame and sort by sentiment
        metrics_df = pd.DataFrame(attraction_metrics)
        metrics_df = metrics_df.sort_values('sentiment', ascending=False)
        
        print("attraction satisfaction metrics:")
        print(metrics_df)
        
        # Create visualization of attraction satisfaction
        plt.figure(figsize=(12, 8))
        
        # Create a subplot for sentiment
        plt.subplot(2, 1, 1)
        bars = plt.barh(metrics_df['attraction_type'], metrics_df['sentiment'], color='skyblue')
        plt.title('Average Sentiment by attraction Type')
        plt.xlabel('Average Sentiment Score (-1 to 1)')
        plt.axvline(x=0, color='r', linestyle='-', alpha=0.3)
        
        # Add sentiment labels
        for bar, value in zip(bars, metrics_df['sentiment']):
            plt.text(value + 0.02, bar.get_y() + bar.get_height()/2, 
                   f'{value:.2f}', va='center', color='black', fontweight='bold')
        
        # Create a subplot for rating if available
        if 'rating' in metrics_df.columns:
            plt.subplot(2, 1, 2)
            bars = plt.barh(metrics_df['attraction_type'], metrics_df['rating'], color='lightgreen')
            plt.title('Average Rating by attraction Type')
            plt.xlabel('Average Rating (1-5)')
            plt.xlim(1, 5)
            
            # Add rating labels
            for bar, value in zip(bars, metrics_df['rating']):
                plt.text(value + 0.1, bar.get_y() + bar.get_height()/2, 
                       f'{value:.2f}', va='center', color='black', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/attraction_satisfaction.png')
        print(f"Saved attraction satisfaction chart to {self.output_dir}/attraction_satisfaction.png")
        
        # Save metrics to file
        metrics_df.to_csv(f'{self.output_dir}/attraction_satisfaction_metrics.csv', index=False)
        
        return df
    
    def generate_attraction_recommendations(self, df):
        """
        Generate attraction-specific recommendations based on the analysis.
        
        Parameters:
        - df: DataFrame with attraction reviews and analysis
        
        Returns:
        - Dictionary with attraction recommendations
        """
        print("Generating attraction-specific recommendations...")
        
        recommendations = {
            "attraction_offerings": [],
            "experience_quality": [],
            "seasonal_considerations": [],
            "visitor_segments": [],
            "marketing": []
        }
        
        # attraction offerings recommendations
        attraction_cols = [f'attraction_{category}' for category in self.attraction_categories.keys()]
        if any(col in df.columns for col in attraction_cols):
            # Check mention counts for each attraction type
            attraction_mentions = {col.replace('attraction_', ''): df[col].sum() 
                                for col in attraction_cols if col in df.columns}
            
            if attraction_mentions:
                # Identify most and least mentioned activities
                most_mentioned = max(attraction_mentions.items(), key=lambda x: x[1])
                least_mentioned = min(attraction_mentions.items(), key=lambda x: x[1])
                
                if most_mentioned[1] > 10:  # If mentioned frequently
                    recommendations["attraction_offerings"].append(
                        f"Leverage high interest in {most_mentioned[0].replace('_', ' ')} activities "
                        f"({most_mentioned[1]} mentions)"
                    )
                
                if least_mentioned[1] < 5:  # If rarely mentioned
                    recommendations["attraction_offerings"].append(
                        f"Consider developing more {least_mentioned[0].replace('_', ' ')} options "
                        f"to diversify offerings (only {least_mentioned[1]} mentions)"
                    )
                
                # Calculate sentiment for each attraction type
                sentiment_by_attraction = {}
                for attraction, col in zip(self.attraction_categories.keys(), attraction_cols):
                    if col in df.columns and df[col].sum() >= 3:
                        sentiment = df[df[col] == 1]['sentiment_score'].mean()
                        sentiment_by_attraction[attraction] = sentiment
                
                if sentiment_by_attraction:
                    # Identify activities with most positive and negative sentiment
                    most_positive = max(sentiment_by_attraction.items(), key=lambda x: x[1])
                    most_negative = min(sentiment_by_attraction.items(), key=lambda x: x[1])
                    
                    if most_positive[1] > 0.2:  # Very positive
                        recommendations["marketing"].append(
                            f"Highlight {most_positive[0].replace('_', ' ')} activities in marketing "
                            f"as they receive the most positive sentiment ({most_positive[1]:.2f})"
                        )
                    
                    if most_negative[1] < 0:  # Negative
                        recommendations["experience_quality"].append(
                            f"Improve {most_negative[0].replace('_', ' ')} experiences "
                            f"which receive negative sentiment ({most_negative[1]:.2f})"
                        )
        
        # Experience quality recommendations
        aspect_cols = [f'aspect_{aspect}' for aspect in self.experience_aspects.keys()]
        if any(col in df.columns for col in aspect_cols):
            # Calculate sentiment for each experience aspect
            sentiment_by_aspect = {}
            for aspect, col in zip(self.experience_aspects.keys(), aspect_cols):
                if col in df.columns and df[col].sum() >= 3:
                    sentiment = df[df[col] == 1]['sentiment_score'].mean()
                    sentiment_by_aspect[aspect] = sentiment
            
            if sentiment_by_aspect:
                # Identify aspects with negative sentiment
                negative_aspects = {k: v for k, v in sentiment_by_aspect.items() if v < 0}
                for aspect, sentiment in negative_aspects.items():
                    recommendations["experience_quality"].append(
                        f"Address issues with {aspect.replace('_', ' ')} which has negative sentiment ({sentiment:.2f})"
                    )
                
                # Identify aspects with very positive sentiment
                positive_aspects = {k: v for k, v in sentiment_by_aspect.items() if v > 0.3}
                for aspect, sentiment in positive_aspects.items():
                    recommendations["marketing"].append(
                        f"Emphasize {aspect.replace('_', ' ')} in marketing materials as a strength ({sentiment:.2f})"
                    )
        
        # Seasonal recommendations
        if 'season' in df.columns:
            try:
                # Calculate sentiment by season
                seasonal_sentiment = df.groupby('season')['sentiment_score'].mean()
                
                # Identify best and worst seasons
                best_season = seasonal_sentiment.idxmax()
                worst_season = seasonal_sentiment.idxmin()
                best_score = seasonal_sentiment.max()
                worst_score = seasonal_sentiment.min()
                
                if best_score > 0:
                    recommendations["seasonal_considerations"].append(
                        f"Promote {best_season} as the optimal season for activities (sentiment: {best_score:.2f})"
                    )
                
                if worst_score < 0:
                    recommendations["seasonal_considerations"].append(
                        f"Develop strategies to improve visitor experience during {worst_season} "
                        f"(sentiment: {worst_score:.2f})"
                    )
                
                # Seasonal attraction preferences
                attraction_cols = [f'attraction_{category}' for category in self.attraction_categories.keys()]
                if any(col in df.columns for col in attraction_cols):
                    seasonal_activities = df.groupby('season')[attraction_cols].mean()
                    
                    for season in seasonal_activities.index:
                        season_data = seasonal_activities.loc[season]
                        top_attraction_col = season_data.idxmax()
                        top_attraction = top_attraction_col.replace('attraction_', '').replace('_', ' ')
                        top_score = season_data.max()
                        
                        if top_score > 0.2:  # If prominent
                            recommendations["seasonal_considerations"].append(
                                f"Focus on {top_attraction} activities during {season} season "
                                f"(prevalence: {top_score:.2f})"
                            )
            except Exception as e:
                print(f"Error generating seasonal recommendations: {str(e)}")
        
        # Visitor segment recommendations
        if 'trip_type_standard' in df.columns:
            segment_top_activities = {}
            
            # For each segment, find the preferred attraction
            for segment in df['trip_type_standard'].unique():
                if pd.notna(segment) and segment != 'unknown':
                    segment_df = df[df['trip_type_standard'] == segment]
                    
                    if len(segment_df) >= 5:  # Only analyze segments with enough data
                        attraction_cols = [f'attraction_{cat}' for cat in self.attraction_categories.keys()]
                        avg_mentions = segment_df[attraction_cols].mean()
                        
                        if not avg_mentions.empty:
                            top_attraction_col = avg_mentions.idxmax()
                            top_attraction = top_attraction_col.replace('attraction_', '').replace('_', ' ')
                            top_score = avg_mentions.max()
                            
                            if top_score > 0.1:  # If notable preference
                                segment_top_activities[segment] = (top_attraction, top_score)
            
            # Generate recommendations based on segment preferences
            for segment, (attraction, score) in segment_top_activities.items():
                recommendations["visitor_segments"].append(
                    f"Target {segment} travelers with {attraction} experiences "
                    f"(preference score: {score:.2f})"
                )
        
        # Save recommendations
        with open(f'{self.output_dir}/attraction_recommendations.json', 'w', encoding='utf-8') as f:
            json.dump(recommendations, f, indent=2)
        
        # Print key recommendations
        print("\nKey attraction recommendations:")
        for category, recs in recommendations.items():
            if recs:
                print(f"\n{category.replace('_', ' ').title()}:")
                for rec in recs:
                    print(f"  - {rec}")
        
        return recommendations
    
    def run_analysis(self, df=None):
        """
        Run the complete attraction analysis.
        
        Parameters:
        - df: DataFrame with review data (optional, will get from base analyzer if None)
        
        Returns:
        - DataFrame with attraction analysis results
        """
        if df is None:
            # Get the processed data from the base analyzer
            df = self.analyzer.get_processed_data()
            
        if df is None or len(df) == 0:
            print("No data available for attraction analysis")
            return None
        
        print("\n=== Running Attraction Analysis ===")
        
        # Assuming df contains only attraction-related data
        attraction_df = df
        
        if len(attraction_df) == 0:
            print("No attraction reviews found in the data")
            return df
        
        # Run attraction-specific analyses
        attraction_df = self.analyze_attraction_types(attraction_df)
        attraction_df = self.analyze_experience_aspects(attraction_df)
        attraction_df = self.analyze_by_visitor_segment(attraction_df)
        attraction_df = self.analyze_seasonal_attraction_patterns(attraction_df)
        attraction_df = self.analyze_attraction_satisfaction(attraction_df)
        
        # Generate attraction-specific recommendations
        self.generate_attraction_recommendations(attraction_df)
        
        print("\nAttraction analysis complete.")
        return attraction_df
