import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
import calendar
import json
from collections import Counter
import numpy as np

class SeasonalPatterns:
    """
    Analyzes seasonal patterns in tourism reviews.
    Focuses on identifying how visitor experiences, preferences, and sentiments
    vary by season and month.
    """
    
    def __init__(self, reviews_df, output_dir='outputs/seasonal_patterns'):
        """
        Initializes the SeasonalPatterns object with the reviews DataFrame.
        
        Parameters:
        - reviews_df: DataFrame containing the review data
        - output_dir: Directory to save analysis outputs
        """
        self.reviews_df = reviews_df
        self.output_dir = output_dir
        self.seasonal_data = None
        
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created seasonal patterns directory: {output_dir}")
            
        # Define seasons for Southern Hemisphere
        self.seasons = {
            'Summer': [12, 1, 2],
            'Autumn': [3, 4, 5],
            'Winter': [6, 7, 8],
            'Spring': [9, 10, 11]
        }
        
        # Weather-related keywords for weather mention detection
        self.weather_keywords = {
            'positive_weather': ['sunny', 'beautiful', 'perfect', 'pleasant', 'warm', 'clear', 
                                'mild', 'sunshine', 'breeze', 'gorgeous'],
            'negative_weather': ['rain', 'rainy', 'storm', 'wind', 'windy', 'humid', 'hot', 
                                'cold', 'cyclone', 'hurricane', 'flood', 'wet'],
            'neutral_weather': ['weather', 'climate', 'temperature', 'season', 'forecast', 
                              'humidity', 'precipitation', 'seasonal']
        }
        
        # Month to season mapping
        self.month_to_season = {
            1: 'Summer', 2: 'Summer', 3: 'Autumn',
            4: 'Autumn', 5: 'Autumn', 6: 'Winter',
            7: 'Winter', 8: 'Winter', 9: 'Spring',
            10: 'Spring', 11: 'Spring', 12: 'Summer'
        }

    def assign_seasons(self):
        """
        Assigns a season to each review based on the month of the published date.
        Maps months to seasons using Southern Hemisphere adjustments.
        
        Returns:
        - DataFrame with season column added
        """
        print("Assigning seasons to reviews...")
        
        # Check if published_date exists
        if 'published_date' not in self.reviews_df.columns:
            print("Warning: No 'published_date' column found. Cannot assign seasons.")
            return self.reviews_df
            
        # Ensure published_date is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.reviews_df['published_date']):
            self.reviews_df['published_date'] = pd.to_datetime(self.reviews_df['published_date'], errors='coerce')
        
        # Extract month and assign season
        self.reviews_df['month'] = self.reviews_df['published_date'].dt.month
        self.reviews_df['month_name'] = self.reviews_df['published_date'].dt.strftime('%b')
        self.reviews_df['season'] = self.reviews_df['month'].map(self.month_to_season)
        
        # Basic stats
        season_counts = self.reviews_df['season'].value_counts()
        print("\nReviews by season:")
        for season, count in season_counts.items():
            print(f"  {season}: {count} reviews")
            
        return self.reviews_df

    def detect_weather_mentions(self):
        """
        Detect mentions of weather in reviews.
        
        Returns:
        - DataFrame with weather mention flags
        """
        print("Detecting weather mentions in reviews...")
        
        # Check if text column exists
        if 'text' not in self.reviews_df.columns:
            print("Warning: No 'text' column found. Cannot detect weather mentions.")
            return self.reviews_df
            
        # Add weather mention flags
        for weather_type, keywords in self.weather_keywords.items():
            pattern = '|'.join(keywords)
            self.reviews_df[f'mentions_{weather_type}'] = self.reviews_df['text'].str.lower().str.contains(
                pattern, na=False).astype(int)
        
        # General weather mention flag
        self.reviews_df['mentions_weather'] = (
            self.reviews_df['mentions_positive_weather'] | 
            self.reviews_df['mentions_negative_weather'] | 
            self.reviews_df['mentions_neutral_weather']
        ).astype(int)
        
        # Print weather mention statistics
        weather_counts = {
            'Total weather mentions': self.reviews_df['mentions_weather'].sum(),
            'Positive weather mentions': self.reviews_df['mentions_positive_weather'].sum(),
            'Negative weather mentions': self.reviews_df['mentions_negative_weather'].sum(),
            'Neutral weather mentions': self.reviews_df['mentions_neutral_weather'].sum()
        }
        
        print("\nWeather mention statistics:")
        for label, count in weather_counts.items():
            print(f"  {label}: {count}")
            
        return self.reviews_df

    def calculate_seasonal_stats(self):
        """
        Calculates basic statistics for each season: count of reviews, average rating,
        and sentiment score if available.
        
        Returns:
        - DataFrame with seasonal statistics
        """
        print("Calculating seasonal statistics...")
        
        # Check if season column exists
        if 'season' not in self.reviews_df.columns:
            print("Warning: No 'season' column found. Run assign_seasons() first.")
            return None
            
        # Define aggregation functions
        agg_funcs = {
            'rating': ['count', 'mean', 'std', 'median']
        }
        
        # Add sentiment if available
        if 'sentiment_score' in self.reviews_df.columns:
            agg_funcs['sentiment_score'] = ['mean', 'std', 'median']
            
        # Add weather mentions if available
        for weather_type in ['mentions_weather', 'mentions_positive_weather', 'mentions_negative_weather']:
            if weather_type in self.reviews_df.columns:
                agg_funcs[weather_type] = ['sum', 'mean']
                
        # Calculate statistics
        self.seasonal_data = self.reviews_df.groupby('season').agg(agg_funcs)
        
        # Flatten the column hierarchy
        self.seasonal_data.columns = ['_'.join(col).strip() for col in self.seasonal_data.columns.values]
        
        # Rename columns for clarity
        self.seasonal_data = self.seasonal_data.rename(columns={
            'rating_count': 'review_count',
            'rating_mean': 'average_rating'
        })
        
        # Save to CSV
        self.seasonal_data.to_csv(os.path.join(self.output_dir, 'seasonal_statistics.csv'))
        
        print("Seasonal Statistics Calculated:")
        print(self.seasonal_data)
        
        return self.seasonal_data

    def visualize_seasonal_trends(self):
        """
        Generates visualizations for seasonal review counts and average ratings.
        
        Returns:
        - None, but saves visualizations to the output directory
        """
        print("Visualizing seasonal trends...")
        
        # Check if seasonal data is available
        if self.seasonal_data is None:
            print("Warning: No seasonal data available. Run calculate_seasonal_stats() first.")
            return
            
        # Create directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # 1. Combined seasonal trends (count and rating)
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Plot review counts
        color = 'tab:blue'
        ax1.set_xlabel('Season')
        ax1.set_ylabel('Count of Reviews', color=color)
        bars = self.seasonal_data['review_count'].plot(kind='bar', ax=ax1, color=color)
        
        # Add count labels
        for bar in bars.patches:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width()/2., 
                height + 5, 
                f'{int(height)}', 
                ha='center', va='bottom', 
                color=color, fontweight='bold'
            )
            
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Plot average ratings
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Average Rating', color=color)
        self.seasonal_data['average_rating'].plot(kind='line', marker='o', ax=ax2, color=color, linewidth=2)
        
        # Add rating labels
        for i, val in enumerate(self.seasonal_data['average_rating']):
            ax2.text(
                i, 
                val + 0.1, 
                f'{val:.2f}', 
                ha='center', va='bottom', 
                color=color, fontweight='bold'
            )
            
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(1, 5.5)  # Assuming 5-star rating scale
        
        plt.title('Seasonal Review Counts and Average Ratings')
        fig.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'seasonal_trends.png'), dpi=300)
        plt.close()
        
        # 2. Sentiment by season if available
        if 'sentiment_score_mean' in self.seasonal_data.columns:
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(
                x=self.seasonal_data.index, 
                y=self.seasonal_data['sentiment_score_mean'],
                palette='viridis'
            )
            
            # Add sentiment labels
            for i, val in enumerate(self.seasonal_data['sentiment_score_mean']):
                ax.text(
                    i, 
                    val + 0.01 if val >= 0 else val - 0.05, 
                    f'{val:.2f}', 
                    ha='center', va='bottom' if val >= 0 else 'top', 
                    fontweight='bold'
                )
                
            plt.title('Average Sentiment by Season')
            plt.xlabel('Season')
            plt.ylabel('Average Sentiment Score')
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'sentiment_by_season.png'), dpi=300)
            plt.close()
            
        # 3. Weather mentions by season if available
        weather_cols = [col for col in self.seasonal_data.columns if col.startswith('mentions_') and col.endswith('_mean')]
        if weather_cols:
            plt.figure(figsize=(12, 6))
            
            # Create a cleaned DataFrame for plotting
            weather_data = self.seasonal_data[weather_cols].copy()
            weather_data.columns = [col.replace('mentions_', '').replace('_mean', '') for col in weather_cols]
            
            # Plot
            ax = weather_data.plot(kind='bar', colormap='viridis')
            
            plt.title('Weather Mentions by Season')
            plt.xlabel('Season')
            plt.ylabel('Mention Rate')
            plt.legend(title='Weather Type')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'weather_mentions_by_season.png'), dpi=300)
            plt.close()
            
        print(f"Seasonal visualizations saved to {self.output_dir}")

    def analyze_monthly_patterns(self):
        """
        Analyze patterns by month, including ratings, sentiment, and weather mentions.
        
        Returns:
        - DataFrame with monthly statistics
        """
        print("Analyzing monthly patterns...")
        
        # Check if month column exists
        if 'month' not in self.reviews_df.columns:
            print("Warning: No 'month' column found. Run assign_seasons() first.")
            return None
            
        # Define aggregation functions
        agg_funcs = {
            'rating': ['count', 'mean', 'std']
        }
        
        # Add sentiment if available
        if 'sentiment_score' in self.reviews_df.columns:
            agg_funcs['sentiment_score'] = ['mean', 'std']
            
        # Add weather mentions if available
        for weather_type in ['mentions_weather', 'mentions_positive_weather', 'mentions_negative_weather']:
            if weather_type in self.reviews_df.columns:
                agg_funcs[weather_type] = ['sum', 'mean']
                
        # Calculate statistics
        monthly_data = self.reviews_df.groupby('month').agg(agg_funcs)
        
        # Flatten the column hierarchy
        monthly_data.columns = ['_'.join(col).strip() for col in monthly_data.columns.values]
        
        # Rename columns for clarity
        monthly_data = monthly_data.rename(columns={
            'rating_count': 'review_count',
            'rating_mean': 'average_rating'
        })
        
        # Add month names for better display
        month_names = {i: calendar.month_name[i] for i in range(1, 13)}
        monthly_data['month_name'] = monthly_data.index.map(month_names)
        
        # Save to CSV
        monthly_data.to_csv(os.path.join(self.output_dir, 'monthly_statistics.csv'))
        
        # Create visualizations
        self._visualize_monthly_patterns(monthly_data)
        
        return monthly_data

    def _visualize_monthly_patterns(self, monthly_data):
        """
        Generate visualizations for monthly patterns.
        
        Parameters:
        - monthly_data: DataFrame with monthly statistics
        
        Returns:
        - None, but saves visualizations to the output directory
        """
        # Create directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # Sort data by month for proper chronological display
        monthly_data = monthly_data.sort_index()
        
        # Set month names in the correct order for x-axis labels
        month_names = [calendar.month_abbr[i] for i in range(1, 13)]
        
        # 1. Monthly review counts
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(
            x=[month_names[i-1] for i in monthly_data.index], 
            y=monthly_data['review_count'],
            palette='Blues_d'
        )
        
        # Add count labels
        for i, val in enumerate(monthly_data['review_count']):
            ax.text(
                i, 
                val + 2, 
                f'{int(val)}', 
                ha='center', va='bottom', 
                fontweight='bold'
            )
            
        plt.title('Number of Reviews by Month')
        plt.xlabel('Month')
        plt.ylabel('Count of Reviews')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'monthly_review_counts.png'), dpi=300)
        plt.close()
        
        # 2. Monthly average ratings
        plt.figure(figsize=(12, 6))
        ax = sns.lineplot(
            x=[month_names[i-1] for i in monthly_data.index], 
            y=monthly_data['average_rating'],
            marker='o',
            color='tab:red',
            linewidth=2
        )
        
        # Add rating labels
        for i, val in enumerate(monthly_data['average_rating']):
            ax.text(
                i, 
                val + 0.1, 
                f'{val:.2f}', 
                ha='center', va='bottom', 
                fontweight='bold'
            )
            
        plt.title('Average Rating by Month')
        plt.xlabel('Month')
        plt.ylabel('Average Rating')
        plt.ylim(1, 5.5)  # Assuming 5-star rating scale
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'monthly_average_ratings.png'), dpi=300)
        plt.close()
        
        # 3. Monthly sentiment if available
        if 'sentiment_score_mean' in monthly_data.columns:
            plt.figure(figsize=(12, 6))
            ax = sns.lineplot(
                x=[month_names[i-1] for i in monthly_data.index], 
                y=monthly_data['sentiment_score_mean'],
                marker='o',
                color='tab:blue',
                linewidth=2
            )
            
            # Add sentiment labels
            for i, val in enumerate(monthly_data['sentiment_score_mean']):
                ax.text(
                    i, 
                    val + 0.01 if val >= 0 else val - 0.05, 
                    f'{val:.2f}', 
                    ha='center', va='bottom' if val >= 0 else 'top', 
                    fontweight='bold'
                )
                
            plt.title('Average Sentiment by Month')
            plt.xlabel('Month')
            plt.ylabel('Average Sentiment Score')
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'monthly_sentiment.png'), dpi=300)
            plt.close()
            
        # 4. Monthly weather mentions if available
        weather_cols = [col for col in monthly_data.columns if col.startswith('mentions_') and col.endswith('_mean')]
        if weather_cols:
            plt.figure(figsize=(12, 6))
            
            # Create a cleaned DataFrame for plotting
            weather_data = monthly_data[weather_cols].copy()
            weather_data.columns = [col.replace('mentions_', '').replace('_mean', '') for col in weather_cols]
            
            # Plot each weather type
            for col in weather_data.columns:
                plt.plot(
                    [month_names[i-1] for i in monthly_data.index],
                    weather_data[col],
                    marker='o',
                    linewidth=2,
                    label=col.replace('_', ' ').title()
                )
                
            plt.title('Weather Mentions by Month')
            plt.xlabel('Month')
            plt.ylabel('Mention Rate')
            plt.legend(title='Weather Type')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'monthly_weather_mentions.png'), dpi=300)
            plt.close()
            
        print(f"Monthly visualizations saved to {self.output_dir}")

    def extract_seasonal_keywords(self):
        """
        Generates and saves a word cloud based on the review texts for each season.
        
        Returns:
        - None, but saves word clouds to the output directory
        """
        print("Extracting seasonal keywords and generating word clouds...")
        
        # Check if required columns exist
        if 'season' not in self.reviews_df.columns or 'text' not in self.reviews_df.columns:
            print("Warning: Required columns ('season', 'text') not found.")
            return
            
        # Create word cloud directory
        wordcloud_dir = os.path.join(self.output_dir, 'wordclouds')
        if not os.path.exists(wordcloud_dir):
            os.makedirs(wordcloud_dir)
            
        # Generate word clouds for each season
        for season, group in self.reviews_df.groupby('season'):
            # Skip if no data
            if len(group) == 0:
                continue
                
            # Combine all text
            text = ' '.join(group['text'].fillna(''))
            
            # Generate word cloud
            wordcloud = WordCloud(
                max_font_size=50, 
                max_words=100, 
                background_color="white",
                width=800,
                height=400,
                colormap='viridis'
            ).generate(text)
            
            # Create a figure
            plt.figure(figsize=(10, 6))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.title(f'Word Cloud for {season}')
            plt.tight_layout(pad=0)
            plt.savefig(os.path.join(wordcloud_dir, f'wordcloud_{season}.png'), dpi=300)
            plt.close()
            
            # Also extract and save top keywords
            words = text.lower().split()
            word_counts = Counter(words)
            
            # Filter out common stopwords
            stopwords = ['the', 'and', 'to', 'of', 'a', 'in', 'for', 'is', 'on', 'with', 'was',
                        'we', 'it', 'at', 'by', 'our', 'as', 'were', 'but', 'that', 'had', 'be',
                        'not', 'this', 'are', 'have', 'from', 'an', 'they', 'you', 'i', 'my',
                        'so', 'there', 'just', 'very', 'all', 'their', 'would', 'could', 'which']
                        
            # Get top keywords excluding stopwords and single-character words
            top_keywords = [(word, count) for word, count in word_counts.most_common(30) 
                           if word not in stopwords and len(word) > 1]
            
            # Save to file
            with open(os.path.join(wordcloud_dir, f'keywords_{season}.txt'), 'w') as f:
                for word, count in top_keywords:
                    f.write(f"{word}: {count}\n")
                    
        print(f"Word clouds and keyword lists saved to {wordcloud_dir}")

    def generate_seasonal_recommendations(self):
        """
        Generate recommendations based on seasonal patterns.
        
        Returns:
        - Dictionary with seasonal recommendations
        """
        print("Generating seasonal recommendations...")
        
        # Check if necessary data is available
        if self.seasonal_data is None:
            print("Warning: No seasonal data available. Run calculate_seasonal_stats() first.")
            return {}
            
        recommendations = {
            "marketing": [],
            "product_development": [],
            "operational": [],
            "weather_related": []
        }
        
        # Generate recommendations based on ratings
        if 'average_rating' in self.seasonal_data.columns:
            best_season = self.seasonal_data['average_rating'].idxmax()
            worst_season = self.seasonal_data['average_rating'].idxmin()
            best_rating = self.seasonal_data.loc[best_season, 'average_rating']
            worst_rating = self.seasonal_data.loc[worst_season, 'average_rating']
            
            recommendations["marketing"].append(
                f"Highlight {best_season} as the optimal season to visit Tonga "
                f"(average rating: {best_rating:.2f}/5)"
            )
            
            if worst_rating < 4.0:  # Only recommend if rating is below 4 stars
                recommendations["operational"].append(
                    f"Improve service quality during {worst_season} "
                    f"when ratings are lower (average: {worst_rating:.2f}/5)"
                )
                
        # Generate recommendations based on sentiment if available
        if 'sentiment_score_mean' in self.seasonal_data.columns:
            best_sentiment_season = self.seasonal_data['sentiment_score_mean'].idxmax()
            worst_sentiment_season = self.seasonal_data['sentiment_score_mean'].idxmin()
            best_sentiment = self.seasonal_data.loc[best_sentiment_season, 'sentiment_score_mean']
            worst_sentiment = self.seasonal_data.loc[worst_sentiment_season, 'sentiment_score_mean']
            
            if best_sentiment > 0.2:
                recommendations["marketing"].append(
                    f"Use positive testimonials from {best_sentiment_season} visitors "
                    f"in marketing materials (sentiment: {best_sentiment:.2f})"
                )
                
            if worst_sentiment < 0:
                recommendations["operational"].append(
                    f"Address specific issues mentioned in {worst_sentiment_season} reviews "
                    f"to improve visitor sentiment (current: {worst_sentiment:.2f})"
                )
                
        # Weather-related recommendations
        if 'mentions_negative_weather_mean' in self.seasonal_data.columns:
            worst_weather_season = self.seasonal_data['mentions_negative_weather_mean'].idxmax()
            worst_weather_rate = self.seasonal_data.loc[worst_weather_season, 'mentions_negative_weather_mean']
            
            if worst_weather_rate > 0.1:  # Only recommend if significant mention rate
                recommendations["weather_related"].append(
                    f"Set realistic weather expectations for visitors during {worst_weather_season} "
                    f"when negative weather mentions peak ({worst_weather_rate:.2f})"
                )
                
                recommendations["product_development"].append(
                    f"Develop more indoor or weather-resistant activities for {worst_weather_season}"
                )
                
        # Save recommendations to file
        with open(os.path.join(self.output_dir, 'seasonal_recommendations.json'), 'w', encoding='utf-8') as f:
            json.dump(recommendations, f, indent=2)
            
        # Print key recommendations
        print("\nKey seasonal recommendations:")
        for category, recs in recommendations.items():
            if recs:
                print(f"\n{category.replace('_', ' ').title()}:")
                for rec in recs:
                    print(f"  - {rec}")
                    
        return recommendations

    def run_analysis(self):
        """
        Run the complete seasonal patterns analysis pipeline.
        
        Returns:
        - DataFrame with seasonal data
        """
        print("\n=== Running Seasonal Patterns Analysis ===")
        
        # Check if we have data
        if self.reviews_df is None or len(self.reviews_df) == 0:
            print("No data available for seasonal analysis")
            return None
            
        # Assign seasons
        self.assign_seasons()
        
        # Detect weather mentions
        self.detect_weather_mentions()
        
        # Calculate seasonal statistics
        self.calculate_seasonal_stats()
        
        # Visualize seasonal trends
        self.visualize_seasonal_trends()
        
        # Analyze monthly patterns
        self.analyze_monthly_patterns()
        
        # Extract seasonal keywords
        self.extract_seasonal_keywords()
        
        # Generate recommendations
        self.generate_seasonal_recommendations()
        
        print("\nSeasonal patterns analysis complete.")
        
        return self.seasonal_data