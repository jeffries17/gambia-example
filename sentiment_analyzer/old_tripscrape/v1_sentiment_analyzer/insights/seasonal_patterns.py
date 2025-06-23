import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import calendar
import json

class SeasonalPatternAnalyzer:
    """
    Analyzes seasonal patterns in tourism reviews.
    Focuses on identifying how visitor experiences, preferences, and sentiments
    vary by season and month.
    """
    
    def __init__(self, output_dir='seasonal_insights'):
        """
        Initialize the seasonal pattern analyzer.
        
        Parameters:
        - output_dir: Directory to save seasonal analysis outputs
        """
        self.output_dir = output_dir
        
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created seasonal insights directory: {output_dir}")
        
        # Define seasons (for Southern Hemisphere)
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
    
    def extract_date_features(self, df):
        """
        Extract month, season, and other date features from review data.
        
        Parameters:
        - df: DataFrame with review data including publishedDate or similar date field
        
        Returns:
        - DataFrame with added date features
        """
        print("Extracting seasonal and date features...")
        
        date_col = None
        
        # Check for various possible date column names
        possible_date_cols = ['publishedDate', 'date', 'review_date', 'travelDate']
        for col in possible_date_cols:
            if col in df.columns:
                date_col = col
                break
        
        if date_col is None:
            print("No date column found for seasonal analysis")
            return df
        
        try:
            # Convert to datetime
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            
            # Extract basic date components
            df['year'] = df[date_col].dt.year
            df['month'] = df[date_col].dt.month
            df['month_name'] = df[date_col].dt.strftime('%b')
            
            # Add season based on month
            df['season'] = df['month'].apply(self._get_season)
            
            # Add quarter
            df['quarter'] = df[date_col].dt.quarter
            
            # Check if the date is in high tourism season (typically Dec-Feb for Tonga)
            df['is_high_season'] = df['month'].isin([12, 1, 2]).astype(int)
            
            print(f"Added date features based on {date_col} column")
            
            # Basic stats on the date distribution
            year_counts = df['year'].value_counts().sort_index()
            print("\nReviews by year:")
            for year, count in year_counts.items():
                print(f"  {year}: {count} reviews")
                
            season_counts = df['season'].value_counts()
            print("\nReviews by season:")
            for season, count in season_counts.items():
                print(f"  {season}: {count} reviews")
            
            return df
            
        except Exception as e:
            print(f"Error extracting date features: {str(e)}")
            return df
    
    def _get_season(self, month):
        """
        Helper method to convert month to season.
        Uses Southern Hemisphere seasons.
        
        Parameters:
        - month: Month number (1-12)
        
        Returns:
        - Season name (Summer, Autumn, Winter, Spring)
        """
        for season, months in self.seasons.items():
            if month in months:
                return season
        return 'Unknown'
    
    def detect_weather_mentions(self, df):
        """
        Detect mentions of weather in reviews.
        
        Parameters:
        - df: DataFrame with review data
        
        Returns:
        - DataFrame with weather mention flags
        """
        print("Detecting weather mentions in reviews...")
        
        for weather_type, keywords in self.weather_keywords.items():
            # Escape parentheses in keywords to avoid regex match groups
            escaped_keywords = []
            for keyword in keywords:
                escaped_keyword = keyword.replace('(', '\\(').replace(')', '\\)')
                escaped_keywords.append(escaped_keyword)
            
            pattern = r'\b(' + '|'.join(escaped_keywords) + r')\b'
            df[f'mentions_{weather_type}'] = df['text'].str.lower().str.contains(
                pattern, na=False).astype(int)
        
        # General weather mention flag
        df['mentions_weather'] = (df['mentions_positive_weather'] | 
                                df['mentions_negative_weather'] | 
                                df['mentions_neutral_weather']).astype(int)
        
        weather_counts = {
            'Total weather mentions': df['mentions_weather'].sum(),
            'Positive weather mentions': df['mentions_positive_weather'].sum(),
            'Negative weather mentions': df['mentions_negative_weather'].sum(),
            'Neutral weather mentions': df['mentions_neutral_weather'].sum()
        }
        
        print("\nWeather mention statistics:")
        for label, count in weather_counts.items():
            print(f"  {label}: {count}")
        
        return df
    
    def analyze_seasonal_patterns(self, df):
        """
        Analyze review patterns by season.
        
        Parameters:
        - df: DataFrame with review data including season
        
        Returns:
        - DataFrame with seasonal analysis
        """
        if 'season' not in df.columns:
            print("Season column not found, cannot analyze seasonal patterns")
            return df
        
        print("\nAnalyzing patterns by season...")
        
        # Calculate average sentiment by season
        seasonal_sentiment = df.groupby('season')['sentiment_score'].mean().reset_index()
        seasonal_sentiment = seasonal_sentiment.sort_values('sentiment_score', ascending=False)
        
        print("Average sentiment by season:")
        print(seasonal_sentiment)
        
        # Create a bar chart of seasonal sentiment
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='season', y='sentiment_score', hue='season', data=seasonal_sentiment, legend=False)
        plt.title('Average Review Sentiment by Season')
        plt.xlabel('Season')
        plt.ylabel('Average Sentiment Score (-1 to 1)')
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        
        # Add sentiment value labels
        for i, row in enumerate(seasonal_sentiment.itertuples()):
            ax.text(i, row.sentiment_score + 0.01, f'{row.sentiment_score:.2f}', 
                  ha='center', va='bottom', color='black', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/sentiment_by_season.png')
        print(f"Saved seasonal sentiment chart to {self.output_dir}/sentiment_by_season.png")
        
        # Analyze rating by season if available
        if 'rating' in df.columns:
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
            seasonal_rating = df.groupby('season')['rating'].mean().reset_index()
            seasonal_rating = seasonal_rating.sort_values('rating', ascending=False)
            
            print("\nAverage rating by season:")
            print(seasonal_rating)
            
            # Create a bar chart of seasonal ratings
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x='season', y='rating', hue='season', data=seasonal_rating, legend=False)
            plt.title('Average Review Rating by Season')
            plt.xlabel('Season')
            plt.ylabel('Average Rating (1-5)')
            plt.ylim(1, 5)
            
            # Add rating value labels
            for i, row in enumerate(seasonal_rating.itertuples()):
                ax.text(i, row.rating - 0.2, f'{row.rating:.2f}', 
                      ha='center', va='bottom', color='white', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/rating_by_season.png')
            print(f"Saved seasonal rating chart to {self.output_dir}/rating_by_season.png")
        
        # Weather mentions by season
        if 'mentions_weather' in df.columns:
            weather_by_season = df.groupby('season')[['mentions_positive_weather', 
                                                    'mentions_negative_weather', 
                                                    'mentions_neutral_weather']].mean().reset_index()
            
            print("\nWeather mentions by season:")
            print(weather_by_season)
            
            # Create a grouped bar chart of weather mentions by season
            plt.figure(figsize=(12, 6))
            weather_long = pd.melt(weather_by_season, id_vars=['season'], 
                                  value_vars=['mentions_positive_weather', 
                                             'mentions_negative_weather', 
                                             'mentions_neutral_weather'],
                                  var_name='weather_type', value_name='mention_rate')
            
            # Clean up weather type labels
            weather_long['weather_type'] = weather_long['weather_type'].str.replace('mentions_', '').str.replace('_', ' ')
            
            ax = sns.barplot(x='season', y='mention_rate', hue='weather_type', data=weather_long)
            plt.title('Weather Mentions by Season')
            plt.xlabel('Season')
            plt.ylabel('Mention Rate')
            plt.legend(title='Weather Type')
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/weather_by_season.png')
            print(f"Saved weather mentions chart to {self.output_dir}/weather_by_season.png")
        
        return df
    
    def analyze_monthly_patterns(self, df):
        """
        Analyze review patterns by month.
        
        Parameters:
        - df: DataFrame with review data including month
        
        Returns:
        - DataFrame with monthly analysis
        """
        if 'month' not in df.columns:
            print("Month column not found, cannot analyze monthly patterns")
            return df
        
        print("\nAnalyzing patterns by month...")
        
        # Create month order for better visualization
        month_order = list(range(1, 13))
        month_names = [calendar.month_abbr[m] for m in month_order]
        
        # Calculate average sentiment by month
        monthly_sentiment = df.groupby('month')['sentiment_score'].mean().reset_index()
        
        # Ensure all months are included (even with no data)
        all_months = pd.DataFrame({'month': month_order})
        monthly_sentiment = all_months.merge(monthly_sentiment, on='month', how='left').fillna(0)
        
        # Add month names
        monthly_sentiment['month_name'] = monthly_sentiment['month'].apply(lambda m: calendar.month_abbr[m])
        
        # Sort by month for visualization
        monthly_sentiment = monthly_sentiment.sort_values('month')
        
        print("Average sentiment by month:")
        print(monthly_sentiment)
        
        # Create a line chart of monthly sentiment
        plt.figure(figsize=(12, 6))
        ax = sns.pointplot(x='month_name', y='sentiment_score', data=monthly_sentiment, 
                          order=month_names, markers='o', linestyles='-', color='blue')
        plt.title('Average Review Sentiment by Month')
        plt.xlabel('Month')
        plt.ylabel('Average Sentiment Score (-1 to 1)')
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.grid(True, alpha=0.3)
        
        # Add sentiment value labels
        for i, row in enumerate(monthly_sentiment.itertuples()):
            if row.sentiment_score != 0:  # Only label non-zero values
                ax.text(i, row.sentiment_score + 0.01, f'{row.sentiment_score:.2f}', 
                      ha='center', va='bottom', color='black', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/sentiment_by_month.png')
        print(f"Saved monthly sentiment chart to {self.output_dir}/sentiment_by_month.png")
        
        # Analyze weather mentions by month if available
        if 'mentions_weather' in df.columns:
            monthly_weather = df.groupby('month')[['mentions_positive_weather', 
                                                 'mentions_negative_weather']].mean().reset_index()
            
            # Ensure all months are included
            monthly_weather = all_months.merge(monthly_weather, on='month', how='left').fillna(0)
            monthly_weather['month_name'] = monthly_weather['month'].apply(lambda m: calendar.month_abbr[m])
            monthly_weather = monthly_weather.sort_values('month')
            
            # Create a line chart of weather mentions by month
            plt.figure(figsize=(12, 6))
            
            plt.plot(range(len(month_names)), monthly_weather['mentions_positive_weather'], 
                   'o-', color='green', label='Positive weather mentions')
            plt.plot(range(len(month_names)), monthly_weather['mentions_negative_weather'], 
                   'o-', color='red', label='Negative weather mentions')
            
            plt.xticks(range(len(month_names)), month_names)
            plt.title('Weather Sentiment by Month')
            plt.xlabel('Month')
            plt.ylabel('Mention Rate')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/weather_sentiment_by_month.png')
            print(f"Saved monthly weather chart to {self.output_dir}/weather_sentiment_by_month.png")
        
        return df
    
    def analyze_theme_seasonality(self, df):
        """
        Analyze how themes vary by season.
        
        Parameters:
        - df: DataFrame with review data including themes and season
        
        Returns:
        - DataFrame with theme seasonality analysis
        """
        if 'season' not in df.columns:
            print("Season column not found, cannot analyze theme seasonality")
            return df
        
        print("\nAnalyzing theme seasonality...")
        
        # Find theme columns
        theme_cols = [col for col in df.columns if col.startswith('theme_')]
        
        if not theme_cols:
            print("No theme columns found for seasonal theme analysis")
            return df
        
        # Calculate theme prevalence by season
        seasonal_themes = df.groupby('season')[theme_cols].mean().reset_index()
        
        # Clean up column names for display
        theme_labels = {col: col.replace('theme_', '') for col in theme_cols}
        seasonal_themes = seasonal_themes.rename(columns=theme_labels)
        
        print("Theme prevalence by season:")
        print(seasonal_themes)
        
        # Create a heatmap of theme prevalence by season
        plt.figure(figsize=(14, 8))
        sns.heatmap(seasonal_themes.set_index('season'), annot=True, cmap='YlGnBu', fmt='.2f')
        plt.title('Theme Prevalence by Season')
        plt.ylabel('Season')
        plt.xlabel('Theme')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/theme_seasonality.png')
        print(f"Saved theme seasonality chart to {self.output_dir}/theme_seasonality.png")
        
        # Identify the top theme for each season
        top_themes_by_season = {}
        
        for _, row in seasonal_themes.iterrows():
            season = row['season']
            # Create a dictionary without the 'season' column
            theme_data = {col: val for col, val in row.items() if col != 'season'}
            # Get the theme with the highest value
            top_theme = max(theme_data.items(), key=lambda x: x[1])
            top_themes_by_season[season] = top_theme
        
        print("\nTop theme by season:")
        for season, (theme, score) in top_themes_by_season.items():
            print(f"  {season}: {theme} ({score:.2f})")
        
        # Save seasonal themes data
        seasonal_themes.to_csv(f'{self.output_dir}/theme_seasonality.csv', index=False)
        
        return df
    
    def generate_seasonal_recommendations(self, df):
        """
        Generate recommendations based on seasonal patterns.
        
        Parameters:
        - df: DataFrame with seasonal analysis
        
        Returns:
        - Dictionary with seasonal recommendations
        """
        print("\nGenerating seasonal recommendations...")
        
        recommendations = {
            "marketing": [],
            "product_development": [],
            "operational": [],
            "weather_related": []
        }
        
        # Get the overall best and worst seasons based on sentiment
        if 'season' in df.columns:
            season_sentiment = df.groupby('season')['sentiment_score'].mean()
            if not season_sentiment.empty:
                best_season = season_sentiment.idxmax()
                worst_season = season_sentiment.idxmin()
                best_score = season_sentiment.max()
                worst_score = season_sentiment.min()
                
                if best_score > 0.1:
                    recommendations["marketing"].append(
                        f"Highlight {best_season} as the optimal season to visit Tonga "
                        f"(sentiment: {best_score:.2f})"
                    )
                
                if worst_score < 0:
                    recommendations["operational"].append(
                        f"Implement service improvement initiatives during {worst_season} "
                        f"to address negative experiences (sentiment: {worst_score:.2f})"
                    )
        
        # Get the months with highest and lowest sentiment
        if 'month' in df.columns:
            month_sentiment = df.groupby('month')['sentiment_score'].mean()
            if not month_sentiment.empty:
                # Convert to integer to avoid the numpy.float64 error
                best_month_idx = int(month_sentiment.idxmax())
                worst_month_idx = int(month_sentiment.idxmin())
                
                # Ensure the indices are valid (between 1 and 12)
                best_month_idx = max(1, min(12, best_month_idx))
                worst_month_idx = max(1, min(12, worst_month_idx))
                
                best_month = calendar.month_name[best_month_idx]
                worst_month = calendar.month_name[worst_month_idx]
                best_month_score = month_sentiment.max()
                worst_month_score = month_sentiment.min()
                
                if best_month_score > 0.2:
                    recommendations["marketing"].append(
                        f"Promote {best_month} in marketing materials as the best month to visit "
                        f"(sentiment: {best_month_score:.2f})"
                    )
                
                if worst_month_score < 0:
                    recommendations["operational"].append(
                        f"Develop special promotions or experiences for {worst_month} "
                        f"to improve visitor satisfaction (sentiment: {worst_month_score:.2f})"
                    )
        
        # Weather-related recommendations
        if all(col in df.columns for col in ['season', 'mentions_negative_weather']):
            season_neg_weather = df.groupby('season')['mentions_negative_weather'].mean()
            if not season_neg_weather.empty:
                worst_weather_season = season_neg_weather.idxmax()
                worst_weather_score = season_neg_weather.max()
                
                if worst_weather_score > 0.1:
                    recommendations["weather_related"].append(
                        f"Prepare visitors with realistic weather expectations for {worst_weather_season} "
                        f"when negative weather mentions peak ({worst_weather_score:.2f})"
                    )
                    
                    recommendations["product_development"].append(
                        f"Develop more indoor or weather-resilient activities for {worst_weather_season}"
                    )
        
        # Season-specific theme recommendations
        if 'primary_theme' in df.columns and 'season' in df.columns:
            try:
                # For each season, find the most common primary theme
                seasonal_theme_counts = df.groupby(['season', 'primary_theme']).size().reset_index(name='count')
                top_themes = seasonal_theme_counts.loc[seasonal_theme_counts.groupby('season')['count'].idxmax()]
                
                for _, row in top_themes.iterrows():
                    recommendations["product_development"].append(
                        f"Focus on {row['primary_theme']} experiences during {row['season']} "
                        f"when this theme is most relevant to visitors"
                    )
            except Exception as e:
                print(f"Error generating theme recommendations: {str(e)}")
        
        # Save recommendations to file
        with open(f'{self.output_dir}/seasonal_recommendations.json', 'w', encoding='utf-8') as f:
            json.dump(recommendations, f, indent=2)
        
        # Print key recommendations
        print("\nKey seasonal recommendations:")
        for category, recs in recommendations.items():
            if recs:
                print(f"\n{category.replace('_', ' ').title()}:")
                for rec in recs:
                    print(f"  - {rec}")
        
        return recommendations
    
    def run_analysis(self, df):
        """
        Run the complete seasonal pattern analysis.
        
        Parameters:
        - df: DataFrame with review data
        
        Returns:
        - DataFrame with seasonal analysis results
        """
        print("\n=== Running Seasonal Pattern Analysis ===")
        
        if df is None or len(df) == 0:
            print("No data available for seasonal analysis")
            return None
        
        # Extract date features
        df = self.extract_date_features(df)
        
        # Detect weather mentions
        df = self.detect_weather_mentions(df)
        
        # Analyze seasonal patterns
        df = self.analyze_seasonal_patterns(df)
        
        # Analyze monthly patterns
        df = self.analyze_monthly_patterns(df)
        
        # Analyze theme seasonality
        df = self.analyze_theme_seasonality(df)
        
        # Generate seasonal recommendations
        self.generate_seasonal_recommendations(df)
        
        print("\nSeasonal pattern analysis complete.")
        return df