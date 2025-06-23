import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json
import re

class TravelerSegmentAnalyzer:
    """
    Analyzes different traveler segments in tourism reviews.
    Focuses on identifying how different types of travelers (families, couples, solo, etc.)
    and visitors from different origins experience Tonga differently.
    """
    
    def __init__(self, output_dir='traveler_insights'):
        """
        Initialize the traveler segment analyzer.
        
        Parameters:
        - output_dir: Directory to save traveler segment analysis outputs
        """
        self.output_dir = output_dir
        
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created traveler segment insights directory: {output_dir}")
        
        # Define standard trip types for categorization
        self.trip_types = {
            'family': ['family', 'kid', 'child', 'parent', 'daughter', 'son', 'mom', 'dad'],
            'couple': ['couple', 'honeymoon', 'anniversary', 'romantic', 'husband', 'wife', 'partner'],
            'solo': ['solo', 'alone', 'myself', 'single'],
            'friends': ['friend', 'buddy', 'group', 'girls trip', 'guys trip', 'mates'],
            'business': ['business', 'work', 'conference', 'meeting', 'colleague']
        }
        
        # Define geographic regions for origin analysis
        self.regions = {
            'Oceania': ['australia', 'new zealand', 'fiji', 'samoa', 'papua new guinea', 'vanuatu'],
            'North America': ['usa', 'united states', 'canada', 'mexico', 'america', 'american'],
            'Europe': ['uk', 'united kingdom', 'england', 'france', 'germany', 'italy', 'spain', 
                      'netherlands', 'switzerland', 'norway', 'sweden', 'denmark'],
            'Asia': ['japan', 'china', 'korea', 'singapore', 'thailand', 'malaysia', 'indonesia',
                    'philippines', 'india', 'hong kong', 'taiwan'],
            'Other': ['south africa', 'brazil', 'argentina', 'chile', 'middle east', 'dubai',
                     'africa', 'south america']
        }
    
    def identify_trip_types(self, df):
      """
      Identify and standardize trip types from review data.
      
      Parameters:
      - df: DataFrame with review data
      
      Returns:
      - DataFrame with standardized trip types
      """
      print("Identifying traveler types from review data...")
      
      # Check if trip type information already exists
      if 'tripType' in df.columns:
          # Create a standardized trip type column
          df['trip_type_standard'] = df['tripType'].astype(str).str.lower()
          
          # Standardize various trip type mentions
          for trip_type, keywords in self.trip_types.items():
              # Create a pattern with escaped keywords
              escaped_keywords = [re.escape(keyword) for keyword in keywords]
              pattern = '|'.join(escaped_keywords)
              mask = df['trip_type_standard'].str.contains(pattern, case=False, na=False)
              df.loc[mask, 'trip_type_standard'] = trip_type
          
          # Count by trip type
          trip_type_counts = df['trip_type_standard'].value_counts()
          
          print("\nTraveler segments identified:")
          for trip_type, count in trip_type_counts.items():
              print(f"  {trip_type}: {count} reviews")
              
      # If no explicit trip type column, try to infer from text
      else:
          print("No explicit trip type column found, inferring from review text...")
          
          # Initialize the standardized trip type column
          df['trip_type_standard'] = 'unknown'
          
          # Try to infer trip type from review text
          for trip_type, keywords in self.trip_types.items():
              # Create a pattern with escaped keywords
              escaped_keywords = [re.escape(keyword) for keyword in keywords]
              pattern = r'\b(' + '|'.join(escaped_keywords) + r')\b'
              mask = df['text'].str.lower().str.contains(pattern, na=False)
              df.loc[mask, 'trip_type_standard'] = trip_type
          
          # Count by inferred trip type
          inferred_counts = df['trip_type_standard'].value_counts()
          
          print("\nInferred traveler segments:")
          for trip_type, count in inferred_counts.items():
              print(f"  {trip_type}: {count} reviews")
      
      return df
    
    def identify_visitor_origins(self, df):
      """
      Identify visitor origins from review data.
      
      Parameters:
      - df: DataFrame with review data
      
      Returns:
      - DataFrame with visitor origin information
      """
      print("\nIdentifying visitor origins...")
      
      # Check for user location information
      location_cols = [col for col in df.columns if 'location' in col.lower() and 'user' in col.lower()]
      location_col = None
      
      if location_cols:
          # Find the most appropriate location column
          for col in location_cols:
              if df[col].notna().sum() > 0:
                  location_col = col
                  break
      
      if location_col:
          print(f"Using location information from column: {location_col}")
          
          # Extract location information
          df['visitor_origin'] = df[location_col].astype(str).str.lower()
          
          # Clean up location text (remove unwanted parts)
          df['visitor_origin'] = df['visitor_origin'].str.replace('nan', '')
          
          # Categorize by region
          df['visitor_region'] = 'unknown'
          
          for region, countries in self.regions.items():
              # Create a pattern with escaped country names
              escaped_countries = [re.escape(country) for country in countries]
              pattern = '|'.join(escaped_countries)
              mask = df['visitor_origin'].str.contains(pattern, case=False, na=False)
              df.loc[mask, 'visitor_region'] = region
          
          # Count by region
          region_counts = df['visitor_region'].value_counts()
          
          print("\nVisitor regions identified:")
          for region, count in region_counts.items():
              print(f"  {region}: {count} reviews")
              
      else:
          print("No user location information found in the data")
          # Try to infer from text by looking for country mentions
          all_countries = [country for countries in self.regions.values() for country in countries]
          escaped_countries = [re.escape(country) for country in all_countries]
          country_pattern = r'\b(' + '|'.join(escaped_countries) + r')\b'
          
          # Check if review mentions "from [country]"
          from_pattern = r'from\s+([a-z\s]+)'
          
          df['inferred_origin'] = 'unknown'
          
          # Extract country mentions with "from" pattern
          for i, row in df.iterrows():
              if isinstance(row['text'], str):
                  text = row['text'].lower()
                  matches = re.findall(from_pattern, text)
                  for match in matches:
                      for region, countries in self.regions.items():
                          if any(country in match for country in countries):
                              df.at[i, 'inferred_origin'] = region
                              break
          
          # Count by inferred origin
          inferred_counts = df['inferred_origin'].value_counts()
          
          if len(inferred_counts) > 1:  # If we found some origins
              print("\nInferred visitor origins:")
              for origin, count in inferred_counts.items():
                  print(f"  {origin}: {count} reviews")
          else:
              print("Could not infer visitor origins from review text")
      
      return df
    
    def analyze_by_trip_type(self, df):
        """
        Analyze preferences and experiences by trip type.
        
        Parameters:
        - df: DataFrame with review data including trip types
        
        Returns:
        - DataFrame with trip type analysis
        """
        if 'trip_type_standard' not in df.columns or df['trip_type_standard'].nunique() <= 1:
            print("Not enough trip type data for analysis")
            return df
        
        print("\nAnalyzing preferences and experiences by trip type...")
        
        # Calculate average sentiment by trip type
        sentiment_by_trip = df.groupby('trip_type_standard')['sentiment_score'].mean().reset_index()
        sentiment_by_trip = sentiment_by_trip.sort_values('sentiment_score', ascending=False)
        
        print("Average sentiment by trip type:")
        print(sentiment_by_trip)
        
        # Create a bar chart of sentiment by trip type
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='trip_type_standard', y='sentiment_score', 
                        hue='trip_type_standard', data=sentiment_by_trip, legend=False)
        plt.title('Average Sentiment by Traveler Segment')
        plt.xlabel('Traveler Segment')
        plt.ylabel('Average Sentiment Score (-1 to 1)')
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        
        # Add sentiment value labels
        for i, row in enumerate(sentiment_by_trip.itertuples()):
            ax.text(i, row.sentiment_score + 0.01, f'{row.sentiment_score:.2f}', 
                  ha='center', va='bottom', color='black', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/sentiment_by_trip_type.png')
        print(f"Saved trip type sentiment chart to {self.output_dir}/sentiment_by_trip_type.png")
        
        # Analyze themes by trip type
        theme_cols = [col for col in df.columns if col.startswith('theme_')]
        
        if theme_cols:
            trip_themes = df.groupby('trip_type_standard')[theme_cols].mean()
            
            # Clean up theme names for display
            trip_themes.columns = [col.replace('theme_', '') for col in trip_themes.columns]
            
            print("\nTheme prevalence by trip type:")
            print(trip_themes)
            
            # Create a heatmap of theme prevalence by trip type
            plt.figure(figsize=(14, 8))
            sns.heatmap(trip_themes, annot=True, cmap='YlGnBu', fmt='.2f')
            plt.title('Theme Prevalence by Traveler Segment')
            plt.ylabel('Traveler Segment')
            plt.xlabel('Theme')
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/themes_by_trip_type.png')
            print(f"Saved theme prevalence chart to {self.output_dir}/themes_by_trip_type.png")
            
            # Identify top theme for each trip type
            top_themes = {}
            for trip_type in trip_themes.index:
                trip_data = trip_themes.loc[trip_type]
                top_theme = trip_data.idxmax()
                top_value = trip_data.max()
                top_themes[trip_type] = (top_theme, top_value)
            
            print("\nTop theme by traveler segment:")
            for trip_type, (theme, value) in top_themes.items():
                print(f"  {trip_type}: {theme} ({value:.2f})")
            
            # Save to file
            trip_themes.to_csv(f'{self.output_dir}/themes_by_trip_type.csv')
        
        # Analyze ratings by trip type if available
        if 'rating' in df.columns:
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
            rating_by_trip = df.groupby('trip_type_standard')['rating'].mean().reset_index()
            rating_by_trip = rating_by_trip.sort_values('rating', ascending=False)
            
            print("\nAverage rating by trip type:")
            print(rating_by_trip)
            
            # Create a bar chart of ratings by trip type
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x='trip_type_standard', y='rating', 
                            hue='trip_type_standard', data=rating_by_trip, legend=False)
            plt.title('Average Rating by Traveler Segment')
            plt.xlabel('Traveler Segment')
            plt.ylabel('Average Rating (1-5)')
            plt.ylim(1, 5)
            
            # Add rating value labels
            for i, row in enumerate(rating_by_trip.itertuples()):
                ax.text(i, row.rating - 0.2, f'{row.rating:.2f}', 
                      ha='center', va='bottom', color='white', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/rating_by_trip_type.png')
            print(f"Saved trip type rating chart to {self.output_dir}/rating_by_trip_type.png")
        
        return df
    
    def analyze_by_visitor_origin(self, df):
        """
        Analyze preferences and experiences by visitor origin.
        
        Parameters:
        - df: DataFrame with review data including visitor origins
        
        Returns:
        - DataFrame with visitor origin analysis
        """
        if (('visitor_region' not in df.columns or df['visitor_region'].nunique() <= 1) and
            ('inferred_origin' not in df.columns or df['inferred_origin'].nunique() <= 1)):
            print("Not enough visitor origin data for analysis")
            return df
        
        print("\nAnalyzing preferences and experiences by visitor origin...")
        
        # Determine which origin column to use
        origin_col = 'visitor_region' if 'visitor_region' in df.columns else 'inferred_origin'
        
        # Calculate average sentiment by origin
        sentiment_by_origin = df.groupby(origin_col)['sentiment_score'].mean().reset_index()
        sentiment_by_origin = sentiment_by_origin.sort_values('sentiment_score', ascending=False)
        
        # Filter out 'unknown' if present
        if 'unknown' in sentiment_by_origin[origin_col].values:
            sentiment_by_origin = sentiment_by_origin[sentiment_by_origin[origin_col] != 'unknown']
        
        if len(sentiment_by_origin) <= 1:
            print("Not enough visitor origin data for visualization")
        else:
            print("Average sentiment by visitor origin:")
            print(sentiment_by_origin)
            
            # Create a bar chart of sentiment by origin
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x=origin_col, y='sentiment_score', 
                            hue=origin_col, data=sentiment_by_origin, legend=False)
            plt.title('Average Sentiment by Visitor Origin')
            plt.xlabel('Visitor Origin')
            plt.ylabel('Average Sentiment Score (-1 to 1)')
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            
            # Add sentiment value labels
            for i, row in enumerate(sentiment_by_origin.itertuples()):
                ax.text(i, row.sentiment_score + 0.01, f'{row.sentiment_score:.2f}', 
                      ha='center', va='bottom', color='black', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/sentiment_by_origin.png')
            print(f"Saved visitor origin sentiment chart to {self.output_dir}/sentiment_by_origin.png")
            
            # Analyze themes by visitor origin
            theme_cols = [col for col in df.columns if col.startswith('theme_')]
            
            if theme_cols:
                origin_themes = df.groupby(origin_col)[theme_cols].mean()
                
                # Filter out 'unknown' if present
                if 'unknown' in origin_themes.index:
                    origin_themes = origin_themes.drop('unknown')
                
                if len(origin_themes) <= 1:
                    print("Not enough visitor origin data for theme analysis")
                else:
                    # Clean up theme names for display
                    origin_themes.columns = [col.replace('theme_', '') for col in origin_themes.columns]
                    
                    print("\nTheme prevalence by visitor origin:")
                    print(origin_themes)
                    
                    # Create a heatmap of theme prevalence by origin
                    plt.figure(figsize=(14, 8))
                    sns.heatmap(origin_themes, annot=True, cmap='YlGnBu', fmt='.2f')
                    plt.title('Theme Prevalence by Visitor Origin')
                    plt.ylabel('Visitor Origin')
                    plt.xlabel('Theme')
                    plt.tight_layout()
                    plt.savefig(f'{self.output_dir}/themes_by_origin.png')
                    print(f"Saved theme prevalence chart to {self.output_dir}/themes_by_origin.png")
                    
                    # Identify top theme for each origin
                    top_themes = {}
                    for origin in origin_themes.index:
                        origin_data = origin_themes.loc[origin]
                        top_theme = origin_data.idxmax()
                        top_value = origin_data.max()
                        top_themes[origin] = (top_theme, top_value)
                    
                    print("\nTop theme by visitor origin:")
                    for origin, (theme, value) in top_themes.items():
                        print(f"  {origin}: {theme} ({value:.2f})")
                    
                    # Save to file
                    origin_themes.to_csv(f'{self.output_dir}/themes_by_origin.csv')
        
        return df
    
    def analyze_cross_segment(self, df):
        """
        Analyze the combination of trip type and visitor origin.
        
        Parameters:
        - df: DataFrame with review data including trip types and origins
        
        Returns:
        - DataFrame with cross-segment analysis
        """
        if ('trip_type_standard' not in df.columns or 
            (('visitor_region' not in df.columns or df['visitor_region'].nunique() <= 1) and
             ('inferred_origin' not in df.columns or df['inferred_origin'].nunique() <= 1))):
            print("Not enough data for cross-segment analysis")
            return df
        
        print("\nAnalyzing cross-segments (trip type × visitor origin)...")
        
        # Determine which origin column to use
        origin_col = 'visitor_region' if 'visitor_region' in df.columns else 'inferred_origin'
        
        # Get valid trip types and origins with enough data
        trip_counts = df['trip_type_standard'].value_counts()
        valid_trips = trip_counts[trip_counts >= 5].index
        
        origin_counts = df[origin_col].value_counts()
        valid_origins = origin_counts[origin_counts >= 5].index
        
        # Filter to valid segments with enough data
        cross_df = df[df['trip_type_standard'].isin(valid_trips) & 
                     df[origin_col].isin(valid_origins)]
        
        if len(cross_df) < 10:
            print("Not enough data for meaningful cross-segment analysis")
            return df
        
        # Calculate average sentiment by trip type and origin
        cross_sentiment = cross_df.groupby(['trip_type_standard', origin_col])['sentiment_score'].mean().reset_index()
        cross_sentiment = cross_sentiment.sort_values('sentiment_score', ascending=False)
        
        print("Top cross-segments by sentiment:")
        for i, row in cross_sentiment.head(5).iterrows():
            print(f"  {row['trip_type_standard']} travelers from {row[origin_col]}: {row['sentiment_score']:.2f}")
        
        # Create a heatmap of sentiment by trip type and origin
        try:
            pivot_sentiment = cross_sentiment.pivot(index='trip_type_standard', 
                                                 columns=origin_col, 
                                                 values='sentiment_score')
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot_sentiment, annot=True, cmap='RdYlGn', center=0, fmt='.2f')
            plt.title('Sentiment by Traveler Segment and Origin')
            plt.ylabel('Traveler Segment')
            plt.xlabel('Visitor Origin')
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/cross_segment_sentiment.png')
            print(f"Saved cross-segment sentiment heatmap to {self.output_dir}/cross_segment_sentiment.png")
            
            # Save to file
            pivot_sentiment.to_csv(f'{self.output_dir}/cross_segment_sentiment.csv')
        except Exception as e:
            print(f"Could not create cross-segment visualization: {str(e)}")
        
        return df
    
    def generate_segment_recommendations(self, df):
        """
        Generate traveler segment-specific recommendations.
        
        Parameters:
        - df: DataFrame with traveler segment analysis
        
        Returns:
        - Dictionary with traveler segment recommendations
        """
        print("\nGenerating traveler segment-specific recommendations...")
        
        recommendations = {
            "marketing_targeting": [],
            "product_development": [],
            "service_improvements": [],
            "segment_specific": {}
        }
        
        # Recommendations based on trip type sentiment
        if 'trip_type_standard' in df.columns:
            trip_sentiment = df.groupby('trip_type_standard')['sentiment_score'].mean()
            
            # Filter out 'unknown' if present
            if 'unknown' in trip_sentiment.index:
                trip_sentiment = trip_sentiment.drop('unknown')
            
            if not trip_sentiment.empty:
                # Identify highest and lowest sentiment segments
                best_segment = trip_sentiment.idxmax()
                worst_segment = trip_sentiment.idxmin()
                best_score = trip_sentiment.max()
                worst_score = trip_sentiment.min()
                
                if best_score > 0.1:
                    recommendations["marketing_targeting"].append(
                        f"Target {best_segment} travelers in marketing campaigns as they report "
                        f"the most positive experiences (sentiment: {best_score:.2f})"
                    )
                
                if worst_score < 0:
                    recommendations["service_improvements"].append(
                        f"Improve services for {worst_segment} travelers who report "
                        f"negative experiences (sentiment: {worst_score:.2f})"
                    )
                
                # Generate segment-specific recommendations
                for segment, sentiment in trip_sentiment.items():
                    if segment != 'unknown':
                        segment_recs = []
                        
                        # Theme-based recommendations for this segment
                        theme_cols = [col for col in df.columns if col.startswith('theme_')]
                        if theme_cols:
                            segment_df = df[df['trip_type_standard'] == segment]
                            theme_means = segment_df[theme_cols].mean()
                            top_theme_col = theme_means.idxmax()
                            top_theme = top_theme_col.replace('theme_', '')
                            top_score = theme_means.max()
                            
                            if top_score > 0.1:
                                segment_recs.append(
                                    f"Focus on {top_theme.replace('_', ' ')} experiences "
                                    f"which are particularly important to this segment (score: {top_score:.2f})"
                                )
                        
                        # Sentiment-based recommendations
                        if sentiment > 0.2:  # Very positive
                            segment_recs.append(
                                f"Leverage highly positive sentiment in testimonials and marketing"
                            )
                        elif sentiment < 0:  # Negative
                            segment_recs.append(
                                f"Address negative experiences to improve satisfaction"
                            )
                        
                        if segment_recs:
                            recommendations["segment_specific"][segment] = segment_recs
        
        # Recommendations based on visitor origin
        origin_col = None
        if 'visitor_region' in df.columns:
            origin_col = 'visitor_region'
        elif 'inferred_origin' in df.columns:
            origin_col = 'inferred_origin'
        
        if origin_col:
            origin_sentiment = df.groupby(origin_col)['sentiment_score'].mean()
            
            # Filter out 'unknown' if present
            if 'unknown' in origin_sentiment.index:
                origin_sentiment = origin_sentiment.drop('unknown')
            
            if not origin_sentiment.empty:
                # Identify highest and lowest sentiment origins
                best_origin = origin_sentiment.idxmax()
                worst_origin = origin_sentiment.idxmin()
                best_score = origin_sentiment.max()
                worst_score = origin_sentiment.min()
                
                if best_score > 0.1:
                    recommendations["marketing_targeting"].append(
                        f"Focus marketing efforts in {best_origin} where visitors report "
                        f"the most positive experiences (sentiment: {best_score:.2f})"
                    )
                
                if worst_score < 0:
                    recommendations["service_improvements"].append(
                        f"Address cultural differences and expectations for visitors from {worst_origin} "
                        f"who report negative experiences (sentiment: {worst_score:.2f})"
                    )
                
                # Theme preferences by origin
                theme_cols = [col for col in df.columns if col.startswith('theme_')]
                if theme_cols:
                    for origin in origin_sentiment.index:
                        if origin != 'unknown':
                            origin_df = df[df[origin_col] == origin]
                            theme_means = origin_df[theme_cols].mean()
                            top_theme_col = theme_means.idxmax()
                            top_theme = top_theme_col.replace('theme_', '')
                            
                            recommendations["product_development"].append(
                                f"For visitors from {origin}, emphasize {top_theme.replace('_', ' ')} "
                                f"experiences in marketing and product offerings"
                            )
        
        # Cross-segment recommendations
        if ('trip_type_standard' in df.columns and origin_col):
            try:
                cross_sentiment = df.groupby(['trip_type_standard', origin_col])['sentiment_score'].mean()
                
                # Get top and bottom cross-segments
                if not cross_sentiment.empty:
                    top_cross = cross_sentiment.idxmax()
                    bottom_cross = cross_sentiment.idxmin()
                    
                    # Only recommend if we have valid data
                    if top_cross[0] != 'unknown' and top_cross[1] != 'unknown':
                        recommendations["marketing_targeting"].append(
                            f"Develop targeted campaigns for {top_cross[0]} travelers from {top_cross[1]} "
                            f"who report the most positive experiences"
                        )
                        
                    if bottom_cross[0] != 'unknown' and bottom_cross[1] != 'unknown':
                        recommendations["service_improvements"].append(
                            f"Address specific needs of {bottom_cross[0]} travelers from {bottom_cross[1]} "
                            f"who report the least positive experiences"
                        )
            except Exception as e:
                print(f"Error generating cross-segment recommendations: {str(e)}")
        
        # Save recommendations to file
        with open(f'{self.output_dir}/traveler_segment_recommendations.json', 'w', encoding='utf-8') as f:
            json.dump(recommendations, f, indent=2)
        
        # Print key recommendations
        print("\nKey traveler segment recommendations:")
        for category, recs in recommendations.items():
            if category != "segment_specific" and recs:
                print(f"\n{category.replace('_', ' ').title()}:")
                for rec in recs:
                    print(f"  - {rec}")
        
        print("\nSegment-specific recommendations:")
        for segment, recs in recommendations["segment_specific"].items():
            print(f"\n  {segment}:")
            for rec in recs:
                print(f"    - {rec}")
        
        return recommendations
    
    def run_analysis(self, df):
        """
        Run the complete traveler segment analysis.
        
        Parameters:
        - df: DataFrame with review data
        
        Returns:
        - DataFrame with traveler segment analysis results
        """
        print("\n=== Running Traveler Segment Analysis ===")
        
        if df is None or len(df) == 0:
            print("No data available for traveler segment analysis")
            return None
        
        # Identify trip types
        df = self.identify_trip_types(df)
        
        # Identify visitor origins
        df = self.identify_visitor_origins(df)
        
        # Analyze by trip type
        df = self.analyze_by_trip_type(df)
        
        # Analyze by visitor origin
        df = self.analyze_by_visitor_origin(df)
        
        # Analyze cross-segments
        df = self.analyze_cross_segment(df)
        
        # Generate traveler segment recommendations
        self.generate_segment_recommendations(df)
        
        print("\nTraveler segment analysis complete.")
        return df