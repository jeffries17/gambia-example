import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import re
from collections import Counter
import numpy as np

class TravelerSegmentAnalyzer:
    """
    Analyzes different traveler segments in tourism reviews.
    Focuses on identifying how different types of travelers (families, couples, solo, etc.)
    and visitors from different origins experience tourism destinations.
    """
    
    def __init__(self, reviews_df, output_dir='outputs/traveler_segments'):
        """
        Initialize the traveler segment analyzer with the dataframe and output directory.
        
        Parameters:
        - reviews_df: DataFrame containing the review data
        - output_dir: Directory to save analysis outputs
        """
        self.reviews_df = reviews_df
        self.output_dir = output_dir
        
        # Ensure the output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")
            
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

    def preprocess_data(self):
        """
        Preprocess data to extract necessary fields for analysis.
        Identifies trip types and visitor origins from the data.
        """
        print("Preprocessing data...")
        
        # Process user location information
        self._process_user_location()
        
        # Process trip type information
        self._process_trip_types()
        
        # Standardize columns
        if 'rating' in self.reviews_df.columns:
            self.reviews_df['rating'] = pd.to_numeric(self.reviews_df['rating'], errors='coerce')
        
        # Add basic sentiment score if not present
        if 'sentiment_score' not in self.reviews_df.columns and 'text' in self.reviews_df.columns:
            print("Note: 'sentiment_score' column not found. Adding a basic sentiment score derived from ratings.")
            
            # Create a simple sentiment mapping from ratings
            # This is a basic approximation - the sentiment analyzer would do a much better job
            if 'rating' in self.reviews_df.columns:
                # Map 1-5 ratings to -1 to 1 sentiment scale
                self.reviews_df['sentiment_score'] = (self.reviews_df['rating'] - 3) / 2
                self.reviews_df['sentiment_category'] = pd.cut(
                    self.reviews_df['sentiment_score'],
                    bins=[-1, -0.1, 0.1, 1],
                    labels=['negative', 'neutral', 'positive']
                )
            else:
                print("Neither sentiment_score nor rating columns available. Sentiment analysis will be limited.")
        
        print(f"Preprocessing complete. Identified {self.reviews_df['trip_type'].nunique()} trip types and {self.reviews_df['country'].nunique()} countries.")
        
        return self.reviews_df

    def _process_user_location(self):
        """Extract and standardize user location information."""
        # Check if user_location column already exists
        if 'user_location' in self.reviews_df.columns:
            # Use existing user_location column
            self.reviews_df['raw_location'] = self.reviews_df['user_location'].astype(str)
        else:
            # Try to extract user location from nested structure
            try:
                # First attempt - check if we have the raw user object
                if 'user' in self.reviews_df.columns and self.reviews_df['user'].notna().any():
                    def extract_location(user_obj):
                        if isinstance(user_obj, dict) and 'userLocation' in user_obj:
                            loc = user_obj['userLocation']
                            if isinstance(loc, dict) and 'shortName' in loc:
                                return loc['shortName']
                            elif isinstance(loc, dict) and 'name' in loc:
                                return loc['name']
                            elif loc is not None:
                                return str(loc)
                        return 'Unknown'
                    
                    self.reviews_df['raw_location'] = self.reviews_df['user'].apply(extract_location)
                else:
                    # If user column doesn't exist, try inferring from text
                    self.reviews_df['raw_location'] = self._infer_country_from_text()
            except Exception as e:
                print(f"Warning: Could not extract user location: {e}")
                # Default to Unknown
                self.reviews_df['raw_location'] = 'Unknown'
        
        # Clean up raw location data
        self.reviews_df['raw_location'] = self.reviews_df['raw_location'].astype(str)
        self.reviews_df['raw_location'] = self.reviews_df['raw_location'].apply(
            lambda x: x if x != 'nan' and x != 'None' and x.lower() != 'unknown' and x.lower() != 'null' else 'Unknown'
        )
        
        # Extract country from city/region format (e.g., "Sydney, Australia" → "Australia")
        self.reviews_df['country'] = self.reviews_df['raw_location'].apply(self._extract_country_from_location)
        
        print(f"Extracted country data: {self.reviews_df['country'].value_counts().head(5)}")
        
        # Map countries to regions
        self.reviews_df['region'] = 'Unknown'
        for region, countries in self.regions.items():
            pattern = '|'.join(countries)
            mask = self.reviews_df['country'].str.lower().str.contains(pattern, na=False)
            self.reviews_df.loc[mask, 'region'] = region
    
    def _extract_country_from_location(self, location_string):
        """
        Extract the country name from a location string like "City, Country" or "City, Region, Country"
        """
        if location_string is None or location_string == 'Unknown':
            return 'Unknown'
            
        # Common country-level domains that might appear in location strings
        common_countries = {
            'australia': 'Australia', 
            'new zealand': 'New Zealand',
            'united states': 'United States', 
            'usa': 'United States',
            'united kingdom': 'United Kingdom',
            'uk': 'United Kingdom',
            'canada': 'Canada',
            'japan': 'Japan',
            'germany': 'Germany',
            'france': 'France',
            'italy': 'Italy',
            'spain': 'Spain',
            'china': 'China',
            'norway': 'Norway',
            'sweden': 'Sweden',
            'denmark': 'Denmark',
            'finland': 'Finland',
            'netherlands': 'Netherlands',
            'singapore': 'Singapore',
            'south africa': 'South Africa',
            'brazil': 'Brazil',
            'mexico': 'Mexico',
            'thailand': 'Thailand',
            'switzerland': 'Switzerland',
            'russia': 'Russia',
            'india': 'India',
            'tonga': 'Tonga',
            'fiji': 'Fiji',
            'samoa': 'Samoa',
            'vanuatu': 'Vanuatu'
        }
        
        # Regex patterns to identify countries
        # Format: "City, Country" or "City, Region, Country"
        location_string = location_string.lower()
        
        # Try to match country directly
        for country_name, standardized_name in common_countries.items():
            if country_name in location_string:
                return standardized_name
        
        # Check if location is in format "City, Country"
        parts = [part.strip() for part in location_string.split(',')]
        if len(parts) > 1:
            # Check if the last part is a known country
            last_part = parts[-1].strip()
            for country_name, standardized_name in common_countries.items():
                if country_name in last_part:
                    return standardized_name
        
        # Return the original string if we couldn't extract a country
        return location_string

    def _process_trip_types(self):
        """Extract and standardize trip type information."""
        # Check if trip type information already exists
        if 'tripType' in self.reviews_df.columns:
            # Create a standardized trip type column
            self.reviews_df['trip_type'] = self.reviews_df['tripType'].astype(str).str.lower()
            
            # Convert COUPLES, FAMILY, etc. to our standard categories
            # Direct mapping for common TripAdvisor trip types
            trip_type_mapping = {
                'couples': 'couple',
                'family': 'family',
                'solo': 'solo',
                'business': 'business',
                'friends': 'friends'
            }
            
            # Apply direct mapping first
            for old, new in trip_type_mapping.items():
                mask = self.reviews_df['trip_type'].str.contains(old, case=False, na=False)
                self.reviews_df.loc[mask, 'trip_type'] = new
                
        else:
            # Initialize with 'other'
            self.reviews_df['trip_type'] = 'other'
        
        # Apply keyword-based classification for anything not caught by direct mapping
        # or for text-based inference
        for trip_category, keywords in self.trip_types.items():
            pattern = '|'.join(keywords)
            
            # Only apply keyword matching to 'other' trip types
            trip_type_mask = (self.reviews_df['trip_type'] == 'other') | (self.reviews_df['trip_type'] == 'nan')
            
            # Check in text column if available
            if 'text' in self.reviews_df.columns:
                text_contains = self.reviews_df['text'].astype(str).str.lower().str.contains(pattern, na=False)
                # Apply to rows that have matching text and haven't been categorized
                self.reviews_df.loc[trip_type_mask & text_contains, 'trip_type'] = trip_category
        
        # Fill any remaining NAs
        self.reviews_df['trip_type'] = self.reviews_df['trip_type'].fillna('other')
        
        # Make sure 'nan' values become 'other'
        self.reviews_df.loc[self.reviews_df['trip_type'] == 'nan', 'trip_type'] = 'other'
        
        # Print trip type distribution for verification
        print("Trip type distribution:")
        print(self.reviews_df['trip_type'].value_counts())

    def _infer_country_from_text(self):
        """Attempt to infer country from review text."""
        if 'text' not in self.reviews_df.columns:
            return pd.Series(['Unknown'] * len(self.reviews_df))
        
        # Combine all country keywords
        all_countries = [country for countries in self.regions.values() for country in countries]
        country_pattern = r'\b(' + '|'.join(all_countries) + r')\b'
        
        # Check for "from [country]" pattern
        from_pattern = r'from\s+([a-z\s]+)'
        
        inferred_countries = []
        
        for text in self.reviews_df['text'].astype(str):
            text = text.lower()
            country_found = False
            
            # Check for "from [country]" pattern
            matches = re.findall(from_pattern, text)
            for match in matches:
                for region, countries in self.regions.items():
                    if any(country in match for country in countries):
                        inferred_countries.append(next((country for country in countries if country in match), region))
                        country_found = True
                        break
                if country_found:
                    break
            
            # If no country found in "from" pattern, check for any country mention
            if not country_found:
                for region, countries in self.regions.items():
                    for country in countries:
                        if re.search(r'\b' + country + r'\b', text):
                            inferred_countries.append(country)
                            country_found = True
                            break
                    if country_found:
                        break
            
            # If still no country found
            if not country_found:
                inferred_countries.append('Unknown')
        
        return pd.Series(inferred_countries)

    def analyze_by_segment(self):
        """
        Analyze the data by different traveler segments and countries.
        
        Returns:
        - tuple: (segment_analysis, origin_analysis) DataFrames
        """
        print("Analyzing traveler segments and visitor origins...")
        
        # Segment analysis by trip type
        segment_analysis = self._analyze_segment_type('trip_type')
        
        # Segment analysis by country
        origin_analysis = self._analyze_segment_type('country')
        
        # Focus on top 5 countries (excluding 'Unknown')
        top_countries = origin_analysis.copy()
        if 'Unknown' in top_countries.index:
            top_countries = top_countries.drop('Unknown')
        top_5_countries = top_countries.head(5)
        
        # Save detailed top 5 countries analysis
        if len(top_5_countries) > 0:
            top_5_countries.to_csv(os.path.join(self.output_dir, 'top_5_countries_analysis.csv'))
            print(f"Top 5 countries of origin: {', '.join(top_5_countries.index.tolist())}")
            
            # Create detailed analysis for top 5 countries
            self._analyze_top_countries(top_5_countries.index)
        
        # Save analysis results
        segment_analysis.to_csv(os.path.join(self.output_dir, 'segment_analysis.csv'))
        origin_analysis.to_csv(os.path.join(self.output_dir, 'origin_analysis.csv'))
        
        # Generate cross-segment analysis
        self._analyze_cross_segments('trip_type', 'country')
        
        return segment_analysis, origin_analysis
        
    def _analyze_top_countries(self, top_countries):
        """
        Create detailed analysis for the top countries of origin.
        
        Parameters:
        - top_countries: List of top country names
        """
        if len(top_countries) == 0:
            return
            
        # Create a subdirectory for detailed country analysis
        country_dir = os.path.join(self.output_dir, 'country_analysis')
        if not os.path.exists(country_dir):
            os.makedirs(country_dir)
            
        # Overall comparison visualization
        self._create_top_countries_comparison(top_countries, country_dir)
        
        # For each top country, analyze trip types and sentiment
        results = {}
        
        for country in top_countries:
            # Filter data for this country
            country_data = self.reviews_df[self.reviews_df['country'] == country]
            
            if len(country_data) < 5:  # Minimum threshold for meaningful analysis
                continue
                
            # Create country subdirectory
            country_subdir = os.path.join(country_dir, country.replace(' ', '_').lower())
            if not os.path.exists(country_subdir):
                os.makedirs(country_subdir)
            
            # Analyze trip types for this country
            trip_stats = country_data.groupby('trip_type').agg({
                'rating': ['mean', 'count', 'std']
            })
            
            # Flatten MultiIndex
            flat_columns = []
            for col in trip_stats.columns:
                outer, inner = col
                flat_columns.append(f"{outer}_{inner}")
            trip_stats.columns = flat_columns
            
            # Filter to trip types with enough data
            trip_stats = trip_stats[trip_stats['rating_count'] >= 3]
            
            # Add sentiment analysis if available
            if 'sentiment_score' in country_data.columns:
                sentiment_stats = country_data.groupby('trip_type')['sentiment_score'].agg(['mean', 'std', 'count'])
                sentiment_stats.columns = ['sentiment_mean', 'sentiment_std', 'sentiment_count']
                trip_stats = trip_stats.join(sentiment_stats, how='left')
            
            # Save country-specific trip type analysis
            trip_stats.to_csv(os.path.join(country_subdir, f'{country}_trip_types.csv'))
            
            # Create trip type visualization for this country
            self._create_country_trip_type_chart(trip_stats, country, country_subdir)
            
            # Get most common and highest-rated trip types
            if len(trip_stats) > 0:
                most_common = trip_stats['rating_count'].idxmax() if len(trip_stats) > 0 else None
                highest_rated = trip_stats['rating_mean'].idxmax() if len(trip_stats) > 0 else None
                
                countries_result = {
                    'review_count': len(country_data),
                    'avg_rating': country_data['rating'].mean(),
                    'most_common_trip': most_common,
                    'highest_rated_trip': highest_rated
                }
                
                if 'sentiment_score' in country_data.columns:
                    countries_result['avg_sentiment'] = country_data['sentiment_score'].mean()
                
                results[country] = countries_result
        
        # Save consolidated results
        with open(os.path.join(country_dir, 'top_countries_analysis.json'), 'w') as f:
            json.dump(results, f, indent=4)
            
    def _create_top_countries_comparison(self, top_countries, output_dir):
        """
        Create comparative visualizations for top countries.
        
        Parameters:
        - top_countries: List of top country names
        - output_dir: Output directory for visualizations
        """
        # Filter to these countries
        countries_df = self.reviews_df[self.reviews_df['country'].isin(top_countries)]
        
        if len(countries_df) < 5:
            return
            
        # Create directory for visualizations
        viz_dir = os.path.join(output_dir, 'visualizations')
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
            
        # Use a consistent color mapping for better visualizations
        country_colors = {
            'New Zealand': '#1E88E5',   # Blue
            'Australia': '#FFC107',     # Amber
            'United States': '#D81B60', # Pink
            'United Kingdom': '#004D40', # Teal
            'Tonga': '#43A047'          # Green
        }
        
        # Create a color list in the same order as the countries for consistency
        colors = [country_colors.get(country, '#BDBDBD') for country in top_countries]
        
        # 1. Rating distribution by country
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='country', y='rating', data=countries_df, palette=colors)
        plt.title('Rating Distribution by Country of Origin', fontsize=14, pad=20)
        plt.xlabel('Country', fontsize=12)
        plt.ylabel('Rating', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'rating_distribution_by_country.png'), dpi=300)
        plt.close()
        
        # 2. Trip type distribution by country
        # Get counts of each trip type per country
        trip_country_counts = pd.crosstab(countries_df['country'], countries_df['trip_type'])
        
        # Convert to percentages
        trip_country_pct = trip_country_counts.div(trip_country_counts.sum(axis=1), axis=0) * 100
        
        # Create stacked bar chart
        plt.figure(figsize=(12, 6))
        
        # Match the order of countries in the palette
        trip_countries = trip_country_pct.index.tolist()
        chart_colors = [country_colors.get(country, '#BDBDBD') for country in trip_countries]
        
        ax = trip_country_pct.plot(kind='bar', stacked=True, colormap='viridis', figsize=(12, 6))
        
        # Color the bars by country (different than the trip type default coloring)
        for i, country in enumerate(trip_countries):
            for container in ax.containers:
                if i < len(container):
                    container[i].set_alpha(0.8)  # Adjust transparency
        
        plt.title('Trip Type Distribution by Country of Origin', fontsize=14, pad=20)
        plt.xlabel('Country', fontsize=12)
        plt.ylabel('Percentage', fontsize=12)
        plt.legend(title='Trip Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'trip_type_by_country.png'), dpi=300)
        plt.close()
        
        # 3. Sentiment comparison if available
        if 'sentiment_score' in countries_df.columns:
            plt.figure(figsize=(12, 6))
            sns.boxplot(x='country', y='sentiment_score', data=countries_df, palette=colors)
            plt.title('Sentiment Distribution by Country of Origin', fontsize=14, pad=20)
            plt.xlabel('Country', fontsize=12)
            plt.ylabel('Sentiment Score', fontsize=12)
            plt.grid(axis='y', alpha=0.3)
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'sentiment_by_country.png'), dpi=300)
            plt.close()
            
        # 4. Heatmap of trip type satisfaction by country
        if len(countries_df) >= 10:
            try:
                # Create pivot table
                pivot_df = pd.pivot_table(
                    countries_df, 
                    values='rating', 
                    index='country',
                    columns='trip_type',
                    aggfunc='mean'
                )
                
                # Create heatmap
                plt.figure(figsize=(12, 8))
                sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt='.2f')
                plt.title('Average Rating by Country and Trip Type', fontsize=14, pad=20)
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, 'country_trip_type_heatmap.png'), dpi=300)
                plt.close()
            except:
                print("Could not create country/trip type heatmap - not enough data")
                
    def _create_country_trip_type_chart(self, trip_stats, country, output_dir):
        """
        Create chart visualizing trip types for a specific country.
        
        Parameters:
        - trip_stats: DataFrame with trip type statistics for this country
        - country: Country name
        - output_dir: Output directory for visualizations
        """
        if len(trip_stats) == 0:
            return
            
        # Create visualizations directory
        viz_dir = os.path.join(output_dir, 'visualizations')
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
            
        # Create combined chart
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Format country name for title
        country_title = country.title() if country != country.upper() else country
        
        # Primary axis for average rating
        color = 'tab:blue'
        ax1.set_xlabel('Trip Type')
        ax1.set_ylabel('Average Rating', color=color)
        bars = ax1.bar(trip_stats.index, trip_stats['rating_mean'], color=color, alpha=0.7)
        
        # Add rating labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width()/2., 
                height + 0.1, 
                f'{height:.2f}', 
                ha='center', va='bottom', 
                color=color, fontweight='bold'
            )
        
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(0, 5.5)  # Assuming 5-star rating scale
        plt.xticks(rotation=45, ha='right')
        
        # Secondary axis for review count
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Number of Reviews', color=color)
        line = ax2.plot(trip_stats.index, trip_stats['rating_count'], marker='o', color=color, linewidth=2)
        
        # Add count labels
        for i, val in enumerate(trip_stats['rating_count']):
            ax2.text(
                i, 
                val + (0.03 * trip_stats['rating_count'].max()), 
                f'{val:.0f}', 
                ha='center', va='bottom', 
                color=color, fontweight='bold'
            )
            
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title(f'Trip Types for Visitors from {country_title}', fontsize=14, pad=20)
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f'{country.lower()}_trip_types.png'), dpi=300)
        plt.close()

    def _analyze_segment_type(self, segment_col):
        """Analyze a specific segment type (trip_type or country)."""
        # Basic statistics
        segment_stats = self.reviews_df.groupby(segment_col).agg({
            'rating': ['mean', 'median', 'count', 'std'],
        })
        
        # Add sentiment statistics if available
        if 'sentiment_score' in self.reviews_df.columns:
            sentiment_stats = self.reviews_df.groupby(segment_col)['sentiment_score'].agg(['mean', 'median', 'std'])
            # Create MultiIndex columns for sentiment stats
            sentiment_cols = pd.MultiIndex.from_product([['sentiment_score'], ['mean', 'median', 'std']])
            sentiment_df = pd.DataFrame(
                sentiment_stats.values, 
                index=sentiment_stats.index, 
                columns=sentiment_cols
            )
            segment_stats = pd.concat([segment_stats, sentiment_df], axis=1)
        
        # Flatten the MultiIndex columns safely
        if isinstance(segment_stats.columns, pd.MultiIndex):
            flat_columns = []
            for col in segment_stats.columns:
                if len(col) == 2:  # Standard case with (outer, inner)
                    outer, inner = col
                    flat_columns.append(f"{outer}_{inner}" if inner else outer)
                else:  # Handle other cases
                    flat_columns.append("_".join(str(c) for c in col if c))
            segment_stats.columns = flat_columns
        
        # Filter out segments with very few reviews
        min_reviews = 3
        count_col = 'rating_count' if 'rating_count' in segment_stats.columns else segment_stats.columns[0]
        segment_stats = segment_stats[segment_stats[count_col] >= min_reviews]
        
        # Sort by count for relevance
        segment_stats = segment_stats.sort_values(count_col, ascending=False)
        
        return segment_stats

    def _analyze_cross_segments(self, segment_col1, segment_col2):
        """Analyze interaction between two segment types."""
        if len(self.reviews_df) < 20:  # Minimum threshold for meaningful cross-analysis
            print("Not enough data for cross-segment analysis")
            return
        
        # Get top segments with enough data
        top_segments1 = self.reviews_df[segment_col1].value_counts().head(5).index
        top_segments2 = self.reviews_df[segment_col2].value_counts().head(5).index
        
        # Filter to these segments
        filtered_df = self.reviews_df[self.reviews_df[segment_col1].isin(top_segments1) & 
                                     self.reviews_df[segment_col2].isin(top_segments2)]
        
        # If we have rating data
        if 'rating' in filtered_df.columns and len(filtered_df) >= 10:
            # Create pivot table
            try:
                cross_ratings = pd.pivot_table(
                    filtered_df, 
                    values='rating', 
                    index=segment_col1,
                    columns=segment_col2,
                    aggfunc='mean'
                )
                
                # Save to CSV
                cross_ratings.to_csv(os.path.join(self.output_dir, f'cross_segment_{segment_col1}_{segment_col2}.csv'))
                
                # Create heatmap
                plt.figure(figsize=(12, 8))
                sns.heatmap(cross_ratings, annot=True, cmap='YlGnBu', fmt='.2f')
                plt.title(f'Average Rating by {segment_col1.title()} and {segment_col2.title()}')
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f'cross_segment_heatmap.png'))
                plt.close()
            except:
                print(f"Could not create cross-segment analysis between {segment_col1} and {segment_col2}")

    def visualize_by_segment(self, stats_df, segment_col):
        """
        Generate visualizations for the analyzed data.
        
        Parameters:
        - stats_df: DataFrame with segment statistics
        - segment_col: Column name for segmentation (trip_type or country)
        """
        print(f"Generating visualizations for {segment_col} segments...")
        
        # Create visualizations directory if needed
        viz_dir = os.path.join(self.output_dir, 'visualizations')
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        
        if segment_col == 'country':
            # For country analysis, group by standard countries to avoid too many categories
            if len(stats_df) > 15:
                print(f"Too many {segment_col} values ({len(stats_df)}), limiting to top countries and creating a regional view")
                
                # Generate a regional view
                region_stats = self._aggregate_by_region(segment_col)
                
                # Visualize regional stats
                if region_stats is not None and len(region_stats) > 0:
                    self._create_bar_chart(
                        region_stats, 
                        'rating_mean', 
                        f'Average Rating by Region',
                        f'average_rating_by_region.png',
                        viz_dir,
                        y_label='Average Rating',
                        color_palette='viridis'
                    )
                    
                    self._create_bar_chart(
                        region_stats, 
                        'rating_count', 
                        f'Number of Reviews by Region',
                        f'review_count_by_region.png',
                        viz_dir,
                        y_label='Number of Reviews',
                        color_palette='magma'
                    )
                    
                    self._create_combined_chart(
                        region_stats, 
                        'region',
                        viz_dir
                    )
                
                # Limit country stats to top 15
                stats_df = stats_df.head(15)
        
        # Filter to top segments for cleaner visualizations
        top_stats = stats_df.head(15) if len(stats_df) > 15 else stats_df
        
        # 1. Average Rating by Segment
        self._create_bar_chart(
            top_stats, 
            'rating_mean', 
            f'Average Rating by {segment_col.title()}',
            f'average_rating_by_{segment_col}.png',
            viz_dir,
            y_label='Average Rating',
            color_palette='viridis'
        )
        
        # 2. Review Count by Segment
        self._create_bar_chart(
            top_stats, 
            'rating_count', 
            f'Number of Reviews by {segment_col.title()}',
            f'review_count_by_{segment_col}.png',
            viz_dir,
            y_label='Number of Reviews',
            color_palette='magma'
        )
        
        # 3. Add sentiment visualization if available
        if 'sentiment_mean' in top_stats.columns:
            self._create_bar_chart(
                top_stats, 
                'sentiment_mean', 
                f'Average Sentiment by {segment_col.title()}',
                f'sentiment_by_{segment_col}.png',
                viz_dir,
                y_label='Average Sentiment Score',
                color_palette='RdYlGn',
                center_zero=True
            )
        
        # 4. Create standalone combined chart
        self._create_combined_chart(
            top_stats, 
            segment_col,
            viz_dir
        )
    
    def _aggregate_by_region(self, segment_col):
        """
        Aggregate statistics by region for better visualization.
        """
        if 'region' not in self.reviews_df.columns:
            return None
            
        # Basic statistics by region
        region_stats = self.reviews_df.groupby('region').agg({
            'rating': ['mean', 'median', 'count', 'std'],
        })
        
        # Add sentiment statistics if available
        if 'sentiment_score' in self.reviews_df.columns:
            sentiment_stats = self.reviews_df.groupby('region')['sentiment_score'].agg(['mean', 'median', 'std'])
            # Create MultiIndex columns for sentiment stats
            sentiment_cols = pd.MultiIndex.from_product([['sentiment_score'], ['mean', 'median', 'std']])
            sentiment_df = pd.DataFrame(
                sentiment_stats.values, 
                index=sentiment_stats.index, 
                columns=sentiment_cols
            )
            region_stats = pd.concat([region_stats, sentiment_df], axis=1)
        
        # Flatten the MultiIndex columns safely
        if isinstance(region_stats.columns, pd.MultiIndex):
            flat_columns = []
            for col in region_stats.columns:
                if len(col) == 2:  # Standard case with (outer, inner)
                    outer, inner = col
                    flat_columns.append(f"{outer}_{inner}" if inner else outer)
                else:  # Handle other cases
                    flat_columns.append("_".join(str(c) for c in col if c))
            region_stats.columns = flat_columns
        
        # Filter out regions with very few reviews
        min_reviews = 5
        count_col = 'rating_count' if 'rating_count' in region_stats.columns else region_stats.columns[0]
        region_stats = region_stats[region_stats[count_col] >= min_reviews]
        
        # Sort by count for relevance
        region_stats = region_stats.sort_values(count_col, ascending=False)
        
        return region_stats
    
    def _create_bar_chart(self, df, value_col, title, filename, output_dir, 
                         y_label='Value', color_palette='viridis', center_zero=False):
        """Create a standardized bar chart."""
        plt.figure(figsize=(12, 6))
        
        # Create color mapper
        if center_zero and df[value_col].min() < 0:
            # For sentiment scores centered at zero
            cmap = plt.cm.get_cmap(color_palette)
            norm = plt.Normalize(vmin=-0.5, vmax=0.5)
            colors = [cmap(norm(val)) for val in df[value_col]]
        else:
            # Regular sequential palette
            colors = plt.cm.get_cmap(color_palette)(np.linspace(0.2, 0.8, len(df)))
            
        # Create bar chart
        bars = plt.bar(df.index, df[value_col], color=colors)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2., 
                height + (0.02 * df[value_col].max()), 
                f'{height:.2f}', 
                ha='center', va='bottom', 
                fontweight='bold'
            )
        
        plt.title(title, fontsize=14, pad=20)
        plt.ylabel(y_label, fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # If we have sentiment, add a zero line
        if center_zero:
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.close()

    def _create_combined_chart(self, df, segment_col, output_dir):
        """Create a combined chart with both rating and count information."""
        if len(df) == 0:
            return
            
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Primary axis for average rating
        color = 'tab:blue'
        ax1.set_xlabel(segment_col.title())
        ax1.set_ylabel('Average Rating', color=color)
        bars = ax1.bar(df.index, df['rating_mean'], color=color, alpha=0.7)
        
        # Add rating labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width()/2., 
                height + 0.1, 
                f'{height:.2f}', 
                ha='center', va='bottom', 
                color=color, fontweight='bold'
            )
        
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(0, 5.5)  # Assuming 5-star rating scale
        plt.xticks(rotation=45, ha='right')
        
        # Secondary axis for review count
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Number of Reviews', color=color)
        line = ax2.plot(df.index, df['rating_count'], marker='o', color=color, linewidth=2)
        
        # Add count labels
        for i, val in enumerate(df['rating_count']):
            ax2.text(
                i, 
                val + (0.03 * df['rating_count'].max()), 
                f'{val:.0f}', 
                ha='center', va='bottom', 
                color=color, fontweight='bold'
            )
            
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title(f'{segment_col.title()} Analysis: Ratings and Review Counts', fontsize=14, pad=20)
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'combined_{segment_col}_analysis.png'), dpi=300)
        plt.close()

    def generate_segment_recommendations(self):
        """
        Generate traveler segment-specific recommendations.
        
        Returns:
        - Dictionary with traveler segment recommendations
        """
        print("Generating traveler segment-specific recommendations...")
        
        recommendations = {
            "marketing_targeting": [],
            "product_development": [],
            "service_improvements": [],
            "segment_specific": {}
        }
        
        # Only proceed if we have enough data
        if len(self.reviews_df) < 10:
            print("Not enough data for generating recommendations")
            return recommendations
        
        # Analyze trip types
        if 'trip_type' in self.reviews_df.columns:
            trip_stats = self.reviews_df.groupby('trip_type').agg({
                'rating': ['mean', 'count']
            })
            
            trip_stats.columns = ['rating_mean', 'rating_count']
            trip_stats = trip_stats[trip_stats['rating_count'] >= 3]
            
            if not trip_stats.empty:
                # Best and worst trip types
                best_trip = trip_stats['rating_mean'].idxmax()
                worst_trip = trip_stats['rating_mean'].idxmin()
                most_common = trip_stats['rating_count'].idxmax()
                
                best_score = trip_stats.loc[best_trip, 'rating_mean']
                worst_score = trip_stats.loc[worst_trip, 'rating_mean']
                
                # Generate recommendations
                if best_score >= 4.0:
                    recommendations["marketing_targeting"].append(
                        f"Target {best_trip} travelers in marketing campaigns as they report "
                        f"the most positive experiences (rating: {best_score:.2f}/5)"
                    )
                
                if worst_score < 3.5 and worst_score > 0:
                    recommendations["service_improvements"].append(
                        f"Improve services for {worst_trip} travelers who report "
                        f"lower ratings (average: {worst_score:.2f}/5)"
                    )
                
                recommendations["product_development"].append(
                    f"Develop more offerings for {most_common} travelers, your most common visitor segment"
                )
                
                # Add segment-specific recommendations
                for trip_type, row in trip_stats.iterrows():
                    segment_recs = []
                    
                    if row['rating_mean'] >= 4.5:
                        segment_recs.append(
                            "Leverage extremely positive experiences in testimonials and marketing"
                        )
                    elif row['rating_mean'] <= 3.0:
                        segment_recs.append(
                            "Address below-average ratings to improve satisfaction"
                        )
                    
                    if segment_recs:
                        recommendations["segment_specific"][trip_type] = segment_recs
        
        # Analyze country/region origin
        if 'country' in self.reviews_df.columns:
            country_stats = self.reviews_df.groupby('country').agg({
                'rating': ['mean', 'count']
            })
            
            country_stats.columns = ['rating_mean', 'rating_count']
            country_stats = country_stats[country_stats['rating_count'] >= 3]
            
            if not country_stats.empty:
                # Best and worst countries
                best_country = country_stats['rating_mean'].idxmax()
                worst_country = country_stats['rating_mean'].idxmin()
                most_common_country = country_stats['rating_count'].idxmax()
                
                best_country_score = country_stats.loc[best_country, 'rating_mean']
                worst_country_score = country_stats.loc[worst_country, 'rating_mean']
                
                # Generate recommendations
                if best_country != 'Unknown' and best_country_score >= 4.0:
                    recommendations["marketing_targeting"].append(
                        f"Focus marketing efforts in {best_country} where visitors report "
                        f"the most positive experiences (rating: {best_country_score:.2f}/5)"
                    )
                
                if worst_country != 'Unknown' and worst_country_score < 3.5 and worst_country_score > 0:
                    recommendations["service_improvements"].append(
                        f"Address cultural expectations for visitors from {worst_country} "
                        f"who report lower ratings (average: {worst_country_score:.2f}/5)"
                    )
                
                if most_common_country != 'Unknown':
                    recommendations["product_development"].append(
                        f"Develop targeted offerings for visitors from {most_common_country}, "
                        f"your most common visitor origin"
                    )
        
        # Analyze cross-segments if we have enough data
        if (len(self.reviews_df) >= 20 and 
            'trip_type' in self.reviews_df.columns and 
            'country' in self.reviews_df.columns):
            
            # Get top segments with enough data
            top_trips = self.reviews_df['trip_type'].value_counts().head(3).index
            top_countries = self.reviews_df['country'].value_counts().head(3).index
            
            # Filter to segments with enough data
            cross_df = self.reviews_df[
                self.reviews_df['trip_type'].isin(top_trips) & 
                self.reviews_df['country'].isin(top_countries)
            ]
            
            if len(cross_df) >= 10:
                # Create pivot table for average ratings
                try:
                    cross_ratings = pd.pivot_table(
                        cross_df, 
                        values='rating', 
                        index='trip_type',
                        columns='country',
                        aggfunc='mean'
                    )
                    
                    # Find best and worst cross-segments
                    max_idx = cross_ratings.stack().idxmax()
                    min_idx = cross_ratings.stack().idxmin()
                    
                    best_trip_country = max_idx
                    worst_trip_country = min_idx
                    
                    best_score = cross_ratings.stack().max()
                    worst_score = cross_ratings.stack().min()
                    
                    # Generate cross-segment recommendations
                    if best_score >= 4.0:
                        trip, country = best_trip_country
                        recommendations["marketing_targeting"].append(
                            f"Develop targeted campaigns for {trip} travelers from {country} "
                            f"who report the most positive experiences (rating: {best_score:.2f}/5)"
                        )
                    
                    if worst_score < 3.5 and worst_score > 0:
                        trip, country = worst_trip_country
                        recommendations["service_improvements"].append(
                            f"Address specific needs of {trip} travelers from {country} "
                            f"who report lower ratings (average: {worst_score:.2f}/5)"
                        )
                except:
                    pass  # Skip if pivot table creation fails
        
        # Save recommendations to file
        with open(os.path.join(self.output_dir, 'traveler_segment_recommendations.json'), 'w', encoding='utf-8') as f:
            json.dump(recommendations, f, indent=2)
        
        # Print key recommendations
        print("\nKey traveler segment recommendations:")
        for category, recs in recommendations.items():
            if category != "segment_specific" and recs:
                print(f"\n{category.replace('_', ' ').title()}:")
                for rec in recs:
                    print(f"  - {rec}")
        
        return recommendations

    def analyze_top5_countries(self):
        """
        Special method to analyze only the top 5 specified countries: 
        New Zealand, Australia, United States, United Kingdom, and Tonga.
        This can be called separately to focus on country analysis.
        
        Returns:
        - DataFrame with detailed analysis for top 5 countries
        """
        print("\n=== Analyzing Top 5 Countries of Origin ===")
        
        # Make sure data is preprocessed
        if 'country' not in self.reviews_df.columns:
            self.preprocess_data()
            
        # Specify the target countries we want to analyze
        target_countries = ['New Zealand', 'Australia', 'United States', 'United Kingdom', 'Tonga']
        
        # Filter data to these countries
        top_countries_df = self.reviews_df[self.reviews_df['country'].isin(target_countries)]
        
        # Get the list of countries present in our data
        available_countries = top_countries_df['country'].unique().tolist()
        top_countries = [country for country in target_countries if country in available_countries]
        
        if len(top_countries) == 0:
            print("No valid country data found for the specified target countries.")
            return None
            
        print(f"Countries analyzed: {', '.join(top_countries)}")
        
        # If we don't have all 5 target countries in our data, add a note
        if len(top_countries) < len(target_countries):
            missing = [country for country in target_countries if country not in top_countries]
            print(f"Note: The following target countries are not present in the data: {', '.join(missing)}")
        
        # Create analysis directory
        country_dir = os.path.join(self.output_dir, 'top5_countries')
        if not os.path.exists(country_dir):
            os.makedirs(country_dir)
            
        # Basic statistics for each country
        country_stats = top_countries_df.groupby('country').agg({
            'rating': ['mean', 'median', 'count', 'std'],
            'trip_type': lambda x: x.value_counts().index[0] if len(x) > 0 else None
        })
        
        # Flatten MultiIndex
        flat_columns = []
        for col in country_stats.columns:
            if isinstance(col, tuple):
                outer, inner = col
                flat_columns.append(f"{outer}_{inner}" if inner != '<lambda_0>' else f"{outer}_most_common")
            else:
                flat_columns.append(col)
        country_stats.columns = flat_columns
        
        # Add sentiment if available
        if 'sentiment_score' in top_countries_df.columns:
            sentiment_stats = top_countries_df.groupby('country')['sentiment_score'].agg(['mean', 'std'])
            sentiment_stats.columns = ['sentiment_mean', 'sentiment_std']
            country_stats = country_stats.join(sentiment_stats)
        
        # Save the analysis
        country_stats.to_csv(os.path.join(country_dir, 'top5_countries_analysis.csv'))
        
        # Create visualizations
        self._create_top_countries_comparison(top_countries, country_dir)
        
        # Individual country analysis
        for country in top_countries:
            # Detailed analysis for this country
            self._analyze_country_details(country, country_dir)
        
        # Create cross-analysis between countries and trip types
        self._analyze_country_trip_type_matrix(top_countries_df, country_dir)
        
        print("\nTop 5 countries analysis complete.")
        return country_stats
        
    def _analyze_country_details(self, country, output_dir):
        """
        Create detailed analysis for a specific country.
        
        Parameters:
        - country: Country name to analyze
        - output_dir: Base output directory
        """
        # Filter to this country
        country_data = self.reviews_df[self.reviews_df['country'] == country]
        
        if len(country_data) < 5:
            return
            
        # Create country subdirectory
        country_subdir = os.path.join(output_dir, country.replace(' ', '_').lower())
        if not os.path.exists(country_subdir):
            os.makedirs(country_subdir)
            
        # Format country name for titles
        country_title = country.title() if country != country.upper() else country
        
        # 1. Trip type distribution for this country
        trip_counts = country_data['trip_type'].value_counts()
        
        plt.figure(figsize=(10, 6))
        trip_counts.plot(kind='pie', autopct='%1.1f%%', colors=plt.cm.tab10.colors, startangle=90)
        plt.title(f'Trip Type Distribution for Visitors from {country_title}', fontsize=14, pad=20)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(country_subdir, 'trip_type_distribution.png'), dpi=300)
        plt.close()
        
        # 2. Rating distribution
        plt.figure(figsize=(10, 6))
        country_data['rating'].value_counts().sort_index().plot(kind='bar', color=plt.cm.viridis(0.5))
        plt.title(f'Rating Distribution for Visitors from {country_title}', fontsize=14, pad=20)
        plt.xlabel('Rating', fontsize=12)
        plt.ylabel('Number of Reviews', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(country_subdir, 'rating_distribution.png'), dpi=300)
        plt.close()
        
        # 3. Sentiment analysis if available
        if 'sentiment_score' in country_data.columns:
            plt.figure(figsize=(10, 6))
            bins = np.linspace(-1, 1, 21)  # 20 bins from -1 to 1
            plt.hist(country_data['sentiment_score'], bins=bins, color=plt.cm.RdYlGn(0.6), alpha=0.7)
            plt.title(f'Sentiment Distribution for Visitors from {country_title}', fontsize=14, pad=20)
            plt.xlabel('Sentiment Score', fontsize=12)
            plt.ylabel('Number of Reviews', fontsize=12)
            plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(country_subdir, 'sentiment_distribution.png'), dpi=300)
            plt.close()
        
        # Save dataset for this country
        country_data.to_csv(os.path.join(country_subdir, f'{country.lower()}_reviews.csv'))
        
    def _analyze_country_trip_type_matrix(self, df, output_dir):
        """
        Create a matrix analysis of countries vs trip types.
        
        Parameters:
        - df: DataFrame containing reviews for top countries
        - output_dir: Output directory
        """
        if len(df) < 10:
            return
            
        # Create visualizations directory
        viz_dir = os.path.join(output_dir, 'visualizations')
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        
        # Use a consistent color mapping for better visualizations
        country_colors = {
            'New Zealand': '#1E88E5',   # Blue
            'Australia': '#FFC107',     # Amber
            'United States': '#D81B60', # Pink
            'United Kingdom': '#004D40', # Teal
            'Tonga': '#43A047'          # Green
        }
            
        # 1. Countries vs Trip Types heatmap (frequency)
        # Create a cross-tabulation of countries and trip types
        country_trip_counts = pd.crosstab(df['country'], df['trip_type'])
        
        # Calculate percentages within each country
        country_trip_pct = country_trip_counts.div(country_trip_counts.sum(axis=1), axis=0) * 100
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(country_trip_pct, annot=True, cmap='YlGnBu', fmt='.1f')
        plt.title('Trip Type Distribution by Country of Origin (%)', fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'country_trip_type_distribution.png'), dpi=300)
        plt.close()
        
        # Create a CSV file for this data
        country_trip_pct.to_csv(os.path.join(output_dir, 'cross_segment_trip_type_country.csv'))
        
        # 2. Countries vs Trip Types heatmap (ratings)
        try:
            # Create pivot table of average ratings
            ratings_pivot = pd.pivot_table(
                df, 
                values='rating', 
                index='country',
                columns='trip_type',
                aggfunc='mean'
            )
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(ratings_pivot, annot=True, cmap='RdYlGn', fmt='.2f')
            plt.title('Average Rating by Country and Trip Type', fontsize=14, pad=20)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'country_trip_type_ratings.png'), dpi=300)
            plt.close()
        except Exception as e:
            print(f"Could not create country/trip type ratings heatmap: {e}")
            
        # 3. Countries vs Trip Types heatmap (sentiment)
        if 'sentiment_score' in df.columns:
            try:
                # Create pivot table of average sentiment
                sentiment_pivot = pd.pivot_table(
                    df, 
                    values='sentiment_score', 
                    index='country',
                    columns='trip_type',
                    aggfunc='mean'
                )
                
                plt.figure(figsize=(12, 8))
                sns.heatmap(sentiment_pivot, annot=True, cmap='RdYlGn', center=0, fmt='.2f')
                plt.title('Average Sentiment by Country and Trip Type', fontsize=14, pad=20)
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, 'country_trip_type_sentiment.png'), dpi=300)
                plt.close()
            except Exception as e:
                print(f"Could not create country/trip type sentiment heatmap: {e}")
                
        # 4. Create a bubble chart of country, trip type, and rating
        try:
            # Group data by country and trip type
            trip_country_rating = df.groupby(['country', 'trip_type']).agg({
                'rating': ['mean', 'count']
            }).reset_index()
            
            # Flatten MultiIndex
            trip_country_rating.columns = ['country', 'trip_type', 'avg_rating', 'review_count']
            
            # Only include combinations with enough reviews
            trip_country_rating = trip_country_rating[trip_country_rating['review_count'] >= 3]
            
            if len(trip_country_rating) >= 5:
                # Create bubble chart
                plt.figure(figsize=(14, 10))
                
                # Get unique countries and trip types
                countries = trip_country_rating['country'].unique()
                trip_types = trip_country_rating['trip_type'].unique()
                
                # Create a color map for countries
                color_map = {country: country_colors.get(country, '#BDBDBD') for country in countries}
                
                # Create a marker map for trip types
                marker_map = {
                    'family': 'o',      # circle
                    'couple': 's',      # square
                    'solo': '^',        # triangle up
                    'friends': 'D',     # diamond
                    'business': 'P',    # plus filled
                    'other': 'X'        # x filled
                }
                
                # Plot each country-trip type combination
                for country in countries:
                    country_data = trip_country_rating[trip_country_rating['country'] == country]
                    
                    for trip_type in trip_types:
                        trip_data = country_data[country_data['trip_type'] == trip_type]
                        
                        if len(trip_data) > 0:
                            plt.scatter(
                                trip_data['country'],
                                trip_data['trip_type'],
                                s=trip_data['review_count'] * 30,  # Size based on review count
                                c=color_map.get(country, '#BDBDBD'),
                                marker=marker_map.get(trip_type, 'o'),
                                alpha=0.7,
                                edgecolors='black',
                                linewidth=1
                            )
                            
                            # Add rating as text label
                            for _, row in trip_data.iterrows():
                                plt.text(
                                    row['country'],
                                    row['trip_type'],
                                    f"{row['avg_rating']:.1f}",
                                    ha='center', va='center',
                                    fontsize=9, fontweight='bold'
                                )
                
                plt.title('Rating by Country and Trip Type', fontsize=16, pad=20)
                plt.xlabel('Country', fontsize=14)
                plt.ylabel('Trip Type', fontsize=14)
                plt.grid(alpha=0.3)
                
                # Create custom legend for bubble size
                min_count = trip_country_rating['review_count'].min()
                max_count = trip_country_rating['review_count'].max()
                med_count = (min_count + max_count) / 2
                
                # Create country color legend
                country_patches = [
                    plt.plot([], [], marker="o", ms=10, ls="", color=color, label=country)[0]
                    for country, color in color_map.items()
                ]
                
                # Create trip type marker legend
                trip_patches = [
                    plt.plot([], [], marker=marker, ms=10, ls="", color='gray', label=trip_type)[0]
                    for trip_type, marker in marker_map.items() if trip_type in trip_types
                ]
                
                # Add both legends
                plt.legend(
                    handles=country_patches + trip_patches,
                    loc='upper left',
                    bbox_to_anchor=(1.05, 1),
                    title="Country & Trip Type"
                )
                
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, 'country_trip_type_bubble.png'), dpi=300)
                plt.close()
        except Exception as e:
            print(f"Could not create country/trip type bubble chart: {e}")

    def run_analysis(self):
        """Run the complete analysis pipeline."""
        print("\n=== Running Traveler Segment Analysis ===")
        
        # Preprocess the data
        self.preprocess_data()
        
        # Analyze segments
        segment_analysis, origin_analysis = self.analyze_by_segment()
        
        # Generate visualizations
        self.visualize_by_segment(segment_analysis, 'trip_type')
        self.visualize_by_segment(origin_analysis, 'country')
        
        # Analyze top 5 countries in detail
        self.analyze_top5_countries()
        
        # Generate recommendations
        self.generate_segment_recommendations()
        
        print("\nTraveler segment analysis complete.")
        return segment_analysis, origin_analysis