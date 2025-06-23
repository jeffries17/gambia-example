#!/usr/bin/env python3
"""
Island Sentiment Comparison Script

This script analyzes and visualizes sentiment scores across Tonga's islands,
broken down by category (accommodations, attractions, restaurants).
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Import visualization styles
try:
    from tonga_analysis.visualization_styles import (
        set_visualization_style, ISLAND_COLORS, ISLAND_COLORS_LOWER
    )
except ImportError:
    # Define simple fallbacks for visualization styles
    def set_visualization_style():
        """Set a simple visualization style."""
        plt.style.use('seaborn-v0_8-whitegrid')
    
    # Island color schemes
    ISLAND_COLORS = {
        'Tongatapu': '#1f77b4',  # Blue
        'Vava\'u': '#ff7f0e',    # Orange
        'Ha\'apai': '#2ca02c',   # Green
        '\'Eua': '#d62728',      # Red
        'Niuas': '#9467bd',      # Purple
    }
    
    # Island colors with lowercase keys for easier matching
    ISLAND_COLORS_LOWER = {
        'tongatapu': '#1f77b4',  # Blue
        'vavau': '#ff7f0e',      # Orange
        'haapai': '#2ca02c',     # Green
        'eua': '#d62728',        # Red
        'niuas': '#9467bd',      # Purple
    }

class IslandSentimentComparison:
    """
    Analyzes and visualizes sentiment scores across Tonga's island groups,
    broken down by category (accommodation, attraction, restaurant).
    """
    
    def __init__(self, data_dir='tonga_data', output_dir=None):
        """
        Initialize the island sentiment comparison analyzer.
        
        Parameters:
        - data_dir: Directory containing the data JSON files
        - output_dir: Directory to save analysis outputs
        """
        self.data_dir = data_dir
        
        # Set default output directory to the standardized location
        if output_dir is None:
            # Use the parent directory's outputs folder
            parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            self.output_dir = os.path.join(parent_dir, "outputs", "island_sentiment")
        else:
            self.output_dir = output_dir
            
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")
            
        # Visualization directory
        self.viz_dir = os.path.join(self.output_dir, 'visualizations')
        if not os.path.exists(self.viz_dir):
            os.makedirs(self.viz_dir)
            
        # Data storage
        self.accommodations_df = None
        self.restaurants_df = None
        self.attractions_df = None
        self.all_reviews_df = None
        
        # Analysis results
        self.island_sentiment = {}
    
    def load_data(self, processed_dir=None):
        """
        Load the processed data files for analysis.
        
        Parameters:
        - processed_dir: Directory containing processed data files
        
        Returns:
        - Boolean indicating success
        """
        print("Loading data for island sentiment analysis...")
        
        # Use the standardized location for processed data
        if processed_dir is None:
            # First check if there are processed data files in the tonga_data directory
            latest_accommodations = None
            latest_attractions = None
            latest_restaurants = None
            
            # Look for processed data in the tonga_data directory
            for dirname in os.listdir(self.data_dir):
                if 'tonga_accommodations_processed_' in dirname:
                    processed_path = os.path.join(self.data_dir, dirname, 'tonga_accommodations_cleaned.json')
                    if os.path.exists(processed_path):
                        if latest_accommodations is None or dirname > latest_accommodations:
                            latest_accommodations = dirname
                
                if 'tonga_attractions_processed_' in dirname:
                    processed_path = os.path.join(self.data_dir, dirname, 'tonga_attractions_cleaned.json')
                    if os.path.exists(processed_path):
                        if latest_attractions is None or dirname > latest_attractions:
                            latest_attractions = dirname
                
                if 'tonga_restaurants_processed_' in dirname:
                    processed_path = os.path.join(self.data_dir, dirname, 'tonga_restaurants_cleaned.json')
                    if os.path.exists(processed_path):
                        if latest_restaurants is None or dirname > latest_restaurants:
                            latest_restaurants = dirname
            
            # If processed data exists, use it
            if latest_accommodations and latest_attractions and latest_restaurants:
                accommodations_path = os.path.join(self.data_dir, latest_accommodations, 'tonga_accommodations_cleaned.json')
                attractions_path = os.path.join(self.data_dir, latest_attractions, 'tonga_attractions_cleaned.json')
                restaurants_path = os.path.join(self.data_dir, latest_restaurants, 'tonga_restaurants_cleaned.json')
            else:
                # Fall back to raw data files
                accommodations_path = os.path.join(self.data_dir, 'tonga_accommodations.json')
                attractions_path = os.path.join(self.data_dir, 'tonga_attractions.json')
                restaurants_path = os.path.join(self.data_dir, 'tonga_restaurants.json')
        else:
            # Use specified processed data directory
            accommodations_path = os.path.join(processed_dir, 'tonga_accommodations_cleaned.json')
            attractions_path = os.path.join(processed_dir, 'tonga_attractions_cleaned.json')
            restaurants_path = os.path.join(processed_dir, 'tonga_restaurants_cleaned.json')
        
        # Load accommodations data
        if os.path.exists(accommodations_path):
            try:
                with open(accommodations_path, 'r', encoding='utf-8') as f:
                    accommodations = json.load(f)
                
                # Create DataFrame
                self.accommodations_df = pd.DataFrame(accommodations)
                self.accommodations_df['category'] = 'accommodation'
                print(f"Loaded {len(self.accommodations_df)} accommodation reviews")
            except Exception as e:
                print(f"Error loading accommodations data: {str(e)}")
        else:
            print(f"Accommodations data file not found at {accommodations_path}")
        
        # Load attractions data
        if os.path.exists(attractions_path):
            try:
                with open(attractions_path, 'r', encoding='utf-8') as f:
                    attractions = json.load(f)
                
                # Create DataFrame
                self.attractions_df = pd.DataFrame(attractions)
                self.attractions_df['category'] = 'attraction'
                print(f"Loaded {len(self.attractions_df)} attraction reviews")
            except Exception as e:
                print(f"Error loading attractions data: {str(e)}")
        else:
            print(f"Attractions data file not found at {attractions_path}")
        
        # Load restaurants data
        if os.path.exists(restaurants_path):
            try:
                with open(restaurants_path, 'r', encoding='utf-8') as f:
                    restaurants = json.load(f)
                
                # Create DataFrame
                self.restaurants_df = pd.DataFrame(restaurants)
                self.restaurants_df['category'] = 'restaurant'
                print(f"Loaded {len(self.restaurants_df)} restaurant reviews")
            except Exception as e:
                print(f"Error loading restaurants data: {str(e)}")
        else:
            print(f"Restaurants data file not found at {restaurants_path}")
        
        # Combine all DataFrames
        dataframes = []
        if self.accommodations_df is not None:
            dataframes.append(self.accommodations_df)
        if self.attractions_df is not None:
            dataframes.append(self.attractions_df)
        if self.restaurants_df is not None:
            dataframes.append(self.restaurants_df)
        
        if dataframes:
            self.all_reviews_df = pd.concat(dataframes, ignore_index=True)
            print(f"Combined {len(self.all_reviews_df)} reviews for analysis")
            return True
        else:
            print("No data was loaded successfully!")
            return False
    
    def standardize_island_name(self, location_string):
        """
        Extract and standardize island name from location string.
        
        Parameters:
        - location_string: Location string from review
        
        Returns:
        - Standardized island name
        """
        if not isinstance(location_string, str) or not location_string:
            return 'Unknown'
        
        # Map known variations to standard names
        island_mapping = {
            'Tongatapu': 'Tongatapu',
            'Vava\'u': 'Vava\'u',
            'Ha\'apai': 'Ha\'apai',
            '\'Eua': '\'Eua',
            'Lifuka': 'Ha\'apai',  # Lifuka is in Ha'apai group
            'Uoleva': 'Ha\'apai',
            'Foa': 'Ha\'apai',
            'Neiafu': 'Vava\'u',
            'Nuku\'alofa': 'Tongatapu',
            'Niuatoputapu': 'Niuas',
            'Niuafo\'ou': 'Niuas'
        }
        
        # Common pattern: "City, Island"
        parts = location_string.split(',')
        
        if len(parts) >= 2:
            # Try to match the second part (island)
            island_part = parts[1].strip()
            
            # Check for island name in the island part
            for key in island_mapping:
                if key.lower() in island_part.lower():
                    return island_mapping[key]
        
        # Try to match the city or any part of the location
        for key, value in island_mapping.items():
            if key.lower() in location_string.lower():
                return value
        
        # If no match found, extract from location
        if len(parts) >= 2:
            return parts[1].strip()
        elif len(parts) == 1:
            return parts[0].strip()
        
        return 'Unknown'
    
    def analyze_sentiment_by_island(self):
        """
        Analyze sentiment scores across islands and categories.
        
        Returns:
        - Dictionary with sentiment analysis results
        """
        if self.all_reviews_df is None:
            print("No data available for analysis! Run load_data() first.")
            return {}
        
        print("\nAnalyzing sentiment by island and category...")
        
        # Extract island information if not present
        print("Extracting island information...")
        if 'island' not in self.all_reviews_df.columns:
            if 'locationString' in self.all_reviews_df.columns:
                self.all_reviews_df['island'] = self.all_reviews_df['locationString'].apply(self.standardize_island_name)
            elif 'placeInfo' in self.all_reviews_df.columns and isinstance(self.all_reviews_df['placeInfo'].iloc[0], dict):
                # If placeInfo is a dictionary column, extract locationString from it
                self.all_reviews_df['island'] = self.all_reviews_df['placeInfo'].apply(
                    lambda x: self.standardize_island_name(x.get('locationString', '')) if isinstance(x, dict) else 'Unknown'
                )
            else:
                # Try to find any column that might contain location information
                location_columns = [col for col in self.all_reviews_df.columns if 'location' in col.lower()]
                if location_columns:
                    self.all_reviews_df['island'] = self.all_reviews_df[location_columns[0]].apply(self.standardize_island_name)
                else:
                    print("No location information found in the data. Creating default island column.")
                    self.all_reviews_df['island'] = 'Unknown'
        
        # Add sentiment analysis if not already done
        if 'sentiment_score' not in self.all_reviews_df.columns:
            print("Running sentiment analysis on the data...")
            from textblob import TextBlob
            
            # Calculate sentiment scores using TextBlob
            self.all_reviews_df['sentiment_score'] = self.all_reviews_df['text'].apply(
                lambda x: TextBlob(str(x)).sentiment.polarity if pd.notnull(x) else 0
            )
            
            # Calculate subjectivity for additional analysis
            self.all_reviews_df['subjectivity'] = self.all_reviews_df['text'].apply(
                lambda x: TextBlob(str(x)).sentiment.subjectivity if pd.notnull(x) else 0
            )
            
            # Categorize sentiment
            self.all_reviews_df['sentiment_category'] = pd.cut(
                self.all_reviews_df['sentiment_score'],
                bins=[-1, -0.1, 0.1, 1],
                labels=['negative', 'neutral', 'positive']
            )
            
            print(f"Sentiment analysis completed for {len(self.all_reviews_df)} reviews")
        
        # Define valid islands to analyze (excluding Unknown, Other, etc.)
        valid_islands = ['Tongatapu', 'Vava\'u', 'Ha\'apai', '\'Eua']
        
        # Filter data to valid islands
        island_data = self.all_reviews_df[self.all_reviews_df['island'].isin(valid_islands)]
        
        # Initialize results
        results = {
            'overall': {},
            'by_category': {},
            'by_island': {},
            'by_island_category': {}
        }
        
        # Overall sentiment statistics
        results['overall'] = {
            'avg_sentiment': island_data['sentiment_score'].mean(),
            'review_count': len(island_data)
        }
        
        # Sentiment by category
        category_sentiment = island_data.groupby('category').agg({
            'sentiment_score': ['mean', 'count', 'std']
        })
        
        # Convert to dictionary format
        for category_name, group_data in category_sentiment.iterrows():
            results['by_category'][category_name] = {
                'avg_sentiment': group_data[('sentiment_score', 'mean')],
                'review_count': group_data[('sentiment_score', 'count')],
                'std_dev': group_data[('sentiment_score', 'std')]
            }
        
        # Sentiment by island
        island_sentiment = island_data.groupby('island').agg({
            'sentiment_score': ['mean', 'count', 'std']
        })
        
        # Convert to dictionary format
        for island_name, group_data in island_sentiment.iterrows():
            results['by_island'][island_name] = {
                'avg_sentiment': group_data[('sentiment_score', 'mean')],
                'review_count': group_data[('sentiment_score', 'count')],
                'std_dev': group_data[('sentiment_score', 'std')]
            }
        
        # Sentiment by island and category
        island_category_sentiment = island_data.groupby(['island', 'category']).agg({
            'sentiment_score': ['mean', 'count', 'std']
        })
        
        # Convert to dictionary format
        for (island_name, category_name), group_data in island_category_sentiment.iterrows():
            if island_name not in results['by_island_category']:
                results['by_island_category'][island_name] = {}
                
            results['by_island_category'][island_name][category_name] = {
                'avg_sentiment': group_data[('sentiment_score', 'mean')],
                'review_count': group_data[('sentiment_score', 'count')],
                'std_dev': group_data[('sentiment_score', 'std')]
            }
        
        self.island_sentiment = results
        return results
    
    def generate_visualizations(self):
        """
        Generate visualizations for island sentiment analysis.
        """
        if not self.island_sentiment:
            print("No sentiment analysis data to visualize! Run analyze_sentiment_by_island() first.")
            return
        
        print("\nGenerating island sentiment visualizations...")
        
        # Apply consistent visualization style
        set_visualization_style()
        
        # Create a data frame for plotting
        plot_data = []
        for island, categories in self.island_sentiment['by_island_category'].items():
            for category, stats in categories.items():
                plot_data.append({
                    'Island': island,
                    'Category': category.title(),
                    'Average Sentiment': stats['avg_sentiment'],
                    'Review Count': stats['review_count'],
                    'Standard Deviation': stats['std_dev']
                })
        
        plot_df = pd.DataFrame(plot_data)
        
        # 1. Bar chart of average sentiment by island and category
        plt.figure(figsize=(12, 8))
        
        # Define colors for islands
        island_colors = {}
        for island in plot_df['Island'].unique():
            island_lower = island.lower().replace("'", "").replace("'", "")
            island_colors[island] = ISLAND_COLORS_LOWER.get(island_lower, '#AAAAAA')
        
        # Create nested bar chart
        ax = sns.barplot(
            x='Island', 
            y='Average Sentiment', 
            hue='Category',
            data=plot_df,
            palette='viridis'
        )
        
        # Customize the plot
        plt.title('Average Sentiment Score by Island and Category', fontsize=16, pad=20)
        plt.xlabel('Island', fontsize=14)
        plt.ylabel('Average Sentiment Score', fontsize=14)
        plt.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=0)
        plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add value labels
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'island_sentiment_by_category.png'), dpi=300)
        plt.close()
        
        # 2. Heatmap of sentiment by island and category
        plt.figure(figsize=(10, 8))
        
        # Create pivot table for heatmap
        heatmap_data = plot_df.pivot(index='Island', columns='Category', values='Average Sentiment')
        
        # Generate heatmap
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt='.3f',
            cmap='viridis',
            linewidths=.5,
            cbar_kws={'label': 'Average Sentiment Score'}
        )
        
        plt.title('Sentiment Score Heatmap by Island and Category', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'island_sentiment_heatmap.png'), dpi=300)
        plt.close()
        
        # 3. Combined chart with sentiment and review counts
        plt.figure(figsize=(14, 10))
        
        # Create plot with 2 y-axes
        fig, ax1 = plt.subplots(figsize=(14, 8))
        
        # Plot sentiment bars
        sns.barplot(
            x='Island', 
            y='Average Sentiment', 
            hue='Category',
            data=plot_df,
            ax=ax1,
            palette='viridis'
        )
        
        # Create second y-axis for review counts
        ax2 = ax1.twinx()
        
        # Plot review counts as markers
        for i, category in enumerate(plot_df['Category'].unique()):
            category_data = plot_df[plot_df['Category'] == category]
            ax2.plot(
                category_data.index, 
                category_data['Review Count'],
                'o-',
                label=f'{category} Reviews',
                alpha=0.7,
                markersize=8
            )
        
        # Customize first y-axis (sentiment)
        ax1.set_title('Sentiment Scores and Review Counts by Island and Category', fontsize=16, pad=20)
        ax1.set_xlabel('Island', fontsize=14)
        ax1.set_ylabel('Average Sentiment Score', fontsize=14)
        ax1.set_ylim(0, 1)
        ax1.grid(axis='y', alpha=0.3)
        
        # Customize second y-axis (review counts)
        ax2.set_ylabel('Number of Reviews', fontsize=14)
        ax2.tick_params(axis='y')
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(1.05, 1))
        
        # Remove the default legend from the first axis
        ax1.get_legend().remove()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'island_sentiment_with_review_counts.png'), dpi=300)
        plt.close()
        
        # 4. Radar chart of sentiment by island and category
        plt.figure(figsize=(12, 10))
        
        # Create pivot table to ensure all combinations exist
        radar_pivot = plot_df.pivot_table(
            index='Island', 
            columns='Category', 
            values='Average Sentiment',
            fill_value=0  # Fill missing values
        )
        
        # Prepare data for radar chart
        categories = radar_pivot.columns.tolist()
        islands = radar_pivot.index.tolist()
        
        # Number of variables
        N = len(categories)
        if N < 2:
            print("Not enough categories for radar chart. Skipping...")
            return
            
        # What will be the angle of each axis in the plot
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Initialize the spider plot
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Draw one axis per variable and add labels
        plt.xticks(angles[:-1], categories, size=12)
        
        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks([0.1, 0.3, 0.5, 0.7, 0.9], ["0.1", "0.3", "0.5", "0.7", "0.9"], size=10)
        plt.ylim(0, 1)
        
        # Plot each island
        for i, island in enumerate(islands):
            # Get sentiment values for this island from pivot table
            values = radar_pivot.loc[island].tolist()
            values += values[:1]  # Close the loop
            
            # Get color for this island
            island_lower = island.lower().replace("'", "").replace("'", "")
            color = ISLAND_COLORS_LOWER.get(island_lower, '#AAAAAA')
            
            # Plot values
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=island, color=color)
            ax.fill(angles, values, color=color, alpha=0.1)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.title('Sentiment Score Comparison by Island and Category', size=16, y=1.1)
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'island_sentiment_radar.png'), dpi=300)
        plt.close()
        
        print(f"Visualizations saved to {self.viz_dir}")
    
    def save_results(self):
        """
        Save analysis results to file.
        """
        if not self.island_sentiment:
            print("No analysis results to save!")
            return
        
        # Save to JSON file
        results_file = os.path.join(self.output_dir, 'island_sentiment_analysis.json')
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.island_sentiment, f, indent=2)
        
        print(f"Analysis results saved to {results_file}")
        
        # Export to CSV for easier reference
        # Create a dataframe for the island-category sentiment data
        csv_data = []
        for island, categories in self.island_sentiment['by_island_category'].items():
            for category, stats in categories.items():
                csv_data.append({
                    'Island': island,
                    'Category': category,
                    'Average Sentiment': stats['avg_sentiment'],
                    'Review Count': stats['review_count'],
                    'Standard Deviation': stats['std_dev']
                })
        
        csv_df = pd.DataFrame(csv_data)
        csv_file = os.path.join(self.output_dir, 'island_sentiment_summary.csv')
        csv_df.to_csv(csv_file, index=False)
        
        print(f"Summary data exported to {csv_file}")
    
    def run_analysis(self):
        """
        Run the complete island sentiment analysis pipeline.
        """
        print("\nStarting Island Sentiment Comparison Analysis...")
        
        # Step 1: Load data
        if self.load_data():
            # Step 2: Analyze sentiment by island
            self.analyze_sentiment_by_island()
            
            # Step 3: Generate visualizations
            self.generate_visualizations()
            
            # Step 4: Save results
            self.save_results()
            
            print("\nIsland sentiment analysis complete!")
            print(f"Results saved to: {self.output_dir}")
            return self.island_sentiment
        else:
            print("Analysis aborted due to data loading issues.")
            return None


if __name__ == "__main__":
    # Create and run the island sentiment comparison
    analyzer = IslandSentimentComparison()
    analyzer.run_analysis()