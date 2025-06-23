#!/usr/bin/env python3
"""
ARCHIVED VERSION: Specialized script to analyze accommodation types across different islands in Tonga.
This script extends the island_review_count.py functionality with detailed accommodation analysis.

This file has been replaced by the consolidated island_analyzer.py script.
DO NOT USE IN PRODUCTION.
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
import numpy as np

class IslandAccommodationAnalyzer:
    """
    Specialized analyzer for accommodation categories across different islands in Tonga.
    """
    
    def __init__(self, data_dir='tonga_data', output_dir='outputs/island_accommodation'):
        """
        Initialize the island accommodation analyzer.
        
        Parameters:
        - data_dir: Directory containing the accommodations JSON file
        - output_dir: Directory to save analysis outputs
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        
        # Visualization directory
        self.viz_dir = os.path.join(output_dir, 'visualizations')
        if not os.path.exists(self.viz_dir):
            os.makedirs(self.viz_dir)
            
        # Data storage
        self.accommodations_df = None
        self.island_accommodation_stats = {}
        
        # Define accommodation types to look for
        self.accommodation_types = {
            'resort': ['resort', 'spa'],
            'hotel': ['hotel', 'motel'],
            'lodge': ['lodge', 'cabin', 'bungalow'],
            'guesthouse': ['guesthouse', 'guest house', 'b&b', 'bed and breakfast'],
            'hostel': ['hostel', 'backpacker', 'dormitory'],
            'vacation_rental': ['vacation rental', 'holiday rental', 'apartment', 'villa', 'condo', 'cottage', 'airbnb', 'house rental'],
            'traditional': ['traditional', 'local', 'homestay', 'fale']
        }

    def standardize_island_name(self, location_string):
        """
        Extract and standardize island name from location string.
        
        Parameters:
        - location_string: Location string from review (e.g., "Neiafu, Vava'u Islands")
        
        Returns:
        - Standardized island name or 'Unknown' if not found
        """
        if not location_string:
            return 'Unknown'
        
        # Common patterns: "City, Island" or just "Island"
        parts = location_string.split(',')
        
        island_part = ''
        if len(parts) >= 2:
            # Extract the island part (after the comma)
            island_part = parts[1].strip()
        elif len(parts) == 1:
            # If there's no comma, assume the whole string might be the island
            island_part = parts[0].strip()
        else:
            return 'Unknown'
            
        # Remove "Islands" or "Island" suffix
        island_part = re.sub(r'\s+Islands$|\s+Island$', '', island_part)
        
        # Map known variations to standard names
        island_mapping = {
            'Tongatapu': 'Tongatapu',
            'Vava\'u': 'Vava\'u',
            'Ha\'apai': 'Ha\'apai',
            '\'Eua': '\'Eua',
            'Lifuka': 'Ha\'apai',  # Lifuka is in Ha'apai group
            'Niuatoputapu': 'Niuas',
            'Niuafo\'ou': 'Niuas'
        }
        
        # Check for each key in the mapping
        for key in island_mapping:
            if key.lower() in island_part.lower():
                return island_mapping[key]
        
        return island_part.strip()

    def extract_city_from_location(self, location_string):
        """
        Extract city name from location string.
        
        Parameters:
        - location_string: Location string from review (e.g., "Neiafu, Vava'u Islands")
        
        Returns:
        - Extracted city name or 'Unknown' if not found
        """
        if not location_string:
            return 'Unknown'
        
        # Common pattern: "City, Island"
        parts = location_string.split(',')
        
        if len(parts) >= 1:
            # Extract the city part (before the comma)
            return parts[0].strip()
        else:
            return 'Unknown'

    def detect_accommodation_type(self, review_text, title, place_name):
        """
        Detect accommodation type based on review text, title, and place name.
        
        Parameters:
        - review_text: Text of the review
        - title: Title of the review
        - place_name: Name of the accommodation
        
        Returns:
        - Detected accommodation type
        """
        # Combine all text for searching
        combined_text = f"{review_text} {title} {place_name}".lower()
        
        # Check for each accommodation type
        for accom_type, keywords in self.accommodation_types.items():
            for keyword in keywords:
                if keyword.lower() in combined_text:
                    return accom_type
        
        # Default to unknown if no type is detected
        return 'other'

    def load_and_process_data(self):
        """
        Load and process accommodation data with island and type classification.
        """
        accommodations_path = os.path.join(self.data_dir, 'tonga_accommodations.json')
        
        if not os.path.exists(accommodations_path):
            print(f"Error: Accommodation data file not found at {accommodations_path}")
            return False
        
        try:
            with open(accommodations_path, 'r', encoding='utf-8') as f:
                accommodations = json.load(f)
                
            print(f"Successfully loaded {len(accommodations)} accommodation reviews")
            
            # Process accommodation reviews
            processed_accommodations = []
            
            for review in accommodations:
                try:
                    # Extract location string
                    location_string = review.get('placeInfo', {}).get('locationString', '')
                    
                    # Extract island and city from location string
                    island = self.standardize_island_name(location_string)
                    city = self.extract_city_from_location(location_string)
                    
                    # Extract place name
                    place_name = review.get('placeInfo', {}).get('name', '')
                    
                    # Detect accommodation type
                    accommodation_type = self.detect_accommodation_type(
                        review.get('text', ''),
                        review.get('title', ''),
                        place_name
                    )
                    
                    # Create processed review
                    processed_review = {
                        'id': review.get('id'),
                        'rating': review.get('rating'),
                        'published_date': review.get('publishedDate'),
                        'trip_type': review.get('tripType'),
                        'text': review.get('text', ''),
                        'title': review.get('title', ''),
                        'helpful_votes': review.get('helpfulVotes', 0),
                        'place_name': place_name,
                        'location_string': location_string,
                        'island': island,
                        'city': city,
                        'accommodation_type': accommodation_type
                    }
                    
                    # Add user information if available
                    user_data = review.get('user', {})
                    if user_data:
                        processed_review['user_name'] = user_data.get('name', '')
                        
                        # Extract user location if available
                        user_location = user_data.get('userLocation', {})
                        if isinstance(user_location, dict):
                            processed_review['user_location'] = user_location.get('name', '')
                        else:
                            processed_review['user_location'] = ''
                    
                    processed_accommodations.append(processed_review)
                    
                except Exception as e:
                    print(f"Error processing review {review.get('id')}: {str(e)}")
                    continue
            
            # Convert to DataFrame
            if processed_accommodations:
                self.accommodations_df = pd.DataFrame(processed_accommodations)
                
                # Convert date to datetime
                self.accommodations_df['published_date'] = pd.to_datetime(
                    self.accommodations_df['published_date'], errors='coerce'
                )
                
                # Convert rating to numeric
                self.accommodations_df['rating'] = pd.to_numeric(
                    self.accommodations_df['rating'], errors='coerce'
                )
                
                print(f"Processed {len(self.accommodations_df)} accommodation reviews")
                
                # Get unique islands and types for reference
                print(f"\nUnique islands found: {', '.join(self.accommodations_df['island'].unique())}")
                print(f"Accommodation types found: {', '.join(self.accommodations_df['accommodation_type'].unique())}")
                
                return True
            else:
                print("No accommodation reviews could be processed!")
                return False
                
        except Exception as e:
            print(f"Error loading accommodation data: {str(e)}")
            return False

    def analyze_by_island_and_type(self, top_n_islands=5):
        """
        Analyze accommodations by island and type.
        
        Parameters:
        - top_n_islands: Number of top islands to analyze by review count
        """
        if self.accommodations_df is None or len(self.accommodations_df) == 0:
            print("No accommodation data to analyze!")
            return
        
        print(f"\nAnalyzing accommodations by island and type...")
        
        # Count reviews by island
        island_counts = self.accommodations_df['island'].value_counts()
        
        # Get top N islands
        top_islands = island_counts.head(top_n_islands).index.tolist()
        print(f"\nTop {len(top_islands)} islands by accommodation review count:")
        for i, island in enumerate(top_islands, 1):
            print(f"{i}. {island}: {island_counts[island]} reviews")
        
        # Create 'Other' category for remaining islands
        self.accommodations_df['island_category'] = self.accommodations_df['island'].apply(
            lambda x: x if x in top_islands else 'Other'
        )
        
        # Analyze accommodations by island and type
        for island in top_islands + ['Other']:
            island_data = self.accommodations_df[self.accommodations_df['island_category'] == island]
            
            # Count by accommodation type
            type_counts = island_data['accommodation_type'].value_counts().to_dict()
            
            # Average ratings by accommodation type
            avg_ratings_by_type = island_data.groupby('accommodation_type')['rating'].mean().to_dict()
            
            # Review counts by accommodation type
            review_counts_by_type = island_data.groupby('accommodation_type')['id'].count().to_dict()
            
            # Get top accommodations by rating (with minimum 3 reviews)
            top_accommodations = (
                island_data.groupby('place_name')
                .filter(lambda x: len(x) >= 3)  # Only places with at least 3 reviews
                .groupby('place_name')
                .agg({
                    'rating': 'mean',
                    'id': 'count',
                    'accommodation_type': lambda x: x.iloc[0]  # Take first type listed
                })
                .sort_values('rating', ascending=False)
                .head(5)
                .reset_index()
            )
            
            # Store the analysis results
            self.island_accommodation_stats[island] = {
                'total_reviews': len(island_data),
                'type_counts': type_counts,
                'avg_ratings_by_type': avg_ratings_by_type,
                'review_counts_by_type': review_counts_by_type,
                'top_accommodations': top_accommodations.to_dict('records') if not top_accommodations.empty else []
            }
        
        # Save the results to JSON
        self._save_results()

    def generate_visualizations(self):
        """
        Generate visualizations for accommodation types and ratings by island.
        """
        if not self.island_accommodation_stats:
            print("No data available for visualizations!")
            return
        
        print("\nGenerating visualizations for accommodation analysis...")
        
        # Set style
        sns.set(style="whitegrid")
        colors = plt.cm.viridis(np.linspace(0, 1, 10))
        
        # 1. Accommodation Type Distribution by Island
        plt.figure(figsize=(14, 8))
        
        # Prepare data for plotting
        type_data = []
        for island, stats in self.island_accommodation_stats.items():
            for accom_type, count in stats['type_counts'].items():
                type_data.append({
                    'Island': island,
                    'Accommodation Type': accom_type,
                    'Count': count
                })
        
        if type_data:
            type_df = pd.DataFrame(type_data)
            
            # Create pivot table
            type_pivot = type_df.pivot(
                index='Island',
                columns='Accommodation Type',
                values='Count'
            ).fillna(0)
            
            # Plot stacked bar chart
            type_pivot.plot(kind='bar', stacked=True, figsize=(14, 8), colormap='viridis')
            plt.title('Accommodation Types by Island', fontsize=16)
            plt.xlabel('Island', fontsize=14)
            plt.ylabel('Number of Reviews', fontsize=14)
            plt.xticks(rotation=45, ha='right', fontsize=12)
            plt.legend(title='Accommodation Type', fontsize=12, title_fontsize=13)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, 'accommodation_types_by_island.png'), dpi=300)
            plt.close()
            
            # Create proportional stacked bar chart
            plt.figure(figsize=(14, 8))
            type_prop = type_pivot.div(type_pivot.sum(axis=1), axis=0)
            type_prop.plot(kind='bar', stacked=True, figsize=(14, 8), colormap='viridis')
            plt.title('Proportion of Accommodation Types by Island', fontsize=16)
            plt.xlabel('Island', fontsize=14)
            plt.ylabel('Proportion', fontsize=14)
            plt.xticks(rotation=45, ha='right', fontsize=12)
            plt.legend(title='Accommodation Type', fontsize=12, title_fontsize=13)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, 'accommodation_types_proportion_by_island.png'), dpi=300)
            plt.close()
        
        # 2. Average Rating by Island and Accommodation Type
        plt.figure(figsize=(14, 8))
        
        # Prepare data for plotting
        rating_data = []
        for island, stats in self.island_accommodation_stats.items():
            for accom_type, rating in stats['avg_ratings_by_type'].items():
                count = stats['review_counts_by_type'].get(accom_type, 0)
                if count >= 3:  # Only include types with at least 3 reviews for more reliable averages
                    rating_data.append({
                        'Island': island,
                        'Accommodation Type': accom_type,
                        'Average Rating': rating,
                        'Review Count': count
                    })
        
        if rating_data:
            rating_df = pd.DataFrame(rating_data)
            
            # Create pivot table for ratings
            rating_pivot = rating_df.pivot(
                index='Island',
                columns='Accommodation Type',
                values='Average Rating'
            )
            
            # Plot heatmap
            plt.figure(figsize=(14, 8))
            sns.heatmap(
                rating_pivot,
                annot=True,
                cmap='YlGnBu',
                linewidths=.5,
                fmt='.2f',
                vmin=3.0,  # Assuming 5-star scale with minimum rating of interest at 3.0
                vmax=5.0
            )
            plt.title('Average Rating by Island and Accommodation Type', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, 'rating_by_island_and_type_heatmap.png'), dpi=300)
            plt.close()
            
            # Plot bar chart
            plt.figure(figsize=(14, 8))
            rating_pivot.plot(kind='bar', figsize=(14, 8), colormap='viridis')
            plt.title('Average Rating by Island and Accommodation Type', fontsize=16)
            plt.xlabel('Island', fontsize=14)
            plt.ylabel('Average Rating', fontsize=14)
            plt.xticks(rotation=45, ha='right', fontsize=12)
            plt.ylim(3.0, 5.0)  # Assuming 5-star scale with minimum 3.0 for better visualization
            plt.legend(title='Accommodation Type', fontsize=12, title_fontsize=13)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, 'rating_by_island_and_type_bars.png'), dpi=300)
            plt.close()
        
        # 3. Top Accommodations by Island (with ratings)
        for island, stats in self.island_accommodation_stats.items():
            top_accommodations = stats.get('top_accommodations', [])
            if top_accommodations:
                plt.figure(figsize=(12, 6))
                
                # Convert to DataFrame for easier plotting
                top_df = pd.DataFrame(top_accommodations)
                
                # Plot
                bars = plt.barh(
                    top_df['place_name'],
                    top_df['rating'],
                    color=[plt.cm.viridis(i/len(top_df)) for i in range(len(top_df))]
                )
                
                # Add count annotations
                for i, (_, row) in enumerate(top_df.iterrows()):
                    plt.text(
                        row['rating'] + 0.05,
                        i,
                        f"({row['id']} reviews)",
                        va='center'
                    )
                
                plt.xlim(3, 5.3)  # Adjust as needed for your rating scale
                plt.title(f'Top Rated Accommodations in {island}', fontsize=15)
                plt.xlabel('Average Rating', fontsize=12)
                plt.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(self.viz_dir, f'top_accommodations_{island.lower()}.png'), dpi=300)
                plt.close()
        
        print(f"Visualizations saved to {self.viz_dir}")

    def _save_results(self):
        """Save analysis results to JSON file."""
        results_file = os.path.join(self.output_dir, 'island_accommodation_summary.json')
        
        # Convert any non-serializable objects to strings or JSON-compatible types
        serializable_stats = {}
        for island, stats in self.island_accommodation_stats.items():
            serializable_stats[island] = {
                'total_reviews': stats['total_reviews'],
                'type_counts': stats['type_counts'],
                'avg_ratings_by_type': {k: float(v) for k, v in stats['avg_ratings_by_type'].items()},
                'review_counts_by_type': stats['review_counts_by_type'],
                'top_accommodations': []
            }
            
            # Process top accommodations
            for accom in stats.get('top_accommodations', []):
                processed_accom = {}
                for key, value in accom.items():
                    if isinstance(value, (pd.Series, pd.DataFrame)):
                        processed_accom[key] = value.to_dict()
                    elif isinstance(value, np.float64):
                        processed_accom[key] = float(value)
                    elif isinstance(value, np.int64):
                        processed_accom[key] = int(value)
                    else:
                        processed_accom[key] = value
                serializable_stats[island]['top_accommodations'].append(processed_accom)
        
        # Save to file
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_stats, f, indent=2)
        
        print(f"Analysis results saved to {results_file}")

    def export_to_excel(self):
        """Export analysis results to Excel for further analysis."""
        if self.accommodations_df is None or len(self.accommodations_df) == 0:
            print("No data to export!")
            return
        
        try:
            # Try to import openpyxl
            import openpyxl
            excel_path = os.path.join(self.output_dir, 'island_accommodation_comparison.xlsx')
            
            # Create Excel writer
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # Sheet 1: Summary statistics
                summary_data = []
                for island, stats in self.island_accommodation_stats.items():
                    row = {
                        'Island': island,
                        'Total Reviews': stats['total_reviews']
                    }
                    
                    # Add counts for each accommodation type
                    for accom_type in self.accommodation_types.keys():
                        row[f'{accom_type.title()} Count'] = stats['type_counts'].get(accom_type, 0)
                        row[f'{accom_type.title()} Avg Rating'] = stats['avg_ratings_by_type'].get(accom_type, 0)
                    
                    summary_data.append(row)
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Sheet 2: Raw data with island and type classifications
                self.accommodations_df.to_excel(writer, sheet_name='Raw Data', index=False)
                
                # Sheet 3: Pivot table of types by island
                type_pivot = pd.pivot_table(
                    self.accommodations_df,
                    values='id',
                    index=['island_category'],
                    columns=['accommodation_type'],
                    aggfunc='count',
                    fill_value=0
                )
                type_pivot.to_excel(writer, sheet_name='Type Counts')
                
                # Sheet 4: Pivot table of average ratings
                rating_pivot = pd.pivot_table(
                    self.accommodations_df,
                    values='rating',
                    index=['island_category'],
                    columns=['accommodation_type'],
                    aggfunc='mean',
                    fill_value=0
                )
                rating_pivot.to_excel(writer, sheet_name='Average Ratings')
                
                # Sheet 5: Top accommodations by island
                top_accommodations = []
                for island, stats in self.island_accommodation_stats.items():
                    for accom in stats.get('top_accommodations', []):
                        accom_dict = {
                            'Island': island,
                            'Place Name': accom.get('place_name', ''),
                            'Average Rating': accom.get('rating', 0),
                            'Review Count': accom.get('id', 0),
                            'Accommodation Type': accom.get('accommodation_type', '')
                        }
                        top_accommodations.append(accom_dict)
                
                if top_accommodations:
                    top_df = pd.DataFrame(top_accommodations)
                    top_df.to_excel(writer, sheet_name='Top Accommodations', index=False)
            
            print(f"Analysis exported to Excel: {excel_path}")
            
        except ImportError:
            print("\nopenpyxl not found. Cannot export to Excel.")
            print("To export to Excel, install openpyxl: pip install openpyxl")
            
            # Export to CSV instead
            csv_path = os.path.join(self.output_dir, 'island_accommodation_summary.csv')
            
            # Prepare summary data
            summary_data = []
            for island, stats in self.island_accommodation_stats.items():
                row = {
                    'Island': island,
                    'Total Reviews': stats['total_reviews']
                }
                
                # Add counts for each accommodation type
                for accom_type in self.accommodation_types.keys():
                    row[f'{accom_type.title()} Count'] = stats['type_counts'].get(accom_type, 0)
                    row[f'{accom_type.title()} Avg Rating'] = stats['avg_ratings_by_type'].get(accom_type, 0)
                
                summary_data.append(row)
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(csv_path, index=False)
            print(f"Summary data exported to CSV: {csv_path}")

    def run_analysis(self, top_n_islands=5):
        """
        Run the complete island accommodation analysis.
        
        Parameters:
        - top_n_islands: Number of top islands to analyze by review count
        """
        print("Starting Tonga Island Accommodation Analysis...")
        
        # Load and process data
        if self.load_and_process_data():
            # Analyze by island and accommodation type
            self.analyze_by_island_and_type(top_n_islands=top_n_islands)
            
            # Generate visualizations
            self.generate_visualizations()
            
            # Export to Excel
            self.export_to_excel()
            
            print("\nIsland accommodation analysis complete!")
        else:
            print("Analysis aborted due to data loading issues.")


if __name__ == "__main__":
    # Create and run the island accommodation analyzer
    analyzer = IslandAccommodationAnalyzer()
    analyzer.run_analysis(top_n_islands=5)