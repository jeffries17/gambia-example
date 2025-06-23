#!/usr/bin/env python3
"""
ARCHIVED VERSION: Specialized script to analyze unique properties across different islands in Tonga.
This script counts unique properties by type for each island rather than counting reviews.

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

class IslandPropertyAnalyzer:
    """
    Specialized analyzer for unique properties across different islands in Tonga.
    Focuses on counting unique properties rather than reviews.
    """
    
    def __init__(self, data_dir='tonga_data', output_dir='outputs/island_properties'):
        """
        Initialize the island property analyzer.
        
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
        self.island_property_stats = {}
        self.properties_df = None  # Will hold unique property data
        
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

    def detect_accommodation_type(self, place_name, place_description=''):
        """
        Detect accommodation type based on place name and description.
        
        Parameters:
        - place_name: Name of the accommodation
        - place_description: Description of the accommodation (if available)
        
        Returns:
        - Detected accommodation type
        """
        # Combine text for searching
        combined_text = f"{place_name} {place_description}".lower()
        
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
        Focus on unique properties rather than reviews.
        """
        accommodations_path = os.path.join(self.data_dir, 'tonga_accommodations.json')
        
        if not os.path.exists(accommodations_path):
            print(f"Error: Accommodation data file not found at {accommodations_path}")
            return False
        
        try:
            with open(accommodations_path, 'r', encoding='utf-8') as f:
                accommodations = json.load(f)
                
            print(f"Successfully loaded {len(accommodations)} accommodation reviews")
            
            # Process all reviews to get metadata about each property
            self.accommodations_df = pd.DataFrame(accommodations)
            
            # Extract properties data from placeInfo
            properties = {}  # Dictionary to hold property data
            
            for _, review in self.accommodations_df.iterrows():
                try:
                    place_info = review.get('placeInfo', {})
                    place_id = place_info.get('id', 'unknown')
                    place_name = place_info.get('name', 'Unknown Property')
                    location_string = place_info.get('locationString', '')
                    
                    # Only process if we have a property ID and it's not already processed
                    if place_id not in properties and place_id != 'unknown':
                        # Extract island and city
                        island = self.standardize_island_name(location_string)
                        city = self.extract_city_from_location(location_string)
                        
                        # Determine accommodation type
                        accom_type = self.detect_accommodation_type(place_name)
                        
                        # Store property data
                        properties[place_id] = {
                            'property_id': place_id,
                            'property_name': place_name,
                            'island': island,
                            'city': city,
                            'accommodation_type': accom_type,
                            'location_string': location_string,
                            'review_count': 0,
                            'avg_rating': 0.0,
                            'ratings': []
                        }
                
                except Exception as e:
                    print(f"Error processing property: {str(e)}")
                    continue
            
            # Now calculate review counts and ratings for each property
            for _, review in self.accommodations_df.iterrows():
                try:
                    place_id = review.get('placeInfo', {}).get('id', 'unknown')
                    rating = review.get('rating')
                    
                    if place_id in properties and rating is not None:
                        properties[place_id]['review_count'] += 1
                        properties[place_id]['ratings'].append(rating)
                
                except Exception as e:
                    continue
            
            # Calculate average ratings
            for place_id, prop in properties.items():
                if prop['ratings']:
                    prop['avg_rating'] = sum(prop['ratings']) / len(prop['ratings'])
                # Remove ratings list to clean up data
                del prop['ratings']
            
            # Convert to DataFrame
            self.properties_df = pd.DataFrame(list(properties.values()))
            
            if not self.properties_df.empty:
                print(f"Processed {len(self.properties_df)} unique properties")
                
                # Get unique islands and types for reference
                print(f"\nUnique islands found: {', '.join(self.properties_df['island'].unique())}")
                print(f"Accommodation types found: {', '.join(self.properties_df['accommodation_type'].unique())}")
                
                # Print counts by island
                print("\nProperty counts by island:")
                island_counts = self.properties_df['island'].value_counts()
                for island, count in island_counts.items():
                    print(f"  {island}: {count} properties")
                
                return True
            else:
                print("No properties could be processed!")
                return False
                
        except Exception as e:
            print(f"Error loading accommodation data: {str(e)}")
            return False

    def analyze_by_island_and_type(self, top_n_islands=5):
        """
        Analyze properties by island and type.
        
        Parameters:
        - top_n_islands: Number of top islands to analyze by property count
        """
        if self.properties_df is None or len(self.properties_df) == 0:
            print("No property data to analyze!")
            return
        
        print(f"\nAnalyzing properties by island and type...")
        
        # Count properties by island
        island_counts = self.properties_df['island'].value_counts()
        
        # Get top N islands
        top_islands = island_counts.head(top_n_islands).index.tolist()
        print(f"\nTop {len(top_islands)} islands by property count:")
        for i, island in enumerate(top_islands, 1):
            print(f"{i}. {island}: {island_counts[island]} properties")
        
        # Create 'Other' category for remaining islands
        self.properties_df['island_category'] = self.properties_df['island'].apply(
            lambda x: x if x in top_islands else 'Other'
        )
        
        # Analyze properties by island and type
        for island in top_islands + ['Other']:
            island_data = self.properties_df[self.properties_df['island_category'] == island]
            
            # Count by accommodation type
            type_counts = island_data['accommodation_type'].value_counts().to_dict()
            
            # Average ratings by accommodation type
            avg_ratings_by_type = island_data.groupby('accommodation_type')['avg_rating'].mean().to_dict()
            
            # Property counts by accommodation type
            property_counts_by_type = island_data.groupby('accommodation_type')['property_id'].count().to_dict()
            
            # Get top properties by rating (with minimum 3 reviews)
            top_properties = (
                island_data[island_data['review_count'] >= 3]  # Only places with at least 3 reviews
                .sort_values('avg_rating', ascending=False)
                .head(5)
            )
            
            # Store the analysis results
            self.island_property_stats[island] = {
                'total_properties': len(island_data),
                'type_counts': type_counts,
                'avg_ratings_by_type': avg_ratings_by_type,
                'property_counts_by_type': property_counts_by_type,
                'top_properties': top_properties.to_dict('records') if not top_properties.empty else []
            }
        
        # Save the results to JSON
        self._save_results()

    def generate_visualizations(self):
        """
        Generate visualizations for property types and ratings by island.
        """
        if not self.island_property_stats:
            print("No data available for visualizations!")
            return
        
        print("\nGenerating visualizations for island property analysis...")
        
        # Set style
        sns.set(style="whitegrid")
        colors = plt.cm.viridis(np.linspace(0, 1, 10))
        
        # 1. Property Type Distribution by Island
        plt.figure(figsize=(14, 8))
        
        # Prepare data for plotting
        type_data = []
        for island, stats in self.island_property_stats.items():
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
            plt.title('Property Types by Island', fontsize=16)
            plt.xlabel('Island', fontsize=14)
            plt.ylabel('Number of Properties', fontsize=14)
            plt.xticks(rotation=45, ha='right', fontsize=12)
            plt.legend(title='Accommodation Type', fontsize=12, title_fontsize=13)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, 'property_types_by_island.png'), dpi=300)
            plt.close()
            
            # Create proportional stacked bar chart
            plt.figure(figsize=(14, 8))
            type_prop = type_pivot.div(type_pivot.sum(axis=1), axis=0)
            type_prop.plot(kind='bar', stacked=True, figsize=(14, 8), colormap='viridis')
            plt.title('Proportion of Property Types by Island', fontsize=16)
            plt.xlabel('Island', fontsize=14)
            plt.ylabel('Proportion', fontsize=14)
            plt.xticks(rotation=45, ha='right', fontsize=12)
            plt.legend(title='Accommodation Type', fontsize=12, title_fontsize=13)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, 'property_types_proportion_by_island.png'), dpi=300)
            plt.close()
        
        # 2. Average Rating by Island and Accommodation Type
        plt.figure(figsize=(14, 8))
        
        # Prepare data for plotting
        rating_data = []
        for island, stats in self.island_property_stats.items():
            for accom_type, rating in stats['avg_ratings_by_type'].items():
                count = stats['property_counts_by_type'].get(accom_type, 0)
                if count >= 2:  # Only include types with at least 2 properties for more reliable averages
                    rating_data.append({
                        'Island': island,
                        'Accommodation Type': accom_type,
                        'Average Rating': rating,
                        'Property Count': count
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
            plt.title('Average Rating by Island and Property Type', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, 'rating_by_island_and_type_heatmap.png'), dpi=300)
            plt.close()
            
            # Plot bar chart
            plt.figure(figsize=(14, 8))
            rating_pivot.plot(kind='bar', figsize=(14, 8), colormap='viridis')
            plt.title('Average Rating by Island and Property Type', fontsize=16)
            plt.xlabel('Island', fontsize=14)
            plt.ylabel('Average Rating', fontsize=14)
            plt.xticks(rotation=45, ha='right', fontsize=12)
            plt.ylim(3.0, 5.0)  # Assuming 5-star scale with minimum 3.0 for better visualization
            plt.legend(title='Accommodation Type', fontsize=12, title_fontsize=13)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, 'rating_by_island_and_type_bars.png'), dpi=300)
            plt.close()
        
        # 3. Top Properties by Island (with ratings)
        for island, stats in self.island_property_stats.items():
            top_properties = stats.get('top_properties', [])
            if top_properties:
                plt.figure(figsize=(12, 6))
                
                # Convert to DataFrame for easier plotting
                top_df = pd.DataFrame(top_properties)
                
                # Plot
                bars = plt.barh(
                    top_df['property_name'],
                    top_df['avg_rating'],
                    color=[plt.cm.viridis(i/len(top_df)) for i in range(len(top_df))]
                )
                
                # Add count annotations
                for i, (_, row) in enumerate(top_df.iterrows()):
                    plt.text(
                        row['avg_rating'] + 0.05,
                        i,
                        f"({row['review_count']} reviews)",
                        va='center'
                    )
                
                plt.xlim(3, 5.3)  # Adjust as needed for your rating scale
                plt.title(f'Top Rated Properties in {island}', fontsize=15)
                plt.xlabel('Average Rating', fontsize=12)
                plt.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(self.viz_dir, f'top_properties_{island.lower().replace("\'", "")}.png'), dpi=300)
                plt.close()
        
        print(f"Visualizations saved to {self.viz_dir}")

    def _save_results(self):
        """Save analysis results to JSON file."""
        results_file = os.path.join(self.output_dir, 'island_property_summary.json')
        
        # Convert any non-serializable objects to strings or JSON-compatible types
        serializable_stats = {}
        for island, stats in self.island_property_stats.items():
            serializable_stats[island] = {
                'total_properties': stats['total_properties'],
                'type_counts': stats['type_counts'],
                'avg_ratings_by_type': {k: float(v) for k, v in stats['avg_ratings_by_type'].items()},
                'property_counts_by_type': stats['property_counts_by_type'],
                'top_properties': []
            }
            
            # Process top properties
            for prop in stats.get('top_properties', []):
                processed_prop = {}
                for key, value in prop.items():
                    if isinstance(value, (pd.Series, pd.DataFrame)):
                        processed_prop[key] = value.to_dict()
                    elif isinstance(value, np.float64):
                        processed_prop[key] = float(value)
                    elif isinstance(value, np.int64):
                        processed_prop[key] = int(value)
                    else:
                        processed_prop[key] = value
                serializable_stats[island]['top_properties'].append(processed_prop)
        
        # Save to file
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_stats, f, indent=2)
        
        print(f"Analysis results saved to {results_file}")

    def export_to_excel(self):
        """Export analysis results to Excel for further analysis."""
        if self.properties_df is None or len(self.properties_df) == 0:
            print("No data to export!")
            return
        
        try:
            # Try to import openpyxl
            import openpyxl
            excel_path = os.path.join(self.output_dir, 'island_property_comparison.xlsx')
            
            # Create Excel writer
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # Sheet 1: Summary statistics
                summary_data = []
                for island, stats in self.island_property_stats.items():
                    row = {
                        'Island': island,
                        'Total Properties': stats['total_properties']
                    }
                    
                    # Add counts for each accommodation type
                    for accom_type in self.accommodation_types.keys():
                        row[f'{accom_type.title()} Count'] = stats['type_counts'].get(accom_type, 0)
                        row[f'{accom_type.title()} Avg Rating'] = stats['avg_ratings_by_type'].get(accom_type, 0)
                    
                    summary_data.append(row)
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Sheet 2: Raw property data with island and type classifications
                self.properties_df.to_excel(writer, sheet_name='All Properties', index=False)
                
                # Sheet 3: Pivot table of types by island
                type_pivot = pd.pivot_table(
                    self.properties_df,
                    values='property_id',
                    index=['island_category'],
                    columns=['accommodation_type'],
                    aggfunc='count',
                    fill_value=0
                )
                type_pivot.to_excel(writer, sheet_name='Type Counts')
                
                # Sheet 4: Pivot table of average ratings
                rating_pivot = pd.pivot_table(
                    self.properties_df,
                    values='avg_rating',
                    index=['island_category'],
                    columns=['accommodation_type'],
                    aggfunc='mean',
                    fill_value=0
                )
                rating_pivot.to_excel(writer, sheet_name='Average Ratings')
                
                # Sheet 5: Top properties by island
                top_properties = []
                for island, stats in self.island_property_stats.items():
                    for prop in stats.get('top_properties', []):
                        prop_dict = {
                            'Island': island,
                            'Property Name': prop.get('property_name', ''),
                            'Average Rating': prop.get('avg_rating', 0),
                            'Review Count': prop.get('review_count', 0),
                            'Accommodation Type': prop.get('accommodation_type', '')
                        }
                        top_properties.append(prop_dict)
                
                if top_properties:
                    top_df = pd.DataFrame(top_properties)
                    top_df.to_excel(writer, sheet_name='Top Properties', index=False)
            
            print(f"Analysis exported to Excel: {excel_path}")
            
        except ImportError:
            print("\nopenpyxl not found. Cannot export to Excel.")
            print("To export to Excel, install openpyxl: pip install openpyxl")
            
            # Export to CSV instead
            csv_path = os.path.join(self.output_dir, 'island_property_summary.csv')
            
            # Prepare summary data
            summary_data = []
            for island, stats in self.island_property_stats.items():
                row = {
                    'Island': island,
                    'Total Properties': stats['total_properties']
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
        Run the complete island property analysis.
        
        Parameters:
        - top_n_islands: Number of top islands to analyze by property count
        """
        print("Starting Tonga Island Property Analysis...")
        
        # Load and process data
        if self.load_and_process_data():
            # Analyze by island and property type
            self.analyze_by_island_and_type(top_n_islands=top_n_islands)
            
            # Generate visualizations
            self.generate_visualizations()
            
            # Export to Excel
            self.export_to_excel()
            
            print("\nIsland property analysis complete!")
        else:
            print("Analysis aborted due to data loading issues.")


if __name__ == "__main__":
    # Create and run the island property analyzer
    analyzer = IslandPropertyAnalyzer()
    analyzer.run_analysis(top_n_islands=5)