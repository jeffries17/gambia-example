import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter

class IslandBasedAnalyzer:
    """
    Analyzer for Tonga tourism reviews with focus on island-specific metrics.
    This class analyzes reviews from accommodations, restaurants, and attractions
    and categorizes them by island.
    """
    
    def __init__(self, data_dir='data', output_dir='outputs'):
        """
        Initialize the island-based analyzer.
        
        Parameters:
        - data_dir: Directory containing the JSON review files
        - output_dir: Directory to save analysis outputs
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
            
        # Island-specific data storage
        self.all_reviews_df = None
        self.reviews_by_island = {}
        self.island_stats = {}
        
        # Review counts
        self.review_counts = {
            'accommodations': 0,
            'restaurants': 0,
            'attractions': 0,
            'total': 0
        }

    def load_json_file(self, filepath):
        """
        Load and validate a JSON file of reviews.
        
        Parameters:
        - filepath: Path to the JSON file
        
        Returns:
        - List of review dictionaries
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Ensure data is a list
            if not isinstance(data, list):
                data = [data]
                
            print(f"Successfully loaded {len(data)} reviews from {os.path.basename(filepath)}")
            return data
            
        except Exception as e:
            print(f"Error loading {filepath}: {str(e)}")
            return []

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

    def extract_basic_fields(self, reviews, category):
        """
        Extract key fields from reviews into a DataFrame with island extraction.
        
        Parameters:
        - reviews: List of review dictionaries
        - category: Category of the reviews (accommodations, restaurants, attractions)
        
        Returns:
        - DataFrame with standardized fields including island information
        """
        processed_reviews = []
        
        for review in reviews:
            try:
                # Extract location string
                location_string = review.get('placeInfo', {}).get('locationString', '')
                
                # Extract island from location string
                island = self.standardize_island_name(location_string)
                
                # Extract basic fields with error handling
                processed_review = {
                    'id': review.get('id'),
                    'category': category,
                    'rating': review.get('rating'),
                    'published_date': review.get('publishedDate'),
                    'trip_type': review.get('tripType'),
                    'text': review.get('text', ''),
                    'title': review.get('title', ''),
                    'helpful_votes': review.get('helpfulVotes', 0),
                    'place_name': review.get('placeInfo', {}).get('name', ''),
                    'location_string': location_string,
                    'island': island,
                    'city': self.extract_city_from_location(location_string)
                }
                
                # Extract user location if available
                user_data = review.get('user', {})
                user_location = user_data.get('userLocation', {})
                if isinstance(user_location, dict):
                    processed_review['user_location'] = user_location.get('name', '')
                else:
                    processed_review['user_location'] = ''
                
                processed_reviews.append(processed_review)
                
            except Exception as e:
                print(f"Error processing review {review.get('id')}: {str(e)}")
                continue
        
        # Convert to DataFrame
        if processed_reviews:
            df = pd.DataFrame(processed_reviews)
            
            # Convert date to datetime
            df['published_date'] = pd.to_datetime(df['published_date'], errors='coerce')
            
            # Convert rating to numeric
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
            
            return df
        else:
            return pd.DataFrame()

    def load_data(self):
        """
        Load all review data from JSON files with island categorization.
        """
        print("Loading review data for island-based analysis...")
        
        dfs = []
        
        # Load accommodations
        accommodations_path = os.path.join(self.data_dir, 'tonga_accommodations.json')
        if os.path.exists(accommodations_path):
            accommodations = self.load_json_file(accommodations_path)
            accommodations_df = self.extract_basic_fields(accommodations, 'accommodation')
            self.review_counts['accommodations'] = len(accommodations_df)
            dfs.append(accommodations_df)
        
        # Load restaurants
        restaurants_path = os.path.join(self.data_dir, 'tonga_restaurants.json')
        if os.path.exists(restaurants_path):
            restaurants = self.load_json_file(restaurants_path)
            restaurants_df = self.extract_basic_fields(restaurants, 'restaurant')
            self.review_counts['restaurants'] = len(restaurants_df)
            dfs.append(restaurants_df)
        
        # Load attractions
        attractions_path = os.path.join(self.data_dir, 'tonga_attractions.json')
        if os.path.exists(attractions_path):
            attractions = self.load_json_file(attractions_path)
            attractions_df = self.extract_basic_fields(attractions, 'attraction')
            self.review_counts['attractions'] = len(attractions_df)
            dfs.append(attractions_df)
            
        if dfs:
            self.all_reviews_df = pd.concat(dfs, ignore_index=True)
            self.review_counts['total'] = len(self.all_reviews_df)
            
            print("\nData loading complete!")
            print(f"Total reviews loaded: {self.review_counts['total']}")
            print(f"- Accommodations: {self.review_counts['accommodations']}")
            print(f"- Restaurants: {self.review_counts['restaurants']}")
            print(f"- Attractions: {self.review_counts['attractions']}")
        else:
            print("No review data loaded!")

    def analyze_islands(self, top_n=10):
        """
        Analyze reviews by island and generate statistics for top N islands.
        
        Parameters:
        - top_n: Number of top islands to analyze by review count
        """
        if self.all_reviews_df is None or len(self.all_reviews_df) == 0:
            print("No data available for island analysis!")
            return
        
        print(f"\nAnalyzing reviews by island (focusing on top {top_n} islands)...")
        
        # Count reviews by island
        island_counts = self.all_reviews_df['island'].value_counts()
        
        # Get top N islands
        top_islands = island_counts.head(top_n).index.tolist()
        print(f"\nTop {top_n} islands by review count:")
        for i, island in enumerate(top_islands, 1):
            print(f"{i}. {island}: {island_counts[island]} reviews")
        
        # Create 'other' category for remaining islands
        self.all_reviews_df['island_category'] = self.all_reviews_df['island'].apply(
            lambda x: x if x in top_islands else 'Other'
        )
        
        # Analyze reviews by island and category
        self.island_stats = {}
        
        # Overall island statistics
        for island in top_islands + ['Other']:
            island_data = self.all_reviews_df[self.all_reviews_df['island_category'] == island]
            
            # Count by category
            category_counts = island_data['category'].value_counts().to_dict()
            
            # Average ratings
            avg_rating_overall = island_data['rating'].mean()
            avg_ratings_by_category = island_data.groupby('category')['rating'].mean().to_dict()
            
            # Store stats
            self.island_stats[island] = {
                'total_reviews': len(island_data),
                'category_counts': category_counts,
                'avg_rating_overall': avg_rating_overall,
                'avg_ratings_by_category': avg_ratings_by_category,
                'top_locations': island_data['location_string'].value_counts().head(5).to_dict()
            }
        
        # Save the statistics
        stats_file = os.path.join(self.output_dir, 'island_stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            # Convert any non-serializable objects to strings
            serializable_stats = self._make_serializable(self.island_stats)
            json.dump(serializable_stats, f, indent=2)
        
        print(f"\nIsland statistics saved to {stats_file}")

    def _make_serializable(self, obj):
        """Helper function to make objects JSON serializable"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(x) for x in obj]
        elif pd.isna(obj):
            return None
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)

    def generate_island_visualizations(self):
        """
        Generate visualizations of review distribution by island and category.
        """
        if self.all_reviews_df is None or len(self.all_reviews_df) == 0:
            print("No data available for visualizations!")
            return
        
        # Create visualizations directory
        viz_dir = os.path.join(self.output_dir, 'island_visualizations')
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        
        # Set style
        sns.set(style="whitegrid")
        
        # 1. Total reviews by island
        plt.figure(figsize=(12, 8))
        island_counts = self.all_reviews_df['island_category'].value_counts()
        ax = sns.barplot(x=island_counts.index, y=island_counts.values)
        plt.title('Total Reviews by Island', fontsize=16)
        plt.ylabel('Number of Reviews', fontsize=14)
        plt.xlabel('Island', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'total_reviews_by_island.png'), dpi=300)
        plt.close()
        
        # 2. Reviews by island and category
        island_category_counts = pd.crosstab(
            self.all_reviews_df['island_category'], 
            self.all_reviews_df['category']
        )
        
        # Plot stacked bar chart
        plt.figure(figsize=(14, 8))
        island_category_counts.plot(kind='bar', stacked=True, colormap='viridis', figsize=(14, 8))
        plt.title('Reviews by Island and Category', fontsize=16)
        plt.ylabel('Number of Reviews', fontsize=14)
        plt.xlabel('Island', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.legend(title='Category', fontsize=12, title_fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'reviews_by_island_and_category.png'), dpi=300)
        plt.close()
        
        # 3. Average rating by island
        plt.figure(figsize=(12, 8))
        avg_ratings = self.all_reviews_df.groupby('island_category')['rating'].mean().sort_values(ascending=False)
        ax = sns.barplot(x=avg_ratings.index, y=avg_ratings.values)
        plt.title('Average Rating by Island', fontsize=16)
        plt.ylabel('Average Rating', fontsize=14)
        plt.xlabel('Island', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.ylim(0, 5)  # Assuming ratings are on a 0-5 scale
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'avg_rating_by_island.png'), dpi=300)
        plt.close()
        
        # 4. Average rating by island and category
        plt.figure(figsize=(14, 8))
        avg_ratings_by_category = self.all_reviews_df.groupby(['island_category', 'category'])['rating'].mean().unstack()
        avg_ratings_by_category.plot(kind='bar', colormap='viridis', figsize=(14, 8))
        plt.title('Average Rating by Island and Category', fontsize=16)
        plt.ylabel('Average Rating', fontsize=14)
        plt.xlabel('Island', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.legend(title='Category', fontsize=12, title_fontsize=14)
        plt.ylim(0, 5)  # Assuming ratings are on a 0-5 scale
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'avg_rating_by_island_and_category.png'), dpi=300)
        plt.close()
        
        # 5. Proportion of categories by island (normalized)
        plt.figure(figsize=(14, 8))
        island_category_prop = island_category_counts.div(island_category_counts.sum(axis=1), axis=0)
        island_category_prop.plot(kind='bar', stacked=True, colormap='viridis', figsize=(14, 8))
        plt.title('Proportion of Review Categories by Island', fontsize=16)
        plt.ylabel('Proportion', fontsize=14)
        plt.xlabel('Island', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.legend(title='Category', fontsize=12, title_fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'category_proportion_by_island.png'), dpi=300)
        plt.close()
        
        print(f"\nIsland visualizations saved to {viz_dir}")

    def export_to_excel(self):
        """
        Export the island analysis results to Excel for further use.
        If openpyxl is not available, export to CSV files instead.
        """
        if self.all_reviews_df is None or len(self.all_reviews_df) == 0:
            print("No data available for export!")
            return
        
        try:
            # Try to import openpyxl
            import openpyxl
            use_excel = True
        except ImportError:
            print("\nopenpyxl not found. Exporting to CSV files instead.")
            print("To export to Excel, install openpyxl: pip install openpyxl")
            use_excel = False
        
        if use_excel:
            # Create Excel file
            excel_path = os.path.join(self.output_dir, 'tonga_island_analysis.xlsx')
            
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # Sheet 1: Overall summary
                summary_data = []
                for island, stats in self.island_stats.items():
                    row = {
                        'Island': island,
                        'Total Reviews': stats['total_reviews'],
                        'Average Rating': stats['avg_rating_overall'],
                        'Accommodations': stats['category_counts'].get('accommodation', 0),
                        'Restaurants': stats['category_counts'].get('restaurant', 0),
                        'Attractions': stats['category_counts'].get('attraction', 0)
                    }
                    summary_data.append(row)
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Sheet 2: Full review data
                self.all_reviews_df.to_excel(writer, sheet_name='All Reviews', index=False)
                
                # Sheet 3: Reviews by island
                island_pivot = pd.pivot_table(
                    self.all_reviews_df,
                    values='id',
                    index=['island_category'],
                    columns=['category'],
                    aggfunc='count',
                    fill_value=0
                )
                island_pivot.to_excel(writer, sheet_name='Island Pivot')
                
                # Sheet 4: Average ratings
                rating_pivot = pd.pivot_table(
                    self.all_reviews_df,
                    values='rating',
                    index=['island_category'],
                    columns=['category'],
                    aggfunc='mean',
                    fill_value=0
                )
                rating_pivot.to_excel(writer, sheet_name='Rating Pivot')
            
            print(f"\nIsland analysis exported to Excel: {excel_path}")
        else:
            # Export to CSV files instead
            csv_dir = os.path.join(self.output_dir, 'csv_exports')
            if not os.path.exists(csv_dir):
                os.makedirs(csv_dir)
            
            # File 1: Overall summary
            summary_data = []
            for island, stats in self.island_stats.items():
                row = {
                    'Island': island,
                    'Total Reviews': stats['total_reviews'],
                    'Average Rating': stats['avg_rating_overall'],
                    'Accommodations': stats['category_counts'].get('accommodation', 0),
                    'Restaurants': stats['category_counts'].get('restaurant', 0),
                    'Attractions': stats['category_counts'].get('attraction', 0)
                }
                summary_data.append(row)
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(os.path.join(csv_dir, 'summary.csv'), index=False)
            
            # File 2: Full review data
            self.all_reviews_df.to_csv(os.path.join(csv_dir, 'all_reviews.csv'), index=False)
            
            # File 3: Reviews by island
            island_pivot = pd.pivot_table(
                self.all_reviews_df,
                values='id',
                index=['island_category'],
                columns=['category'],
                aggfunc='count',
                fill_value=0
            )
            island_pivot.to_csv(os.path.join(csv_dir, 'island_pivot.csv'))
            
            # File 4: Average ratings
            rating_pivot = pd.pivot_table(
                self.all_reviews_df,
                values='rating',
                index=['island_category'],
                columns=['category'],
                aggfunc='mean',
                fill_value=0
            )
            rating_pivot.to_csv(os.path.join(csv_dir, 'rating_pivot.csv'))
            
            print(f"\nIsland analysis exported to CSV files in: {csv_dir}")

    def run_analysis(self, top_n=10):
        """
        Run the complete island-based analysis pipeline.
        
        Parameters:
        - top_n: Number of top islands to analyze by review count
        """
        print("Starting Tonga tourism island-based review analysis...")
        
        # Load data
        self.load_data()
        
        # Analyze by island
        self.analyze_islands(top_n=top_n)
        
        # Generate visualizations
        self.generate_island_visualizations()
        
        # Export to Excel
        self.export_to_excel()
        
        print("\nIsland-based analysis complete!")


if __name__ == "__main__":
    # Create and run the island analyzer
    analyzer = IslandBasedAnalyzer(data_dir='data', output_dir='outputs')
    analyzer.run_analysis(top_n=10)