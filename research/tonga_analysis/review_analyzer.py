import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from collections import Counter

class TongaReviewAnalyzer:
    """
    Base class for analyzing Tonga tourism reviews from TripAdvisor.
    """
    
    def __init__(self, data_dir='data', output_dir='outputs'):
        """
        Initialize the analyzer with data and output directories.
        
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
            
        # Initialize data storage
        self.accommodations_df = None
        self.restaurants_df = None
        self.attractions_df = None
        self.all_reviews_df = None
        
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

    def extract_basic_fields(self, reviews, category):
        """
        Extract key fields from reviews into a DataFrame.
        
        Parameters:
        - reviews: List of review dictionaries
        - category: Category of the reviews (accommodations, restaurants, attractions)
        
        Returns:
        - DataFrame with standardized fields
        """
        processed_reviews = []
        
        for review in reviews:
            try:
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
                    'place_location': review.get('placeInfo', {}).get('locationString', '')
                }
                
                # Extract user location if available
                user_data = review.get('user', {})
                user_location = user_data.get('userLocation', {})
                if isinstance(user_location, dict):
                    processed_review['user_location'] = user_location.get('name', '')
                else:
                    processed_review['user_location'] = ''
                
                # Extract subratings if available
                subratings = review.get('subratings', [])
                if subratings:
                    for subrating in subratings:
                        name = subrating.get('name', '').lower().replace(' ', '_')
                        value = subrating.get('value')
                        if name and value is not None:
                            processed_review[f'subrating_{name}'] = value
                
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
        Load all review data from JSON files.
        """
        print("Loading review data...")
        
        # Load accommodations
        accommodations_path = os.path.join(self.data_dir, 'tonga_accommodations.json')
        if os.path.exists(accommodations_path):
            accommodations = self.load_json_file(accommodations_path)
            self.accommodations_df = self.extract_basic_fields(accommodations, 'accommodation')
            self.review_counts['accommodations'] = len(self.accommodations_df)
        
        # Load restaurants
        restaurants_path = os.path.join(self.data_dir, 'tonga_restaurants.json')
        if os.path.exists(restaurants_path):
            restaurants = self.load_json_file(restaurants_path)
            self.restaurants_df = self.extract_basic_fields(restaurants, 'restaurant')
            self.review_counts['restaurants'] = len(self.restaurants_df)
        
        # Load attractions
        attractions_path = os.path.join(self.data_dir, 'tonga_attractions.json')
        if os.path.exists(attractions_path):
            attractions = self.load_json_file(attractions_path)
            self.attractions_df = self.extract_basic_fields(attractions, 'attraction')
            self.review_counts['attractions'] = len(self.attractions_df)
        
        # Combine all reviews
        dfs = []
        if self.accommodations_df is not None:
            dfs.append(self.accommodations_df)
        if self.restaurants_df is not None:
            dfs.append(self.restaurants_df)
        if self.attractions_df is not None:
            dfs.append(self.attractions_df)
            
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

    def generate_basic_stats(self):
        """
        Generate basic statistics about the reviews.
        """
        if self.all_reviews_df is None or len(self.all_reviews_df) == 0:
            print("No data available for analysis!")
            return
        
        stats = {
            'review_counts': self.review_counts,
            'rating_stats': {},
            'temporal_stats': {},
            'trip_type_stats': {},
            'location_stats': {}
        }
        
        # Rating statistics
        rating_stats = self.all_reviews_df['rating'].describe()
        stats['rating_stats'] = {
            'mean': rating_stats['mean'],
            'median': rating_stats['50%'],
            'std': rating_stats['std'],
            'min': rating_stats['min'],
            'max': rating_stats['max']
        }
        
        # Temporal statistics
        if 'published_date' in self.all_reviews_df.columns:
            date_stats = self.all_reviews_df['published_date'].describe()
            stats['temporal_stats'] = {
                'earliest': date_stats['min'],
                'latest': date_stats['max'],
                'date_range_days': (date_stats['max'] - date_stats['min']).days
            }
        
        # Trip type distribution
        if 'trip_type' in self.all_reviews_df.columns:
            trip_type_counts = self.all_reviews_df['trip_type'].value_counts()
            stats['trip_type_stats'] = trip_type_counts.to_dict()
        
        # Location statistics
        if 'user_location' in self.all_reviews_df.columns:
            location_counts = self.all_reviews_df['user_location'].value_counts()
            stats['location_stats'] = location_counts.head(10).to_dict()
        
        # Save stats
        stats_file = os.path.join(self.output_dir, 'basic_stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            # Convert any non-serializable objects to strings
            serializable_stats = self._make_serializable(stats)
            json.dump(serializable_stats, f, indent=2)
        
        print(f"\nBasic statistics saved to {stats_file}")
        return stats

    def _make_serializable(self, obj):
        """Helper function to make objects JSON serializable"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(x) for x in obj]
        elif pd.isna(obj):
            return None
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)

    def generate_basic_visualizations(self):
        """
        Generate basic visualizations of the review data.
        """
        if self.all_reviews_df is None or len(self.all_reviews_df) == 0:
            print("No data available for visualizations!")
            return
        
        # Create visualizations directory
        viz_dir = os.path.join(self.output_dir, 'visualizations')
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        
        # 1. Review counts by category
        plt.figure(figsize=(10, 6))
        category_counts = self.all_reviews_df['category'].value_counts()
        sns.barplot(x=category_counts.index, y=category_counts.values)
        plt.title('Number of Reviews by Category')
        plt.ylabel('Number of Reviews')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'review_counts_by_category.png'))
        plt.close()
        
        # 2. Rating distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.all_reviews_df, x='rating', bins=5)
        plt.title('Distribution of Ratings')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'rating_distribution.png'))
        plt.close()
        
        # 3. Rating distribution by category
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=self.all_reviews_df, x='category', y='rating')
        plt.title('Rating Distribution by Category')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'rating_by_category.png'))
        plt.close()
        
        # 4. Reviews over time
        if 'published_date' in self.all_reviews_df.columns:
            plt.figure(figsize=(12, 6))
            reviews_by_date = self.all_reviews_df.groupby(['published_date', 'category']).size().unstack()
            reviews_by_date.plot(kind='line', marker='o')
            plt.title('Reviews Over Time by Category')
            plt.xlabel('Date')
            plt.ylabel('Number of Reviews')
            plt.legend(title='Category')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'reviews_over_time.png'))
            plt.close()
        
        # 5. Trip type distribution
        if 'trip_type' in self.all_reviews_df.columns:
            plt.figure(figsize=(10, 6))
            trip_type_counts = self.all_reviews_df['trip_type'].value_counts()
            sns.barplot(x=trip_type_counts.index, y=trip_type_counts.values)
            plt.title('Reviews by Trip Type')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'reviews_by_trip_type.png'))
            plt.close()
        
        print(f"\nBasic visualizations saved to {viz_dir}")

    def run_analysis(self):
        """
        Run the complete analysis pipeline.
        """
        print("Starting Tonga tourism review analysis...")
        
        # Load data
        self.load_data()
        
        # Generate statistics
        stats = self.generate_basic_stats()
        
        # Generate visualizations
        self.generate_basic_visualizations()
        
        print("\nAnalysis complete!")
        return stats