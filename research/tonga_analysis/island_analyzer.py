#!/usr/bin/env python3
"""
Comprehensive island analysis script for Tonga tourism data.
This consolidated script combines functionality from:
1. island_accommodation_analysis.py (review-based analysis)
2. island_property_analysis.py (property-based analysis)

It provides a complete view of both review counts and unique property counts
across different islands and accommodation types.
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
import numpy as np
from datetime import datetime
from tonga_analysis.visualization_styles import (
    ISLAND_COLORS, ISLAND_COLORS_LOWER, set_visualization_style,
    get_island_palette, apply_island_style
)

class IslandAnalyzer:
    """
    Comprehensive analyzer for tourism data across different islands in Tonga.
    Provides both review-based and property-based analyses in a single interface.
    """
    
    def __init__(self, data_dir='tonga_data', output_dir=None):
        """
        Initialize the island analyzer.
        
        Parameters:
        - data_dir: Directory containing the data JSON files
        - output_dir: Directory to save analysis outputs
        """
        self.data_dir = data_dir
        
        # Set default output directory to the standardized location
        if output_dir is None:
            # Use the parent directory's outputs folder
            parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            self.output_dir = os.path.join(parent_dir, "outputs", "island_analysis")
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
        self.properties_df = None  # Will hold unique property data
        
        # Analysis results storage
        self.island_stats = {}
        self.property_stats = {}
        self.overall_stats = {}
        
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
        if not isinstance(location_string, str) or not location_string:
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
        if not isinstance(location_string, str) or not location_string:
            return 'Unknown'
        
        # Common pattern: "City, Island"
        parts = location_string.split(',')
        
        if len(parts) >= 1:
            # Extract the city part (before the comma)
            return parts[0].strip()
        else:
            return 'Unknown'

    def detect_accommodation_type(self, review_text='', title='', place_name=''):
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

    def load_accommodation_data(self):
        """
        Load accommodation data with island and type classification.
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
            properties = {}  # For tracking unique properties
            
            for review in accommodations:
                try:
                    # Extract location string
                    location_string = review.get('placeInfo', {}).get('locationString', '')
                    
                    # Extract island and city from location string
                    island = self.standardize_island_name(location_string)
                    city = self.extract_city_from_location(location_string)
                    
                    # Extract place name and ID
                    place_info = review.get('placeInfo', {})
                    place_name = place_info.get('name', '')
                    place_id = place_info.get('id', 'unknown')
                    
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
                        'place_id': place_id,
                        'place_name': place_name,
                        'location_string': location_string,
                        'island': island,
                        'city': city,
                        'accommodation_type': accommodation_type,
                        'category': 'accommodation'
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
                    
                    # Also track property info for property-based analysis
                    if place_id != 'unknown' and place_id not in properties:
                        properties[place_id] = {
                            'property_id': place_id,
                            'property_name': place_name,
                            'island': island,
                            'city': city,
                            'accommodation_type': accommodation_type,
                            'location_string': location_string,
                            'review_count': 0,
                            'avg_rating': 0.0,
                            'ratings': [],
                            'category': 'accommodation'
                        }
                    
                    # Update property review stats
                    if place_id in properties and review.get('rating') is not None:
                        properties[place_id]['review_count'] += 1
                        properties[place_id]['ratings'].append(review.get('rating'))
                    
                except Exception as e:
                    print(f"Error processing review {review.get('id')}: {str(e)}")
                    continue
            
            # Calculate average ratings for properties
            for place_id, prop in properties.items():
                if prop['ratings']:
                    prop['avg_rating'] = sum(prop['ratings']) / len(prop['ratings'])
                # Remove ratings list to clean up data
                del prop['ratings']
            
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
                
                # Save properties data
                self.properties_df = pd.DataFrame(list(properties.values()))
                print(f"Processed {len(self.properties_df)} unique properties")
                
                # Get unique islands for reference
                print(f"\nUnique islands found: {', '.join(self.accommodations_df['island'].unique())}")
                
                return True
            else:
                print("No accommodation reviews could be processed!")
                return False
                
        except Exception as e:
            print(f"Error loading accommodation data: {str(e)}")
            return False

    def load_restaurant_data(self):
        """
        Load restaurant data with island classification.
        """
        restaurants_path = os.path.join(self.data_dir, 'tonga_restaurants.json')
        
        if not os.path.exists(restaurants_path):
            print(f"Error: Restaurant data file not found at {restaurants_path}")
            return False
        
        try:
            with open(restaurants_path, 'r', encoding='utf-8') as f:
                restaurants = json.load(f)
                
            print(f"Successfully loaded {len(restaurants)} restaurant reviews")
            
            # Process restaurant reviews
            processed_restaurants = []
            restaurant_properties = {}  # For tracking unique restaurants
            
            for review in restaurants:
                try:
                    # Extract location string
                    location_string = review.get('placeInfo', {}).get('locationString', '')
                    
                    # Extract island and city from location string
                    island = self.standardize_island_name(location_string)
                    city = self.extract_city_from_location(location_string)
                    
                    # Extract place name and ID
                    place_info = review.get('placeInfo', {})
                    place_name = place_info.get('name', '')
                    place_id = place_info.get('id', 'unknown')
                    
                    # Create processed review
                    processed_review = {
                        'id': review.get('id'),
                        'rating': review.get('rating'),
                        'published_date': review.get('publishedDate'),
                        'trip_type': review.get('tripType'),
                        'text': review.get('text', ''),
                        'title': review.get('title', ''),
                        'helpful_votes': review.get('helpfulVotes', 0),
                        'place_id': place_id,
                        'place_name': place_name,
                        'location_string': location_string,
                        'island': island,
                        'city': city,
                        'category': 'restaurant'
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
                    
                    processed_restaurants.append(processed_review)
                    
                    # Also track property info for property-based analysis
                    if place_id != 'unknown' and place_id not in restaurant_properties:
                        restaurant_properties[place_id] = {
                            'property_id': place_id,
                            'property_name': place_name,
                            'island': island,
                            'city': city,
                            'location_string': location_string,
                            'review_count': 0,
                            'avg_rating': 0.0,
                            'ratings': [],
                            'category': 'restaurant'
                        }
                    
                    # Update property review stats
                    if place_id in restaurant_properties and review.get('rating') is not None:
                        restaurant_properties[place_id]['review_count'] += 1
                        restaurant_properties[place_id]['ratings'].append(review.get('rating'))
                    
                except Exception as e:
                    print(f"Error processing review {review.get('id')}: {str(e)}")
                    continue
            
            # Calculate average ratings for properties
            for place_id, prop in restaurant_properties.items():
                if prop['ratings']:
                    prop['avg_rating'] = sum(prop['ratings']) / len(prop['ratings'])
                # Remove ratings list to clean up data
                del prop['ratings']
            
            # Convert to DataFrame
            if processed_restaurants:
                self.restaurants_df = pd.DataFrame(processed_restaurants)
                
                # Convert date to datetime
                self.restaurants_df['published_date'] = pd.to_datetime(
                    self.restaurants_df['published_date'], errors='coerce'
                )
                
                # Convert rating to numeric
                self.restaurants_df['rating'] = pd.to_numeric(
                    self.restaurants_df['rating'], errors='coerce'
                )
                
                print(f"Processed {len(self.restaurants_df)} restaurant reviews")
                
                # Add restaurant properties to main properties DataFrame
                if self.properties_df is None:
                    self.properties_df = pd.DataFrame(list(restaurant_properties.values()))
                else:
                    self.properties_df = pd.concat([
                        self.properties_df, 
                        pd.DataFrame(list(restaurant_properties.values()))
                    ], ignore_index=True)
                
                # Get unique islands for reference
                print(f"\nUnique islands for restaurants: {', '.join(self.restaurants_df['island'].unique())}")
                
                return True
            else:
                print("No restaurant reviews could be processed!")
                return False
                
        except Exception as e:
            print(f"Error loading restaurant data: {str(e)}")
            return False

    def load_attraction_data(self):
        """
        Load attraction data with island classification.
        """
        attractions_path = os.path.join(self.data_dir, 'tonga_attractions.json')
        
        if not os.path.exists(attractions_path):
            print(f"Error: Attraction data file not found at {attractions_path}")
            return False
        
        try:
            with open(attractions_path, 'r', encoding='utf-8') as f:
                attractions = json.load(f)
                
            print(f"Successfully loaded {len(attractions)} attraction reviews")
            
            # Process attraction reviews
            processed_attractions = []
            attraction_properties = {}  # For tracking unique attractions
            
            for review in attractions:
                try:
                    # Extract location string
                    location_string = review.get('placeInfo', {}).get('locationString', '')
                    
                    # Extract island and city from location string
                    island = self.standardize_island_name(location_string)
                    city = self.extract_city_from_location(location_string)
                    
                    # Extract place name and ID
                    place_info = review.get('placeInfo', {})
                    place_name = place_info.get('name', '')
                    place_id = place_info.get('id', 'unknown')
                    
                    # Create processed review
                    processed_review = {
                        'id': review.get('id'),
                        'rating': review.get('rating'),
                        'published_date': review.get('publishedDate'),
                        'trip_type': review.get('tripType'),
                        'text': review.get('text', ''),
                        'title': review.get('title', ''),
                        'helpful_votes': review.get('helpfulVotes', 0),
                        'place_id': place_id,
                        'place_name': place_name,
                        'location_string': location_string,
                        'island': island,
                        'city': city,
                        'category': 'attraction'
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
                    
                    processed_attractions.append(processed_review)
                    
                    # Also track property info for property-based analysis
                    if place_id != 'unknown' and place_id not in attraction_properties:
                        attraction_properties[place_id] = {
                            'property_id': place_id,
                            'property_name': place_name,
                            'island': island,
                            'city': city,
                            'location_string': location_string,
                            'review_count': 0,
                            'avg_rating': 0.0,
                            'ratings': [],
                            'category': 'attraction'
                        }
                    
                    # Update property review stats
                    if place_id in attraction_properties and review.get('rating') is not None:
                        attraction_properties[place_id]['review_count'] += 1
                        attraction_properties[place_id]['ratings'].append(review.get('rating'))
                    
                except Exception as e:
                    print(f"Error processing review {review.get('id')}: {str(e)}")
                    continue
            
            # Calculate average ratings for properties
            for place_id, prop in attraction_properties.items():
                if prop['ratings']:
                    prop['avg_rating'] = sum(prop['ratings']) / len(prop['ratings'])
                # Remove ratings list to clean up data
                del prop['ratings']
            
            # Convert to DataFrame
            if processed_attractions:
                self.attractions_df = pd.DataFrame(processed_attractions)
                
                # Convert date to datetime
                self.attractions_df['published_date'] = pd.to_datetime(
                    self.attractions_df['published_date'], errors='coerce'
                )
                
                # Convert rating to numeric
                self.attractions_df['rating'] = pd.to_numeric(
                    self.attractions_df['rating'], errors='coerce'
                )
                
                print(f"Processed {len(self.attractions_df)} attraction reviews")
                
                # Add attraction properties to main properties DataFrame
                if self.properties_df is None:
                    self.properties_df = pd.DataFrame(list(attraction_properties.values()))
                else:
                    self.properties_df = pd.concat([
                        self.properties_df, 
                        pd.DataFrame(list(attraction_properties.values()))
                    ], ignore_index=True)
                
                # Get unique islands for reference
                print(f"\nUnique islands for attractions: {', '.join(self.attractions_df['island'].unique())}")
                
                return True
            else:
                print("No attraction reviews could be processed!")
                return False
                
        except Exception as e:
            print(f"Error loading attraction data: {str(e)}")
            return False

    def load_data(self):
        """Load all available data for island analysis."""
        print("Loading data for island analysis...")
        
        # Load each data type
        accom_success = self.load_accommodation_data()
        rest_success = self.load_restaurant_data()
        attr_success = self.load_attraction_data()
        
        # Combine all reviews into a single DataFrame if needed
        if any([accom_success, rest_success, attr_success]):
            # Combine all DataFrames
            dataframes = []
            
            if self.accommodations_df is not None:
                dataframes.append(self.accommodations_df)
            
            if self.restaurants_df is not None:
                dataframes.append(self.restaurants_df)
            
            if self.attractions_df is not None:
                dataframes.append(self.attractions_df)
            
            if dataframes:
                self.all_reviews_df = pd.concat(dataframes, ignore_index=True)
                print(f"Combined {len(self.all_reviews_df)} reviews for analysis")
                return True
        
        print("No data was loaded successfully!")
        return False

    def analyze_islands(self, top_n=5):
        """
        Analyze islands based on reviews and properties.
        
        Parameters:
        - top_n: Number of top islands to focus on (by review count)
        """
        if not hasattr(self, 'all_reviews_df') or self.all_reviews_df is None:
            print("No review data available! Run load_data() first.")
            return
        
        print("\nAnalyzing islands...")
        
        # Count reviews by island
        island_counts = self.all_reviews_df['island'].value_counts()
        
        # Get top N islands
        top_islands = island_counts.head(top_n).index.tolist()
        print(f"\nTop {len(top_islands)} islands by review count:")
        for i, island in enumerate(top_islands, 1):
            print(f"{i}. {island}: {island_counts[island]} reviews")
        
        # Create 'Other' category for remaining islands
        self.all_reviews_df['island_category'] = self.all_reviews_df['island'].apply(
            lambda x: x if x in top_islands else 'Other'
        )
        
        if self.properties_df is not None:
            self.properties_df['island_category'] = self.properties_df['island'].apply(
                lambda x: x if x in top_islands else 'Other'
            )
        
        # Analysis by island
        for island in top_islands + ['Other']:
            # Get all reviews for this island
            island_reviews = self.all_reviews_df[self.all_reviews_df['island_category'] == island]
            
            # Count by category
            category_counts = island_reviews['category'].value_counts().to_dict()
            
            # Average ratings by category
            rating_by_category = island_reviews.groupby('category')['rating'].mean().to_dict()
            
            # Review counts by category
            reviews_by_category = island_reviews.groupby('category')['id'].count().to_dict()
            
            # Store in island_stats
            self.island_stats[island] = {
                'total_reviews': len(island_reviews),
                'category_counts': category_counts,
                'avg_ratings_by_category': rating_by_category,
                'reviews_by_category': reviews_by_category,
                'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # If we have accommodation data, analyze by type
            if 'accommodation_type' in island_reviews.columns:
                accom_reviews = island_reviews[island_reviews['category'] == 'accommodation']
                if not accom_reviews.empty:
                    # Count by accommodation type
                    type_counts = accom_reviews['accommodation_type'].value_counts().to_dict()
                    
                    # Average ratings by accommodation type
                    avg_ratings_by_type = accom_reviews.groupby('accommodation_type')['rating'].mean().to_dict()
                    
                    # Store in island_stats
                    self.island_stats[island]['accommodation_types'] = type_counts
                    self.island_stats[island]['avg_ratings_by_accommodation_type'] = avg_ratings_by_type
            
            # Property-based analysis
            if self.properties_df is not None:
                island_properties = self.properties_df[self.properties_df['island_category'] == island]
                
                if not island_properties.empty:
                    # Count properties by category
                    property_category_counts = island_properties['category'].value_counts().to_dict()
                    
                    # Average ratings by category
                    property_rating_by_category = island_properties.groupby('category')['avg_rating'].mean().to_dict()
                    
                    # Store property stats
                    self.property_stats[island] = {
                        'total_properties': len(island_properties),
                        'category_counts': property_category_counts,
                        'avg_ratings_by_category': property_rating_by_category
                    }
                    
                    # If we have accommodation data, analyze by type
                    accom_properties = island_properties[island_properties['category'] == 'accommodation']
                    if 'accommodation_type' in island_properties.columns and not accom_properties.empty:
                        # Count by accommodation type
                        property_type_counts = accom_properties['accommodation_type'].value_counts().to_dict()
                        
                        # Average ratings by accommodation type
                        property_avg_ratings_by_type = accom_properties.groupby('accommodation_type')['avg_rating'].mean().to_dict()
                        
                        # Store in property_stats
                        self.property_stats[island]['accommodation_types'] = property_type_counts
                        self.property_stats[island]['avg_ratings_by_accommodation_type'] = property_avg_ratings_by_type
                    
                    # Get top properties by category
                    top_properties = {}
                    for category in island_properties['category'].unique():
                        category_props = island_properties[island_properties['category'] == category]
                        if len(category_props) > 0:
                            # Only include properties with at least 3 reviews
                            filtered_props = category_props[category_props['review_count'] >= 3]
                            if not filtered_props.empty:
                                top_props = (
                                    filtered_props
                                    .sort_values('avg_rating', ascending=False)
                                    .head(5)
                                )
                                top_properties[category] = top_props.to_dict('records')
                    
                    if top_properties:
                        self.property_stats[island]['top_properties'] = top_properties
        
        # Create overall stats
        self.overall_stats = {
            'total_reviews': len(self.all_reviews_df),
            'reviews_by_island': island_counts.to_dict(),
            'reviews_by_category': self.all_reviews_df['category'].value_counts().to_dict(),
            'avg_rating_by_island': self.all_reviews_df.groupby('island')['rating'].mean().to_dict(),
            'avg_rating_by_category': self.all_reviews_df.groupby('category')['rating'].mean().to_dict(),
            'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        if self.properties_df is not None:
            self.overall_stats['total_properties'] = len(self.properties_df)
            self.overall_stats['properties_by_island'] = self.properties_df['island'].value_counts().to_dict()
            self.overall_stats['properties_by_category'] = self.properties_df['category'].value_counts().to_dict()
        
        # Save results
        self._save_results()
        
        print("\nIsland analysis complete!")
        return self.island_stats, self.property_stats, self.overall_stats

    def generate_visualizations(self):
        """Generate comprehensive visualizations for island analysis."""
        if not self.island_stats or not self.overall_stats:
            print("No analysis data available for visualizations! Run analyze_islands() first.")
            return
        
        print("\nGenerating island analysis visualizations...")
        
        # Apply consistent visualization style
        set_visualization_style()
        
        # 1. Total Reviews by Island
        plt.figure(figsize=(12, 8))
        island_review_counts = pd.Series(self.overall_stats['reviews_by_island'])
        island_review_counts = island_review_counts.sort_values(ascending=False)
        
        # Create a DataFrame for easier color mapping
        island_df = pd.DataFrame({
            'island': island_review_counts.index,
            'reviews': island_review_counts.values
        })
        
        # Assign colors based on island names
        island_colors = []
        for island in island_df['island']:
            # Normalize island names for consistent lookup
            island_lower = island.lower().replace("'", "").replace("'", "")
            color = ISLAND_COLORS_LOWER.get(island_lower, '#AAAAAA')  # Gray for unknown islands
            island_colors.append(color)
        
        # Create the bar plot with custom colors
        ax = plt.bar(
            island_df['island'],
            island_df['reviews'],
            color=island_colors
        )
        
        # Apply consistent styling
        apply_island_style(plt.gca(), 
                          title='Total Reviews by Island',
                          x_label='Island', 
                          y_label='Number of Reviews')
        
        plt.xticks(rotation=45, ha='right')
        
        # Add data labels
        for i, v in enumerate(island_df['reviews']):
            plt.text(i, v + 5, str(v), ha='center', fontweight='bold')
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'total_reviews_by_island.png'), dpi=300)
        plt.close()
        
        # 2. Reviews by Island and Category
        plt.figure(figsize=(14, 8))
        
        # Prepare data
        review_data = []
        for island, stats in self.island_stats.items():
            for category, count in stats.get('category_counts', {}).items():
                review_data.append({
                    'Island': island,
                    'Category': category.title(),
                    'Reviews': count
                })
        
        if review_data:
            review_df = pd.DataFrame(review_data)
            
            # Create pivot table
            review_pivot = review_df.pivot(
                index='Island',
                columns='Category',
                values='Reviews'
            ).fillna(0)
            
            # Plot stacked bar chart
            review_pivot.plot(kind='bar', stacked=True, figsize=(14, 8), colormap='viridis')
            plt.title('Reviews by Island and Category', fontsize=16)
            plt.xlabel('Island', fontsize=14)
            plt.ylabel('Number of Reviews', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Category', fontsize=12, title_fontsize=13)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, 'reviews_by_island_and_category.png'), dpi=300)
            plt.close()
            
            # 3. Category Proportion by Island
            plt.figure(figsize=(14, 8))
            
            # Calculate proportions
            prop_pivot = review_pivot.div(review_pivot.sum(axis=1), axis=0)
            
            # Plot stacked bar chart
            prop_pivot.plot(kind='bar', stacked=True, figsize=(14, 8), colormap='viridis')
            plt.title('Category Proportion by Island', fontsize=16)
            plt.xlabel('Island', fontsize=14)
            plt.ylabel('Proportion', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Category', fontsize=12, title_fontsize=13)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, 'category_proportion_by_island.png'), dpi=300)
            plt.close()
        
        # 4. Average Rating by Island
        plt.figure(figsize=(12, 8))
        avg_rating_by_island = pd.Series(self.overall_stats['avg_rating_by_island'])
        avg_rating_by_island = avg_rating_by_island.sort_values(ascending=False)
        
        # Only show islands with significant data
        min_reviews = 10  # Minimum number of reviews needed to include in average rating
        significant_islands = {}
        for island in avg_rating_by_island.index:
            if self.overall_stats['reviews_by_island'].get(island, 0) >= min_reviews:
                significant_islands[island] = avg_rating_by_island[island]
        
        if significant_islands:
            sig_avg_ratings = pd.Series(significant_islands)
            
            bars = plt.bar(
                sig_avg_ratings.index,
                sig_avg_ratings.values,
                color=plt.cm.viridis(np.linspace(0, 1, len(sig_avg_ratings)))
            )
            
            plt.title('Average Rating by Island', fontsize=16)
            plt.xlabel('Island', fontsize=14)
            plt.ylabel('Average Rating', fontsize=14)
            plt.ylim(3.0, 5.2)  # Typical range for ratings
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            
            # Add data labels
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + 0.05,
                    f'{height:.2f}',
                    ha='center'
                )
                
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, 'avg_rating_by_island.png'), dpi=300)
            plt.close()
        
        # 5. Average Rating by Island and Category
        plt.figure(figsize=(14, 8))
        
        # Prepare data
        rating_data = []
        for island, stats in self.island_stats.items():
            for category, rating in stats.get('avg_ratings_by_category', {}).items():
                review_count = stats.get('reviews_by_category', {}).get(category, 0)
                if review_count >= 5:  # Only include categories with enough reviews
                    rating_data.append({
                        'Island': island,
                        'Category': category.title(),
                        'Rating': rating,
                        'Reviews': review_count
                    })
        
        if rating_data:
            rating_df = pd.DataFrame(rating_data)
            
            # Create pivot table
            rating_pivot = rating_df.pivot(
                index='Island',
                columns='Category',
                values='Rating'
            )
            
            # Set up the figure with consistent styling
            fig, ax = plt.subplots(figsize=(14, 8))
            set_visualization_style()
            
            # Create custom color map for islands
            island_colors = {}
            for island in rating_pivot.index:
                island_lower = island.lower().replace("'", "").replace("'", "")
                island_colors[island] = ISLAND_COLORS_LOWER.get(island_lower, '#AAAAAA')
            
            # Use a categorical color palette for the categories that contrasts with island colors
            category_colors = sns.color_palette('Set2', len(rating_pivot.columns))
            
            # Plot grouped bar chart with custom styling
            rating_pivot.plot(
                kind='bar', 
                ax=ax,
                figsize=(14, 8),
                color=category_colors,
                width=0.8
            )
            
            # Apply consistent styling
            apply_island_style(
                ax,
                title='Average Rating by Island and Category',
                x_label='Island',
                y_label='Average Rating'
            )
            
            # Additional styling
            plt.ylim(3.0, 5.2)  # Typical range for ratings
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Category', fontsize=12, title_fontsize=13, loc='upper right')
            
            # Add a note about the minimum number of reviews
            plt.annotate(
                f"*Only includes categories with at least 5 reviews per island",
                xy=(0.02, 0.02),
                xycoords='figure fraction',
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8)
            )
            
            plt.tight_layout()
            
            # Add data labels to the bars for better readability
            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f', fontsize=9)
                
            plt.savefig(os.path.join(self.viz_dir, 'avg_rating_by_island_and_category.png'), dpi=300)
            plt.close()
        
        # 6. If we have accommodation types, plot those too
        accom_type_data = []
        for island, stats in self.island_stats.items():
            if 'accommodation_types' in stats:
                for accom_type, count in stats['accommodation_types'].items():
                    accom_type_data.append({
                        'Island': island,
                        'Accommodation Type': accom_type.title(),
                        'Count': count
                    })
        
        if accom_type_data:
            accom_type_df = pd.DataFrame(accom_type_data)
            
            # Create pivot table
            accom_pivot = accom_type_df.pivot(
                index='Island',
                columns='Accommodation Type',
                values='Count'
            ).fillna(0)
            
            # Plot stacked bar chart
            plt.figure(figsize=(14, 8))
            accom_pivot.plot(kind='bar', stacked=True, figsize=(14, 8), colormap='viridis')
            plt.title('Accommodation Types by Island', fontsize=16)
            plt.xlabel('Island', fontsize=14)
            plt.ylabel('Number of Reviews', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Accommodation Type', fontsize=12, title_fontsize=13)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, 'accommodation_types_by_island.png'), dpi=300)
            plt.close()
        
        # 7. Property-based visualizations
        if self.property_stats:
            # Properties by Island and Category
            prop_data = []
            for island, stats in self.property_stats.items():
                for category, count in stats.get('category_counts', {}).items():
                    prop_data.append({
                        'Island': island,
                        'Category': category.title(),
                        'Properties': count
                    })
            
            if prop_data:
                prop_df = pd.DataFrame(prop_data)
                
                # Create pivot table
                prop_pivot = prop_df.pivot(
                    index='Island',
                    columns='Category',
                    values='Properties'
                ).fillna(0)
                
                # Plot stacked bar chart
                plt.figure(figsize=(14, 8))
                prop_pivot.plot(kind='bar', stacked=True, figsize=(14, 8), colormap='viridis')
                plt.title('Properties by Island and Category', fontsize=16)
                plt.xlabel('Island', fontsize=14)
                plt.ylabel('Number of Properties', fontsize=14)
                plt.xticks(rotation=45, ha='right')
                plt.legend(title='Category', fontsize=12, title_fontsize=13)
                plt.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(self.viz_dir, 'properties_by_island_and_category.png'), dpi=300)
                plt.close()
            
            # Accommodation Types by Island (Property-based)
            accom_prop_data = []
            for island, stats in self.property_stats.items():
                if 'accommodation_types' in stats:
                    for accom_type, count in stats['accommodation_types'].items():
                        accom_prop_data.append({
                            'Island': island,
                            'Accommodation Type': accom_type.title(),
                            'Count': count
                        })
            
            if accom_prop_data:
                accom_prop_df = pd.DataFrame(accom_prop_data)
                
                # Create pivot table
                accom_prop_pivot = accom_prop_df.pivot(
                    index='Island',
                    columns='Accommodation Type',
                    values='Count'
                ).fillna(0)
                
                # Plot stacked bar chart
                plt.figure(figsize=(14, 8))
                accom_prop_pivot.plot(kind='bar', stacked=True, figsize=(14, 8), colormap='viridis')
                plt.title('Accommodation Property Types by Island', fontsize=16)
                plt.xlabel('Island', fontsize=14)
                plt.ylabel('Number of Properties', fontsize=14)
                plt.xticks(rotation=45, ha='right')
                plt.legend(title='Accommodation Type', fontsize=12, title_fontsize=13)
                plt.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(self.viz_dir, 'accommodation_property_types_by_island.png'), dpi=300)
                plt.close()
            
            # Top properties visualizations
            for island, stats in self.property_stats.items():
                if 'top_properties' in stats:
                    for category, properties in stats['top_properties'].items():
                        if properties and len(properties) > 0:
                            # Convert to DataFrame
                            top_props_df = pd.DataFrame(properties)
                            
                            # Sort by rating
                            top_props_df = top_props_df.sort_values('avg_rating', ascending=False)
                            
                            # Plot
                            plt.figure(figsize=(12, 6))
                            bars = plt.barh(
                                top_props_df['property_name'],
                                top_props_df['avg_rating'],
                                color=[plt.cm.viridis(i/len(top_props_df)) for i in range(len(top_props_df))]
                            )
                            
                            # Add review count annotations
                            for i, row in enumerate(top_props_df.itertuples()):
                                plt.text(
                                    row.avg_rating + 0.05,
                                    i,
                                    f"({row.review_count} reviews)",
                                    va='center'
                                )
                            
                            plt.xlim(3, 5.3)  # Adjust for rating scale
                            plt.title(f'Top {category.title()} in {island}', fontsize=15)
                            plt.xlabel('Average Rating', fontsize=12)
                            plt.grid(axis='x', alpha=0.3)
                            plt.tight_layout()
                            
                            # Sanitize filename
                            safe_island = island.lower().replace("'", "").replace(" ", "_")
                            safe_category = category.lower().replace(" ", "_")
                            
                            plt.savefig(
                                os.path.join(self.viz_dir, f'top_{safe_category}_{safe_island}.png'), 
                                dpi=300
                            )
                            plt.close()
        
        print(f"Visualizations saved to {self.viz_dir}")

    def _save_results(self):
        """Save analysis results to JSON files."""
        # Save island stats
        island_results_file = os.path.join(self.output_dir, 'island_analysis_summary.json')
        
        # Convert non-serializable objects to strings
        def make_serializable(obj):
            if isinstance(obj, (pd.Series, pd.DataFrame)):
                return obj.to_dict()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, datetime):
                return obj.strftime("%Y-%m-%d %H:%M:%S")
            elif isinstance(obj, list) and obj and hasattr(obj[0], 'to_dict'):
                return [item.to_dict() for item in obj]
            return obj
        
        # Clean island stats for JSON
        clean_island_stats = {}
        for island, stats in self.island_stats.items():
            clean_island_stats[island] = {}
            for key, value in stats.items():
                if isinstance(value, dict):
                    clean_island_stats[island][key] = {k: make_serializable(v) for k, v in value.items()}
                else:
                    clean_island_stats[island][key] = make_serializable(value)
        
        # Save to file
        with open(island_results_file, 'w', encoding='utf-8') as f:
            json.dump(clean_island_stats, f, indent=2)
        
        # Save property stats if available
        if self.property_stats:
            property_results_file = os.path.join(self.output_dir, 'property_analysis_summary.json')
            
            # Clean property stats for JSON
            clean_property_stats = {}
            for island, stats in self.property_stats.items():
                clean_property_stats[island] = {}
                for key, value in stats.items():
                    if key == 'top_properties':
                        clean_property_stats[island][key] = {}
                        for category, props in value.items():
                            clean_property_stats[island][key][category] = []
                            for prop in props:
                                clean_prop = {k: make_serializable(v) for k, v in prop.items()}
                                clean_property_stats[island][key][category].append(clean_prop)
                    elif isinstance(value, dict):
                        clean_property_stats[island][key] = {k: make_serializable(v) for k, v in value.items()}
                    else:
                        clean_property_stats[island][key] = make_serializable(value)
            
            # Save to file
            with open(property_results_file, 'w', encoding='utf-8') as f:
                json.dump(clean_property_stats, f, indent=2)
        
        # Save overall stats
        overall_results_file = os.path.join(self.output_dir, 'overall_analysis.json')
        
        # Clean overall stats for JSON
        clean_overall_stats = {}
        for key, value in self.overall_stats.items():
            if isinstance(value, dict):
                clean_overall_stats[key] = {k: make_serializable(v) for k, v in value.items()}
            else:
                clean_overall_stats[key] = make_serializable(value)
        
        # Save to file
        with open(overall_results_file, 'w', encoding='utf-8') as f:
            json.dump(clean_overall_stats, f, indent=2)
        
        print(f"Analysis results saved to {self.output_dir}")

    def export_to_excel(self):
        """Export island analysis results to Excel for further analysis."""
        if not self.island_stats:
            print("No island analysis data to export!")
            return
        
        try:
            # Try to import openpyxl
            import openpyxl
            excel_path = os.path.join(self.output_dir, 'island_analysis.xlsx')
            
            # Create Excel writer
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # Sheet 1: Island Review Summary
                island_summary = []
                for island, stats in self.island_stats.items():
                    row = {
                        'Island': island,
                        'Total Reviews': stats['total_reviews']
                    }
                    
                    # Add counts for each category
                    for category in ['accommodation', 'restaurant', 'attraction']:
                        row[f'{category.title()} Reviews'] = stats.get('category_counts', {}).get(category, 0)
                        row[f'{category.title()} Avg Rating'] = stats.get('avg_ratings_by_category', {}).get(category, 0)
                    
                    # Add accommodation type counts if available
                    if 'accommodation_types' in stats:
                        for accom_type, count in stats['accommodation_types'].items():
                            row[f'{accom_type.title()} Count'] = count
                    
                    island_summary.append(row)
                
                # Create DataFrame and save to sheet
                if island_summary:
                    summary_df = pd.DataFrame(island_summary)
                    summary_df.to_excel(writer, sheet_name='Island Summary', index=False)
                
                # Sheet 2: Property Summary (if available)
                if self.property_stats:
                    property_summary = []
                    for island, stats in self.property_stats.items():
                        row = {
                            'Island': island,
                            'Total Properties': stats['total_properties']
                        }
                        
                        # Add counts for each category
                        for category in ['accommodation', 'restaurant', 'attraction']:
                            row[f'{category.title()} Properties'] = stats.get('category_counts', {}).get(category, 0)
                            row[f'{category.title()} Avg Rating'] = stats.get('avg_ratings_by_category', {}).get(category, 0)
                        
                        # Add accommodation type counts if available
                        if 'accommodation_types' in stats:
                            for accom_type, count in stats['accommodation_types'].items():
                                row[f'{accom_type.title()} Properties'] = count
                        
                        property_summary.append(row)
                    
                    # Create DataFrame and save to sheet
                    if property_summary:
                        prop_summary_df = pd.DataFrame(property_summary)
                        prop_summary_df.to_excel(writer, sheet_name='Property Summary', index=False)
                
                # Sheet 3: Top Properties by Island and Category
                if self.property_stats:
                    top_properties = []
                    for island, stats in self.property_stats.items():
                        if 'top_properties' in stats:
                            for category, props in stats['top_properties'].items():
                                for prop in props:
                                    prop_dict = {
                                        'Island': island,
                                        'Category': category,
                                        'Property Name': prop.get('property_name', ''),
                                        'Average Rating': prop.get('avg_rating', 0),
                                        'Review Count': prop.get('review_count', 0)
                                    }
                                    
                                    if category == 'accommodation' and 'accommodation_type' in prop:
                                        prop_dict['Accommodation Type'] = prop['accommodation_type']
                                    
                                    top_properties.append(prop_dict)
                    
                    # Create DataFrame and save to sheet
                    if top_properties:
                        top_props_df = pd.DataFrame(top_properties)
                        top_props_df.to_excel(writer, sheet_name='Top Properties', index=False)
                
                # Sheet 4: All Reviews Data (sample)
                if hasattr(self, 'all_reviews_df') and self.all_reviews_df is not None:
                    # Take a sample to avoid Excel limitations
                    sample_size = min(10000, len(self.all_reviews_df))
                    sample_df = self.all_reviews_df.sample(sample_size) if sample_size > 0 else self.all_reviews_df
                    sample_df.to_excel(writer, sheet_name='Review Sample', index=False)
                
                # Sheet 5: All Properties Data
                if self.properties_df is not None:
                    self.properties_df.to_excel(writer, sheet_name='All Properties', index=False)
            
            print(f"Analysis exported to Excel: {excel_path}")
            
        except ImportError:
            print("\nopenpyxl not found. Cannot export to Excel.")
            print("To export to Excel, install openpyxl: pip install openpyxl")
            
            # Export summary to CSV instead
            csv_path = os.path.join(self.output_dir, 'island_analysis_summary.csv')
            
            # Create summary data
            island_summary = []
            for island, stats in self.island_stats.items():
                row = {
                    'Island': island,
                    'Total Reviews': stats['total_reviews']
                }
                
                # Add counts for each category
                for category in ['accommodation', 'restaurant', 'attraction']:
                    row[f'{category.title()} Reviews'] = stats.get('category_counts', {}).get(category, 0)
                    row[f'{category.title()} Avg Rating'] = stats.get('avg_ratings_by_category', {}).get(category, 0)
                
                island_summary.append(row)
            
            summary_df = pd.DataFrame(island_summary)
            summary_df.to_csv(csv_path, index=False)
            print(f"Summary data exported to CSV: {csv_path}")

    def run_analysis(self, top_n=5):
        """
        Run the complete island analysis.
        
        Parameters:
        - top_n: Number of top islands to analyze (by review count)
        """
        print("\nStarting Tonga Island Analysis...")
        
        # Step 1: Load all data
        if self.load_data():
            # Step 2: Analyze islands
            self.analyze_islands(top_n=top_n)
            
            # Step 3: Generate visualizations
            self.generate_visualizations()
            
            # Step 4: Export to Excel
            self.export_to_excel()
            
            print("\nIsland analysis complete!")
            print(f"Analysis results saved to: {self.output_dir}")
            return self.island_stats, self.property_stats, self.overall_stats
        else:
            print("Analysis aborted due to data loading issues.")
            return None, None, None


if __name__ == "__main__":
    # Create and run the island analyzer
    analyzer = IslandAnalyzer()
    analyzer.run_analysis(top_n=5)