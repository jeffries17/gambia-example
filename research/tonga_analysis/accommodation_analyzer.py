import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from wordcloud import WordCloud
import json

class AccommodationAnalyzer:
    """
    Specialized analyzer for accommodation reviews in Tonga.
    Focuses on room features, property amenities, and guest experiences.
    Analyzes different types of accommodations and their reception by various traveler segments.
    """
    
    def __init__(self, sentiment_analyzer=None, output_dir='outputs/accommodation_analysis'):
        """
        Initialize the accommodation analyzer.
        
        Parameters:
        - sentiment_analyzer: Optional SentimentAnalyzer instance for text processing
        - output_dir: Directory for accommodation-specific outputs
        """
        self.sentiment_analyzer = sentiment_analyzer
        self.output_dir = output_dir
        
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created accommodation analysis directory: {output_dir}")
            
        # Define accommodation types
        self.accommodation_types = {
            'hotel': ['hotel', 'chain', 'inn', 'motel'],
            'resort': ['resort', 'spa', 'luxury', 'all-inclusive', 'beachfront', 'waterfront'],
            'guesthouse': ['guesthouse', 'guest house', 'b&b', 'bed and breakfast', 'homestay', 'host'],
            'vacation_rental': ['rental', 'apartment', 'villa', 'house', 'cottage', 'cabin', 'airbnb', 'vrbo', 'self-catering'],
            'hostel': ['hostel', 'backpacker', 'dorm', 'dormitory', 'shared'],
            'lodge': ['lodge', 'eco', 'bungalow', 'chalet', 'hut', 'cabin'],
            'traditional': ['traditional', 'local', 'authentic', 'cultural', 'native', 'fale']
        }
        
        # Define room features
        self.room_features = {
            'cleanliness': [
                'clean', 'dirty', 'spotless', 'tidy', 'dust', 'stain', 'hygiene',
                'maintenance', 'housekeeping', 'sanitary', 'smell'
            ],
            'comfort': [
                'comfortable', 'bed', 'mattress', 'pillow', 'sleep', 'quiet',
                'peaceful', 'relaxing', 'cozy', 'soft', 'hard', 'rest'
            ],
            'size': [
                'size', 'space', 'spacious', 'tiny', 'large', 'small', 'big', 
                'cramped', 'roomy'
            ],
            'bathroom': [
                'bathroom', 'shower', 'toilet', 'bath', 'hot water', 'pressure', 
                'sink', 'towels'
            ],
            'amenities': [
                'amenity', 'ac', 'air conditioning', 'tv', 'wifi', 'internet', 
                'fridge', 'minibar', 'kettle', 'coffee', 'tea', 'safe', 'closet',
                'fan'
            ],
            'view': [
                'view', 'ocean', 'sea', 'beach', 'scenic', 'window', 'overlook',
                'balcony', 'terrace', 'patio', 'sunset'
            ],
            'noise': [
                'noise', 'quiet', 'peaceful', 'loud', 'disturb', 'sleep', 
                'soundproof', 'hear', 'traffic'
            ]
        }
        
        # Define property features
        self.property_features = {
            'location': [
                'location', 'central', 'convenient', 'beach', 'downtown', 'access', 
                'close', 'distance', 'walk', 'near', 'far', 'town', 'accessible', 
                'remote', 'quiet'
            ],
            'service': [
                'service', 'staff', 'reception', 'front desk', 'friendly', 'helpful', 
                'professional', 'manager', 'concierge', 'attentive', 'welcoming',
                'host', 'hospitality'
            ],
            'facilities': [
                'facility', 'pool', 'restaurant', 'bar', 'garden', 'spa', 'gym', 
                'parking', 'reception', 'lounge', 'common area', 'fitness', 'business'
            ],
            'breakfast': [
                'breakfast', 'buffet', 'morning meal', 'continental', 'coffee', 
                'fruit', 'cereal', 'eggs'
            ],
            'wifi': [
                'wifi', 'internet', 'connection', 'online', 'slow', 'fast', 'reliable'
            ],
            'safety': [
                'safe', 'security', 'secure', 'lock', 'camera', 'theft', 
                'steal', 'emergency'
            ],
            'value': [
                'value', 'price', 'worth', 'expensive', 'cheap', 'affordable', 'cost', 
                'overpriced', 'budget', 'reasonable', 'luxury'
            ]
        }

    def filter_accommodation_reviews(self, df):
        """
        Filter to only accommodation reviews and add accommodation-specific features.
        
        Parameters:
        - df: DataFrame with review data
        
        Returns:
        - DataFrame with only accommodation reviews and added feature flags
        """
        print("Filtering and processing accommodation reviews...")
        
        # Method 1: Use category if available
        if 'category' in df.columns:
            accommodation_df = df[df['category'] == 'accommodation'].copy()
            print(f"Identified {len(accommodation_df)} accommodation reviews by category")
            
            # If we have too few reviews, try other methods
            if len(accommodation_df) < 10:
                print("Few accommodation reviews found by category, using keyword matching...")
                accommodation_df = self._identify_by_keywords(df)
        else:
            # Method 2: Use keyword matching
            accommodation_df = self._identify_by_keywords(df)
        
        # If we have accommodation reviews, add feature flags
        if len(accommodation_df) > 0:
            # Add accommodation type flags
            for acc_type, keywords in self.accommodation_types.items():
                pattern = '|'.join(keywords)
                accommodation_df[f'type_{acc_type}'] = accommodation_df['text'].str.lower().str.contains(
                    pattern, na=False).astype(int)
            
            # Add room feature flags
            for feature, keywords in self.room_features.items():
                pattern = '|'.join(keywords)
                accommodation_df[f'room_{feature}'] = accommodation_df['text'].str.lower().str.contains(
                    pattern, na=False).astype(int)
            
            # Add property feature flags
            for feature, keywords in self.property_features.items():
                pattern = '|'.join(keywords)
                accommodation_df[f'property_{feature}'] = accommodation_df['text'].str.lower().str.contains(
                    pattern, na=False).astype(int)
            
            print(f"Analysis will use {len(accommodation_df)} accommodation reviews")
            
            # Save a list of accommodation review IDs for reference
            if 'id' in accommodation_df.columns:
                with open(os.path.join(self.output_dir, 'accommodation_review_ids.txt'), 'w') as f:
                    for review_id in accommodation_df['id'].astype(str):
                        f.write(f"{review_id}\n")
        else:
            print("No accommodation reviews found")
            
        return accommodation_df
    
    def _identify_by_keywords(self, df):
        """
        Identify accommodation reviews using keyword matching.
        
        Parameters:
        - df: DataFrame with review data
        
        Returns:
        - DataFrame with accommodation reviews identified by keywords
        """
        # Accommodation-related terms
        accommodation_terms = [
            'hotel', 'resort', 'stay', 'room', 'accommodation', 'lodge', 'hostel', 
            'guesthouse', 'apartment', 'villa', 'inn', 'motel', 'bed', 'sleep',
            'reception', 'check-in', 'check-out', 'lobby', 'suite', 'fale'
        ]
        
        # Create regex pattern
        pattern = '|'.join(accommodation_terms)
        
        # Filter reviews that mention accommodation terms
        if 'text' in df.columns:
            accommodation_mask = df['text'].str.lower().str.contains(pattern, na=False)
            accommodation_df = df[accommodation_mask].copy()
            
            # Check place name if available
            place_columns = [col for col in df.columns if 'place' in col.lower() and 'name' in col.lower()]
            
            if place_columns:
                place_col = place_columns[0]
                place_mask = df[place_col].str.lower().str.contains(
                    'hotel|resort|lodge|villa|apartments|inn|motel|accommodation', na=False)
                
                # Combine indices without duplicates
                combined_indices = pd.Index(list(set(accommodation_df.index) | set(df[place_mask].index)))
                accommodation_df = df.loc[combined_indices].copy()
            
            print(f"Identified {len(accommodation_df)} accommodation reviews by keyword matching")
            return accommodation_df
        else:
            print("No text column available for keyword matching")
            return pd.DataFrame()

    def analyze_accommodation_types(self, df):
        """
        Analyze different types of accommodations mentioned in reviews.
        
        Parameters:
        - df: DataFrame with accommodation reviews
        
        Returns:
        - DataFrame with accommodation type analysis and dictionary of statistics
        """
        print("Analyzing accommodation types mentioned in reviews...")
        
        # Prepare results dictionary
        type_stats = {}
        
        # Check if we have the necessary columns
        type_cols = [f'type_{acc_type}' for acc_type in self.accommodation_types.keys()]
        missing_cols = [col for col in type_cols if col not in df.columns]
        
        if missing_cols:
            print(f"Warning: Missing columns {missing_cols}")
            return df, type_stats
        
        # Calculate type mention counts
        type_counts = {acc_type: df[f'type_{acc_type}'].sum() for acc_type in self.accommodation_types.keys()}
        type_counts = {k: v for k, v in sorted(type_counts.items(), key=lambda item: item[1], reverse=True)}
        
        print("Accommodation type mentions:")
        for acc_type, count in type_counts.items():
            print(f"  {acc_type}: {count} mentions")
        
        # Analyze each accommodation type
        for acc_type in self.accommodation_types.keys():
            col = f'type_{acc_type}'
            type_reviews = df[df[col] == 1]
            
            if len(type_reviews) >= 3:  # Only analyze with sufficient data
                stats = {
                    'review_count': len(type_reviews),
                    'mention_count': int(df[col].sum()),
                    'avg_sentiment': float(type_reviews['sentiment_score'].mean()) if 'sentiment_score' in type_reviews.columns else None,
                    'avg_rating': float(type_reviews['rating'].mean()) if 'rating' in type_reviews.columns else None,
                }
                
                # Add sentiment breakdown if available
                if 'sentiment_category' in type_reviews.columns:
                    sentiment_counts = type_reviews['sentiment_category'].value_counts()
                    stats['positive_reviews'] = int(sentiment_counts.get('positive', 0))
                    stats['neutral_reviews'] = int(sentiment_counts.get('neutral', 0))
                    stats['negative_reviews'] = int(sentiment_counts.get('negative', 0))
                
                # Add top features if enough data
                if len(type_reviews) >= 5:
                    stats['top_features'] = self._get_top_features(type_reviews)
                    stats['common_phrases'] = self._extract_common_phrases(type_reviews)
                
                type_stats[acc_type] = stats
        
        # Create visualizations
        self._visualize_accommodation_types(df, type_stats)
        
        # Save the statistics
        self._save_type_stats(type_stats)
        
        return df, type_stats

    def analyze_room_features(self, df):
        """
        Analyze room features mentioned in accommodation reviews.
        
        Parameters:
        - df: DataFrame with accommodation reviews
        
        Returns:
        - DataFrame with room feature analysis and dictionary of statistics
        """
        print("Analyzing room features mentioned in reviews...")
        
        # Prepare results dictionary
        feature_stats = {}
        
        # Check if we have the necessary columns
        feature_cols = [f'room_{feature}' for feature in self.room_features.keys()]
        missing_cols = [col for col in feature_cols if col not in df.columns]
        
        if missing_cols:
            print(f"Warning: Missing columns {missing_cols}")
            return df, feature_stats
        
        # Calculate feature mention counts
        feature_counts = {feature: df[f'room_{feature}'].sum() for feature in self.room_features.keys()}
        feature_counts = {k: v for k, v in sorted(feature_counts.items(), key=lambda item: item[1], reverse=True)}
        
        print("Room feature mentions:")
        for feature, count in feature_counts.items():
            print(f"  {feature}: {count} mentions")
        
        # Analyze each room feature
        for feature in self.room_features.keys():
            col = f'room_{feature}'
            feature_reviews = df[df[col] == 1]
            
            if len(feature_reviews) >= 3:  # Only analyze with sufficient data
                stats = {
                    'review_count': len(feature_reviews),
                    'mention_count': int(df[col].sum()),
                    'avg_sentiment': float(feature_reviews['sentiment_score'].mean()) if 'sentiment_score' in feature_reviews.columns else None,
                    'avg_rating': float(feature_reviews['rating'].mean()) if 'rating' in feature_reviews.columns else None,
                }
                
                # Add sentiment breakdown if available
                if 'sentiment_category' in feature_reviews.columns:
                    sentiment_counts = feature_reviews['sentiment_category'].value_counts()
                    stats['positive_reviews'] = int(sentiment_counts.get('positive', 0))
                    stats['neutral_reviews'] = int(sentiment_counts.get('neutral', 0))
                    stats['negative_reviews'] = int(sentiment_counts.get('negative', 0))
                
                # Add common phrases if enough data
                if len(feature_reviews) >= 5:
                    stats['common_phrases'] = self._extract_common_phrases(feature_reviews)
                
                feature_stats[feature] = stats
        
        # Create visualizations
        self._visualize_room_features(df, feature_stats)
        
        # Save the statistics
        self._save_feature_stats(feature_stats, 'room')
        
        return df, feature_stats

    def analyze_property_features(self, df):
        """
        Analyze property-wide features mentioned in accommodation reviews.
        
        Parameters:
        - df: DataFrame with accommodation reviews
        
        Returns:
        - DataFrame with property feature analysis and dictionary of statistics
        """
        print("Analyzing property features mentioned in reviews...")
        
        # Prepare results dictionary
        feature_stats = {}
        
        # Check if we have the necessary columns
        feature_cols = [f'property_{feature}' for feature in self.property_features.keys()]
        missing_cols = [col for col in feature_cols if col not in df.columns]
        
        if missing_cols:
            print(f"Warning: Missing columns {missing_cols}")
            return df, feature_stats
        
        # Calculate feature mention counts
        feature_counts = {feature: df[f'property_{feature}'].sum() for feature in self.property_features.keys()}
        feature_counts = {k: v for k, v in sorted(feature_counts.items(), key=lambda item: item[1], reverse=True)}
        
        print("Property feature mentions:")
        for feature, count in feature_counts.items():
            print(f"  {feature}: {count} mentions")
        
        # Analyze each property feature
        for feature in self.property_features.keys():
            col = f'property_{feature}'
            feature_reviews = df[df[col] == 1]
            
            if len(feature_reviews) >= 3:  # Only analyze with sufficient data
                stats = {
                    'review_count': len(feature_reviews),
                    'mention_count': int(df[col].sum()),
                    'avg_sentiment': float(feature_reviews['sentiment_score'].mean()) if 'sentiment_score' in feature_reviews.columns else None,
                    'avg_rating': float(feature_reviews['rating'].mean()) if 'rating' in feature_reviews.columns else None,
                }
                
                # Add sentiment breakdown if available
                if 'sentiment_category' in feature_reviews.columns:
                    sentiment_counts = feature_reviews['sentiment_category'].value_counts()
                    stats['positive_reviews'] = int(sentiment_counts.get('positive', 0))
                    stats['neutral_reviews'] = int(sentiment_counts.get('neutral', 0))
                    stats['negative_reviews'] = int(sentiment_counts.get('negative', 0))
                
                # Add common phrases if enough data
                if len(feature_reviews) >= 5:
                    stats['common_phrases'] = self._extract_common_phrases(feature_reviews)
                
                feature_stats[feature] = stats
        
        # Create visualizations
        self._visualize_property_features(df, feature_stats)
        
        # Save the statistics
        self._save_feature_stats(feature_stats, 'property')
        
        return df, feature_stats

    def analyze_by_trip_purpose(self, df):
        """
        Analyze accommodation preferences and sentiment by trip purpose (trip type).
        
        Parameters:
        - df: DataFrame with accommodation reviews
        
        Returns:
        - DataFrame with trip purpose analysis and dictionary of statistics
        """
        print("Analyzing accommodation preferences by trip purpose...")
        
        # Prepare results dictionary
        trip_stats = {}
        
        # Determine which trip type column to use
        trip_col = None
        for col in ['trip_type_standard', 'trip_type', 'tripType']:
            if col in df.columns:
                trip_col = col
                break
        
        if not trip_col:
            print("No trip type information available for trip purpose analysis")
            return df, trip_stats
        
        # Get valid trip types with enough data
        trip_counts = df[trip_col].value_counts()
        valid_trips = trip_counts[trip_counts >= 3].index.tolist()
        
        if not valid_trips:
            print("Not enough data for trip purpose analysis")
            return df, trip_stats
        
        # Filter to segments with enough data
        segment_df = df[df[trip_col].isin(valid_trips)]
        
        # Get feature columns
        room_cols = [col for col in df.columns if col.startswith('room_')]
        property_cols = [col for col in df.columns if col.startswith('property_')]
        all_feature_cols = room_cols + property_cols
        
        # For each valid trip type, analyze preferences
        for trip_type in valid_trips:
            if pd.isna(trip_type) or trip_type == 'unknown' or trip_type == 'other':
                continue
                
            trip_reviews = df[df[trip_col] == trip_type]
            
            if len(trip_reviews) >= 3:  # Minimum threshold for analysis
                stats = {
                    'review_count': len(trip_reviews),
                    'avg_sentiment': float(trip_reviews['sentiment_score'].mean()) if 'sentiment_score' in trip_reviews.columns else None,
                    'avg_rating': float(trip_reviews['rating'].mean()) if 'rating' in trip_reviews.columns else None,
                }
                
                # Add feature preferences if we have feature columns
                if all_feature_cols:
                    # Calculate the percentage of each segment that mentions each feature
                    feature_mentions = {}
                    for col in all_feature_cols:
                        if col in trip_reviews.columns:
                            feature_name = col.replace('room_', '').replace('property_', '')
                            feature_type = 'Room' if col.startswith('room_') else 'Property'
                            feature_key = f"{feature_type}: {feature_name}"
                            feature_mentions[feature_key] = float(trip_reviews[col].mean())
                    
                    # Get top mentioned features
                    sorted_features = sorted(feature_mentions.items(), key=lambda x: x[1], reverse=True)
                    stats['top_features'] = dict(sorted_features[:5])  # Top 5 features
                
                # Add common phrases if enough data
                if len(trip_reviews) >= 5:
                    stats['common_phrases'] = self._extract_common_phrases(trip_reviews)
                
                trip_stats[trip_type] = stats
        
        # Create visualizations if we have enough data
        if trip_stats:
            self._visualize_trip_preferences(df, trip_stats, trip_col)
        
        # Save the statistics
        self._save_trip_stats(trip_stats)
        
        return df, trip_stats

    def generate_accommodation_recommendations(self, type_stats, room_stats, property_stats, trip_stats):
        """
        Generate accommodation-specific recommendations based on the analysis.
        
        Parameters:
        - type_stats: Statistics about accommodation types
        - room_stats: Statistics about room features
        - property_stats: Statistics about property features
        - trip_stats: Statistics about trip type preferences
        
        Returns:
        - Dictionary with accommodation recommendations
        """
        print("Generating accommodation-specific recommendations...")
        
        recommendations = {
            "accommodation_types": [],
            "room_improvements": [],
            "property_improvements": [],
            "visitor_segments": [],
            "marketing": []
        }
        
        # Accommodation type recommendations
        if type_stats:
            # Sort by sentiment
            types_by_sentiment = {k: v.get('avg_sentiment', 0) for k, v in type_stats.items() if v.get('avg_sentiment') is not None}
            
            if types_by_sentiment:
                # Most positive and negative types
                most_positive = max(types_by_sentiment.items(), key=lambda x: x[1])
                most_negative = min(types_by_sentiment.items(), key=lambda x: x[1])
                
                if most_positive[1] > 0.2:  # Very positive
                    recommendations["marketing"].append(
                        f"Highlight {most_positive[0].replace('_', ' ')} options in marketing "
                        f"as they receive the most positive sentiment ({most_positive[1]:.2f})"
                    )
                
                if most_negative[1] < 0:  # Negative
                    recommendations["accommodation_types"].append(
                        f"Focus on improving {most_negative[0].replace('_', ' ')} accommodations "
                        f"which receive negative sentiment ({most_negative[1]:.2f})"
                    )
        
        # Room feature recommendations
        if room_stats:
            # Sort by sentiment
            features_by_sentiment = {k: v.get('avg_sentiment', 0) for k, v in room_stats.items() if v.get('avg_sentiment') is not None}
            
            if features_by_sentiment:
                # Features with negative sentiment
                negative_features = {k: v for k, v in features_by_sentiment.items() if v < 0}
                for feature, sentiment in negative_features.items():
                    recommendations["room_improvements"].append(
                        f"Address issues with {feature.replace('_', ' ')} which has negative "
                        f"sentiment ({sentiment:.2f})"
                    )
                
                # Features with very positive sentiment
                positive_features = {k: v for k, v in features_by_sentiment.items() if v > 0.3}
                for feature, sentiment in positive_features.items():
                    recommendations["marketing"].append(
                        f"Emphasize {feature.replace('_', ' ')} in marketing materials "
                        f"as it receives very positive sentiment ({sentiment:.2f})"
                    )
        
        # Property feature recommendations
        if property_stats:
            # Sort by sentiment
            features_by_sentiment = {k: v.get('avg_sentiment', 0) for k, v in property_stats.items() if v.get('avg_sentiment') is not None}
            
            if features_by_sentiment:
                # Features with negative sentiment
                negative_features = {k: v for k, v in features_by_sentiment.items() if v < 0}
                for feature, sentiment in negative_features.items():
                    recommendations["property_improvements"].append(
                        f"Improve {feature.replace('_', ' ')} which has negative "
                        f"sentiment ({sentiment:.2f})"
                    )
                
                # Most important features by mention
                features_by_mention = {k: v.get('mention_count', 0) for k, v in property_stats.items()}
                top_features = sorted(features_by_mention.items(), key=lambda x: x[1], reverse=True)[:3]
                
                recommendations["property_improvements"].append(
                    f"Focus on key property features: {', '.join([f[0].replace('_', ' ') for f in top_features])}, "
                    f"which are mentioned most frequently in reviews"
                )
        
        # Trip purpose recommendations
        if trip_stats:
            # Sort by sentiment
            sentiment_by_trip = {k: v.get('avg_sentiment', 0) for k, v in trip_stats.items() if v.get('avg_sentiment') is not None}
            
            if sentiment_by_trip:
                # Trip types with negative sentiment
                negative_trips = {k: v for k, v in sentiment_by_trip.items() if v < 0}
                for trip, sentiment in negative_trips.items():
                    recommendations["visitor_segments"].append(
                        f"Focus on improving accommodations for {trip.replace('_', ' ')} travelers "
                        f"who report negative experiences ({sentiment:.2f})"
                    )
                
                # Trip types with very positive sentiment
                positive_trips = {k: v for k, v in sentiment_by_trip.items() if v > 0.3}
                for trip, sentiment in positive_trips.items():
                    recommendations["marketing"].append(
                        f"Target marketing towards {trip.replace('_', ' ')} travelers "
                        f"who particularly enjoy Tonga accommodations ({sentiment:.2f})"
                    )
                
                # Add feature preferences by trip type
                for trip, stats in trip_stats.items():
                    if 'top_features' in stats:
                        top_feature = max(stats['top_features'].items(), key=lambda x: x[1])
                        
                        if top_feature[1] > 0.2:  # If mentioned frequently
                            recommendations["visitor_segments"].append(
                                f"For {trip.replace('_', ' ')} travelers, prioritize {top_feature[0].lower()} "
                                f"({top_feature[1]:.2f} mention rate)"
                            )
        
        # Save recommendations
        with open(os.path.join(self.output_dir, 'accommodation_recommendations.json'), 'w', encoding='utf-8') as f:
            json.dump(recommendations, f, indent=2)
        
        # Print key recommendations
        print("\nKey accommodation recommendations:")
        for category, recs in recommendations.items():
            if recs:
                print(f"\n{category.replace('_', ' ').title()}:")
                for rec in recs:
                    print(f"  - {rec}")
        
        return recommendations

    def _visualize_accommodation_types(self, df, type_stats):
        """
        Create visualizations for accommodation type analysis.
        
        Parameters:
        - df: DataFrame with accommodation reviews
        - type_stats: Dictionary with type statistics
        """
        # Create visualizations directory if needed
        viz_dir = os.path.join(self.output_dir, 'visualizations')
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        
        # If we don't have enough data, skip
        if not type_stats or len(type_stats) < 2:
            return
        
        # Prepare data for visualization
        type_metrics = []
        for acc_type, stats in type_stats.items():
            metrics = {'type': acc_type, 'count': stats['mention_count']}
            
            if 'avg_sentiment' in stats and stats['avg_sentiment'] is not None:
                metrics['sentiment'] = stats['avg_sentiment']
                
            if 'avg_rating' in stats and stats['avg_rating'] is not None:
                metrics['rating'] = stats['avg_rating']
                
            type_metrics.append(metrics)
        
        # Sort by count
        type_metrics.sort(key=lambda x: x['count'], reverse=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-colorblind')
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(type_metrics)))
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. Counts
        x = range(len(type_metrics))
        counts = [m['count'] for m in type_metrics]
        bars1 = ax1.bar(x, counts, color=colors)
        ax1.set_title('Reviews by Accommodation Type', fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels([m['type'].replace('_', ' ').title() for m in type_metrics], rotation=45, ha='right')
        ax1.set_ylabel('Number of Reviews')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add count labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Sentiment if available
        if all('sentiment' in m for m in type_metrics):
            sentiments = [m['sentiment'] for m in type_metrics]
            bars2 = ax2.bar(x, sentiments, color=colors)
            ax2.set_title('Average Sentiment by Accommodation Type', fontsize=14)
            ax2.set_xticks(x)
            ax2.set_xticklabels([m['type'].replace('_', ' ').title() for m in type_metrics], rotation=45, ha='right')
            ax2.set_ylabel('Average Sentiment Score')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax2.grid(axis='y', alpha=0.3)
            
            # Add sentiment labels
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
        
        # Or rating if available but no sentiment
        elif all('rating' in m for m in type_metrics):
            ratings = [m['rating'] for m in type_metrics]
            bars2 = ax2.bar(x, ratings, color=colors)
            ax2.set_title('Average Rating by Accommodation Type', fontsize=14)
            ax2.set_xticks(x)
            ax2.set_xticklabels([m['type'].replace('_', ' ').title() for m in type_metrics], rotation=45, ha='right')
            ax2.set_ylabel('Average Rating (1-5)')
            ax2.set_ylim(1, 5.5)
            ax2.grid(axis='y', alpha=0.3)
            
            # Add rating labels
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
        else:
            # If we don't have sentiment or rating data, use only one graph
            fig.delaxes(ax2)
            plt.tight_layout()
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'accommodation_types.png'), dpi=300)
        plt.close()
        
        # 3. Rating distribution by accommodation type if ratings available
        if 'rating' in df.columns:
            plt.figure(figsize=(12, 6))
            
            for i, acc_type in enumerate(sorted(type_stats.keys())):
                col = f'type_{acc_type}'
                if col in df.columns:
                    type_reviews = df[df[col] == 1]
                    if len(type_reviews) > 5:  # Only plot if we have enough data
                        plt.hist(type_reviews['rating'], alpha=0.5, 
                            label=acc_type.replace('_', ' ').title(),
                            color=plt.cm.viridis(i/len(type_stats)))
            
            plt.title('Rating Distribution by Accommodation Type', fontsize=14)
            plt.xlabel('Rating')
            plt.ylabel('Number of Reviews')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'rating_distribution_by_type.png'), dpi=300)
            plt.close()
        
        print(f"Accommodation type visualizations saved to {viz_dir}")

    def _visualize_room_features(self, df, feature_stats):
        """
        Create visualizations for room feature analysis.
        
        Parameters:
        - df: DataFrame with accommodation reviews
        - feature_stats: Dictionary with feature statistics
        """
        # Create visualizations directory if needed
        viz_dir = os.path.join(self.output_dir, 'visualizations')
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        
        # If we don't have enough data, skip
        if not feature_stats or len(feature_stats) < 2:
            return
        
        # Prepare data for visualization
        feature_metrics = []
        for feature, stats in feature_stats.items():
            metrics = {'feature': feature, 'count': stats['mention_count']}
            
            if 'avg_sentiment' in stats and stats['avg_sentiment'] is not None:
                metrics['sentiment'] = stats['avg_sentiment']
                
            if 'avg_rating' in stats and stats['avg_rating'] is not None:
                metrics['rating'] = stats['avg_rating']
                
            feature_metrics.append(metrics)
        
        # Sort by count
        feature_metrics.sort(key=lambda x: x['count'], reverse=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-colorblind')
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(feature_metrics)))
        
        # Create plot for mentions and sentiment
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 1. Mention counts
        x = range(len(feature_metrics))
        counts = [m['count'] for m in feature_metrics]
        bars1 = ax1.bar(x, counts, color=colors)
        ax1.set_title('Room Features Mentioned in Reviews', fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels([m['feature'].replace('_', ' ').title() for m in feature_metrics], rotation=45, ha='right')
        ax1.set_ylabel('Number of Mentions')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add count labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Sentiment if available
        if all('sentiment' in m for m in feature_metrics):
            sentiments = [m['sentiment'] for m in feature_metrics]
            # Use safe comparison for coloring
            colors_sent = ['green' if s is not None and s > 0 else 'red' for s in sentiments]
            bars2 = ax2.bar(x, sentiments, color=colors_sent)
            ax2.set_title('Average Sentiment for Room Features', fontsize=14)
            ax2.set_xticks(x)
            ax2.set_xticklabels([m['feature'].replace('_', ' ').title() for m in feature_metrics], rotation=45, ha='right')
            ax2.set_ylabel('Average Sentiment Score')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax2.grid(axis='y', alpha=0.3)
            
            # Add sentiment labels
            for bar in bars2:
                height = bar.get_height()
                if height is not None:
                    label_pos = height + 0.02 if height > 0 else height - 0.08
                    ax2.text(bar.get_x() + bar.get_width()/2., label_pos,
                            f'{height:.2f}', ha='center', va='bottom' if height > 0 else 'top',
                        color='black', fontweight='bold')
        else:
            # If we don't have sentiment data, use only one graph
            fig.delaxes(ax2)
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'room_features_analysis.png'), dpi=300)
        plt.close()
        
        print(f"Room feature visualizations saved to {viz_dir}")

    def _visualize_property_features(self, df, feature_stats):
        """
        Create visualizations for property feature analysis.
        
        Parameters:
        - df: DataFrame with accommodation reviews
        - feature_stats: Dictionary with feature statistics
        """
        # Create visualizations directory if needed
        viz_dir = os.path.join(self.output_dir, 'visualizations')
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        
        # If we don't have enough data, skip
        if not feature_stats or len(feature_stats) < 2:
            return
        
        # Prepare data for visualization
        feature_metrics = []
        for feature, stats in feature_stats.items():
            metrics = {'feature': feature, 'count': stats['mention_count']}
            
            if 'avg_sentiment' in stats and stats['avg_sentiment'] is not None:
                metrics['sentiment'] = stats['avg_sentiment']
                
            if 'avg_rating' in stats and stats['avg_rating'] is not None:
                metrics['rating'] = stats['avg_rating']
                
            feature_metrics.append(metrics)
        
        # Sort by count
        feature_metrics.sort(key=lambda x: x['count'], reverse=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-colorblind')
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(feature_metrics)))
        
        # Create plot for mentions and sentiment
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 1. Mention counts
        x = range(len(feature_metrics))
        counts = [m['count'] for m in feature_metrics]
        bars1 = ax1.bar(x, counts, color=colors)
        ax1.set_title('Property Features Mentioned in Reviews', fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels([m['feature'].replace('_', ' ').title() for m in feature_metrics], rotation=45, ha='right')
        ax1.set_ylabel('Number of Mentions')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add count labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Sentiment if available
        if all('sentiment' in m for m in feature_metrics):
            sentiments = [m['sentiment'] for m in feature_metrics]
            colors_sent = ['green' if s is not None and s > 0 else 'red' for s in sentiments]
            bars2 = ax2.bar(x, sentiments, color=colors_sent)
            ax2.set_title('Average Sentiment for Property Features', fontsize=14)
            ax2.set_xticks(x)
            ax2.set_xticklabels([m['feature'].replace('_', ' ').title() for m in feature_metrics], rotation=45, ha='right')
            ax2.set_ylabel('Average Sentiment Score')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax2.grid(axis='y', alpha=0.3)
            
            # Add sentiment labels
            for bar in bars2:
                height = bar.get_height()
                if height is not None:
                    label_pos = height + 0.02 if height > 0 else height - 0.08
                    ax2.text(bar.get_x() + bar.get_width()/2., label_pos,
                            f'{height:.2f}', ha='center', va='bottom' if height > 0 else 'top',
                        color='black', fontweight='bold')
        else:
            # If we don't have sentiment data, use only one graph
            fig.delaxes(ax2)
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'property_features_analysis.png'), dpi=300)
        plt.close()
        
        print(f"Property feature visualizations saved to {viz_dir}")
        
    def visualize_island_accommodation_comparisons(self, island_results, output_dir):
        """
        Create side-by-side visualizations comparing accommodation features across islands.
        
        Parameters:
        - island_results: Dictionary with results by island from run_island_analysis
        - output_dir: Directory to save the comparison visualizations
        """
        print("\nGenerating cross-island accommodation comparison visualizations...")
        
        # Create visualizations directory
        viz_dir = os.path.join(output_dir, 'island_comparisons')
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        
        # Get islands with enough data
        valid_islands = [island for island, data in island_results.items() 
                        if data['review_count'] >= 10]
        
        if len(valid_islands) < 2:
            print("Not enough islands with sufficient data for comparison")
            return
        
        # Set visualization style
        plt.style.use('seaborn-v0_8-colorblind')
        
        # 1. Accommodation Types Comparison
        accommodation_types = {}
        
        # Collect data for all islands
        for island in valid_islands:
            type_stats = island_results[island]['accommodation_types']
            for acc_type, stats in type_stats.items():
                if 'mention_count' in stats:
                    if acc_type not in accommodation_types:
                        accommodation_types[acc_type] = {}
                    # Calculate percentage of reviews mentioning this type
                    accommodation_types[acc_type][island] = (stats['mention_count'] / 
                                                        island_results[island]['review_count'] * 100)
        
        # Create visualization if we have data
        if accommodation_types:
            # Convert to DataFrame
            df_acc_types = pd.DataFrame(accommodation_types)
            
            # Sort columns by overall mention frequency
            total_mentions = df_acc_types.sum()
            sorted_types = total_mentions.sort_values(ascending=False).index.tolist()
            
            # Limit to top 7 types if there are too many
            if len(sorted_types) > 7:
                sorted_types = sorted_types[:7]
            
            # Select and sort data
            plot_data = df_acc_types[sorted_types].copy()
            
            # Create plot
            plt.figure(figsize=(14, 8))
            plot_data.plot(kind='bar', figsize=(14, 8))
            plt.title('Accommodation Types by Island (% of Reviews)', fontsize=16)
            plt.xlabel('Island', fontsize=14)
            plt.ylabel('% of Reviews Mentioning', fontsize=14)
            plt.xticks(rotation=45, ha='right', fontsize=12)
            plt.legend(title='Accommodation Type', fontsize=12, title_fontsize=14)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'accommodation_types_by_island.png'), dpi=300)
            plt.close()
        
        # 2. Room Features Comparison
        room_features = {}
        
        # Collect data for all islands
        for island in valid_islands:
            feature_stats = island_results[island]['room_features']
            for feature, stats in feature_stats.items():
                if 'mention_count' in stats:
                    if feature not in room_features:
                        room_features[feature] = {}
                    # Calculate percentage of reviews mentioning this feature
                    room_features[feature][island] = (stats['mention_count'] / 
                                                    island_results[island]['review_count'] * 100)
        
        # Create visualization if we have data
        if room_features:
            # Convert to DataFrame
            df_room = pd.DataFrame(room_features)
            
            # Sort columns by overall mention frequency
            total_mentions = df_room.sum()
            sorted_features = total_mentions.sort_values(ascending=False).index.tolist()
            
            # Limit to top 7 features if there are too many
            if len(sorted_features) > 7:
                sorted_features = sorted_features[:7]
            
            # Select and sort data
            plot_data = df_room[sorted_features].copy()
            
            # Create plot
            plt.figure(figsize=(14, 8))
            plot_data.plot(kind='bar', figsize=(14, 8))
            plt.title('Room Features by Island (% of Reviews)', fontsize=16)
            plt.xlabel('Island', fontsize=14)
            plt.ylabel('% of Reviews Mentioning', fontsize=14)
            plt.xticks(rotation=45, ha='right', fontsize=12)
            plt.legend(title='Room Feature', fontsize=12, title_fontsize=14)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'room_features_by_island.png'), dpi=300)
            plt.close()
        
        # 3. Property Features Comparison
        property_features = {}
        
        # Collect data for all islands
        for island in valid_islands:
            feature_stats = island_results[island]['property_features']
            for feature, stats in feature_stats.items():
                if 'mention_count' in stats:
                    if feature not in property_features:
                        property_features[feature] = {}
                    # Calculate percentage of reviews mentioning this feature
                    property_features[feature][island] = (stats['mention_count'] / 
                                                    island_results[island]['review_count'] * 100)
        
        # Create visualization if we have data
        if property_features:
            # Convert to DataFrame
            df_property = pd.DataFrame(property_features)
            
            # Sort columns by overall mention frequency
            total_mentions = df_property.sum()
            sorted_features = total_mentions.sort_values(ascending=False).index.tolist()
            
            # Limit to top 7 features if there are too many
            if len(sorted_features) > 7:
                sorted_features = sorted_features[:7]
            
            # Select and sort data
            plot_data = df_property[sorted_features].copy()
            
            # Create plot
            plt.figure(figsize=(14, 8))
            plot_data.plot(kind='bar', figsize=(14, 8))
            plt.title('Property Features by Island (% of Reviews)', fontsize=16)
            plt.xlabel('Island', fontsize=14)
            plt.ylabel('% of Reviews Mentioning', fontsize=14)
            plt.xticks(rotation=45, ha='right', fontsize=12)
            plt.legend(title='Property Feature', fontsize=12, title_fontsize=14)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'property_features_by_island.png'), dpi=300)
            plt.close()
        
        # 4. Average Ratings Comparison
        if all('avg_rating' in island_results[island] and 
            island_results[island]['avg_rating'] is not None 
            for island in valid_islands):
            
            # Collect ratings for each island
            ratings = {island: island_results[island]['avg_rating'] for island in valid_islands}
            
            # Create ratings DataFrame
            df_ratings = pd.DataFrame(list(ratings.items()), columns=['Island', 'Rating'])
            df_ratings = df_ratings.sort_values('Rating', ascending=False)
            
            # Create bar chart
            plt.figure(figsize=(12, 6))
            bars = plt.bar(df_ratings['Island'], df_ratings['Rating'], color='skyblue')
            plt.title('Average Accommodation Ratings by Island', fontsize=16)
            plt.xlabel('Island', fontsize=14)
            plt.ylabel('Average Rating (1-5)', fontsize=14)
            plt.ylim(min(df_ratings['Rating']) - 0.5 if min(df_ratings['Rating']) > 0.5 else 0, 5.5)
            plt.xticks(rotation=45, ha='right', fontsize=12)
            plt.grid(axis='y', alpha=0.3)
            
            # Add rating labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'average_ratings_by_island.png'), dpi=300)
            plt.close()
        
        # 5. Sentiment Comparison (if sentiment data is available)
        sentiment_available = True
        for island in valid_islands:
            # Check if room features have sentiment data
            room_has_sentiment = False
            for _, stats in island_results[island]['room_features'].items():
                if 'avg_sentiment' in stats and stats['avg_sentiment'] is not None:
                    room_has_sentiment = True
                    break
            
            if not room_has_sentiment:
                sentiment_available = False
                break
        
        if sentiment_available:
            # Collect average sentiment for top features across islands
            feature_sentiment = {}
            
            # Room features
            for island in valid_islands:
                feature_stats = island_results[island]['room_features']
                for feature, stats in feature_stats.items():
                    if ('avg_sentiment' in stats and stats['avg_sentiment'] is not None and
                        'mention_count' in stats and stats['mention_count'] >= 5):
                        if feature not in feature_sentiment:
                            feature_sentiment[feature] = {}
                        feature_sentiment[feature][island] = stats['avg_sentiment']
            
            # Filter to features with data from multiple islands
            feature_sentiment = {f: data for f, data in feature_sentiment.items() 
                                if len(data) >= 2}
            
            if feature_sentiment:
                # Convert to DataFrame
                df_sentiment = pd.DataFrame(feature_sentiment)
                
                # Sort columns by absolute sentiment values
                abs_sentiment = df_sentiment.abs().mean()
                sorted_features = abs_sentiment.sort_values(ascending=False).index.tolist()
                
                # Limit to top 5 features with most variation in sentiment
                if len(sorted_features) > 5:
                    sorted_features = sorted_features[:5]
                
                # Select and sort data
                plot_data = df_sentiment[sorted_features].copy()
                
                # Create plot
                plt.figure(figsize=(14, 8))
                plot_data.plot(kind='bar', figsize=(14, 8))
                plt.title('Room Feature Sentiment by Island', fontsize=16)
                plt.xlabel('Island', fontsize=14)
                plt.ylabel('Average Sentiment Score', fontsize=14)
                plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                plt.xticks(rotation=45, ha='right', fontsize=12)
                plt.legend(title='Room Feature', fontsize=12, title_fontsize=14)
                plt.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, 'room_sentiment_by_island.png'), dpi=300)
                plt.close()
        
        print(f"Island comparison visualizations saved to {viz_dir}")

    def _visualize_trip_preferences(self, df, trip_stats, trip_col):
        """
        Create visualizations for trip type preference analysis.
        
        Parameters:
        - df: DataFrame with accommodation reviews
        - trip_stats: Dictionary with trip type statistics
        - trip_col: Column name for trip type
        """
        # Create visualizations directory if needed
        viz_dir = os.path.join(self.output_dir, 'visualizations')
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        
        # If we don't have enough data, skip
        if not trip_stats or len(trip_stats) < 2:
            return
        
        # Create a heatmap of feature preferences by trip type
        feature_mentions = {}
        
        # Check if we have top features info for each trip type
        have_features = all('top_features' in stats for stats in trip_stats.values())
        
        if have_features:
            # Gather all unique feature names
            all_features = set()
            for trip, stats in trip_stats.items():
                all_features.update(stats['top_features'].keys())
            
            # Create a data frame for the heatmap
            heatmap_data = pd.DataFrame(index=trip_stats.keys(), columns=sorted(all_features))
            
            # Fill in the data
            for trip, stats in trip_stats.items():
                for feature, value in stats['top_features'].items():
                    if value is not None:  # Only add non-None values
                        heatmap_data.loc[trip, feature] = value
            
            # Fill NaN with zeros
            heatmap_data = heatmap_data.fillna(0)
            
            # Create the heatmap
            plt.figure(figsize=(15, 8))
            sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt='.2f')
            plt.title('Accommodation Feature Preferences by Trip Purpose', fontsize=14, pad=20)
            plt.ylabel('Trip Purpose')
            plt.xlabel('Accommodation Feature')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'trip_purpose_feature_preferences.png'), dpi=300)
            plt.close()
        
        # Create a bar chart of sentiment by trip type
        # Check if all trip stats have valid sentiment values
        valid_sentiments = True
        for _, stats in trip_stats.items():
            if 'avg_sentiment' not in stats or stats['avg_sentiment'] is None:
                valid_sentiments = False
                break
        
        if valid_sentiments:
            # Sort by sentiment
            sorted_trips = sorted(trip_stats.items(), key=lambda x: x[1]['avg_sentiment'], reverse=True)
            
            # Create the plot
            plt.figure(figsize=(10, 6))
            x = range(len(sorted_trips))
            sentiments = [stats['avg_sentiment'] for _, stats in sorted_trips]
            colors = ['green' if s is not None and s > 0 else 'red' for s in sentiments]
            
            bars = plt.bar(x, sentiments, color=colors)
            plt.title('Average Sentiment by Trip Purpose', fontsize=14, pad=20)
            plt.xlabel('Trip Purpose')
            plt.ylabel('Average Sentiment Score')
            plt.xticks(x, [trip.replace('_', ' ').title() for trip, _ in sorted_trips], rotation=45, ha='right')
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.grid(axis='y', alpha=0.3)
            
            # Add sentiment labels
            for bar in bars:
                height = bar.get_height()
                if height is not None:  # Only add labels for non-None values
                    label_pos = height + 0.02 if height > 0 else height - 0.08
                    plt.text(bar.get_x() + bar.get_width()/2., label_pos,
                        f'{height:.2f}', ha='center', va='bottom' if height > 0 else 'top',
                        color='black', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'sentiment_by_trip_purpose.png'), dpi=300)
            plt.close()
        
        print(f"Trip preference visualizations saved to {viz_dir}")

    def _extract_common_phrases(self, df, min_count=2, max_phrases=15):
        """
        Extract common phrases from a set of reviews.
        
        Parameters:
        - df: DataFrame with reviews
        - min_count: Minimum count to include a phrase
        - max_phrases: Maximum number of phrases to return
        
        Returns:
        - Dictionary of common phrases and their counts
        """
        # If no processed text column, try to use regular text
        text_col = 'processed_text' if 'processed_text' in df.columns else 'text'
        
        if text_col not in df.columns or len(df) == 0:
            return {}
        
        # Combine all text
        text = ' '.join(df[text_col].fillna('').astype(str))
        
        # Convert to lowercase and split into words
        words = text.lower().split()
        
        # Remove common stopwords
        stopwords = ['the', 'and', 'to', 'of', 'a', 'in', 'for', 'is', 'on', 'with', 'was',
                    'i', 'we', 'our', 'it', 'that', 'this', 'my', 'at', 'as', 'by', 'be',
                    'you', 'your', 'are', 'they', 'their', 'from', 'have', 'has', 'had',
                    'so', 'but', 'just', 'all', 'or', 'not', 'very', 'an', 'no', 'which',
                    'there', 'when', 'what', 'who', 'how', 'why', 'where', 'me']
        
        # Count words excluding stopwords and single characters
        word_freq = Counter([w for w in words if w not in stopwords and len(w) > 1])
        
        # Get most common phrases
        common_phrases = {word: count for word, count in word_freq.most_common(max_phrases) 
                         if count >= min_count}
        
        return common_phrases

    def _get_top_features(self, df, top_n=5):
        """
        Get the most mentioned features in a set of reviews.
        
        Parameters:
        - df: DataFrame with reviews
        - top_n: Number of top features to return
        
        Returns:
        - Dictionary of top features and their mention rates
        """
        # Get all feature columns
        room_cols = [col for col in df.columns if col.startswith('room_')]
        property_cols = [col for col in df.columns if col.startswith('property_')]
        all_cols = room_cols + property_cols
        
        if not all_cols:
            return {}
        
        # Calculate the mention rate for each feature
        feature_rates = {}
        for col in all_cols:
            if col in df.columns:
                feature_name = col.replace('room_', 'Room: ').replace('property_', 'Property: ')
                feature_rates[feature_name] = float(df[col].mean())
        
        # Sort by mention rate and take top N
        top_features = dict(sorted(feature_rates.items(), key=lambda x: x[1], reverse=True)[:top_n])
        
        return top_features

    def _save_type_stats(self, type_stats):
        """
        Save accommodation type statistics to a file.
        
        Parameters:
        - type_stats: Dictionary with type statistics
        """
        # Convert to DataFrame for easier manipulation
        stats_list = []
        for acc_type, stats in type_stats.items():
            row = {'accommodation_type': acc_type}
            row.update(stats)
            stats_list.append(row)
        
        if stats_list:
            stats_df = pd.DataFrame(stats_list)
            
            # Save to CSV
            stats_df.to_csv(os.path.join(self.output_dir, 'accommodation_type_stats.csv'), index=False)
            
            # Save to JSON for more complex structures
            with open(os.path.join(self.output_dir, 'accommodation_type_analysis.json'), 'w', encoding='utf-8') as f:
                json.dump(type_stats, f, indent=2, default=self._json_serialize)

    def _save_feature_stats(self, feature_stats, feature_type):
        """
        Save feature statistics to a file.
        
        Parameters:
        - feature_stats: Dictionary with feature statistics
        - feature_type: Type of feature ('room' or 'property')
        """
        # Convert to DataFrame for easier manipulation
        stats_list = []
        for feature, stats in feature_stats.items():
            row = {'feature': feature}
            row.update(stats)
            stats_list.append(row)
        
        if stats_list:
            stats_df = pd.DataFrame(stats_list)
            
            # Save to CSV
            stats_df.to_csv(os.path.join(self.output_dir, f'{feature_type}_feature_stats.csv'), index=False)
            
            # Save to JSON for more complex structures
            with open(os.path.join(self.output_dir, f'{feature_type}_feature_analysis.json'), 'w', encoding='utf-8') as f:
                json.dump(feature_stats, f, indent=2, default=self._json_serialize)

    def _save_trip_stats(self, trip_stats):
        """
        Save trip purpose statistics to a file.
        
        Parameters:
        - trip_stats: Dictionary with trip purpose statistics
        """
        # Convert to DataFrame for easier manipulation
        stats_list = []
        for trip_type, stats in trip_stats.items():
            row = {'trip_purpose': trip_type}
            
            # Handle nested dictionaries
            flat_stats = {}
            for key, value in stats.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        flat_stats[f"{key}_{sub_key}"] = sub_value
                else:
                    flat_stats[key] = value
                    
            row.update(flat_stats)
            stats_list.append(row)
        
        if stats_list:
            stats_df = pd.DataFrame(stats_list)
            
            # Save to CSV
            stats_df.to_csv(os.path.join(self.output_dir, 'trip_purpose_stats.csv'), index=False)
            
            # Save to JSON for more complex structures
            with open(os.path.join(self.output_dir, 'trip_purpose_analysis.json'), 'w', encoding='utf-8') as f:
                json.dump(trip_stats, f, indent=2, default=self._json_serialize)

    def _json_serialize(self, obj):
        """
        Helper function for JSON serialization.
        
        Parameters:
        - obj: Object to serialize
        
        Returns:
        - JSON serializable version of object
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return str(obj)

    def run_analysis(self, df=None):
        """
        Run the complete accommodation analysis pipeline.
        
        Parameters:
        - df: DataFrame with all reviews (optional)
        
        Returns:
        - Dictionary with analysis results
        """
        print("\n=== Running Accommodation Analysis ===")
        
        # Check if we have data
        if df is None:
            if self.sentiment_analyzer is not None and hasattr(self.sentiment_analyzer, 'reviews_df'):
                df = self.sentiment_analyzer.reviews_df
            else:
                print("No data available for accommodation analysis")
                return None
        
        if df is None or len(df) == 0:
            print("No data available for accommodation analysis")
            return None
        
        # Filter to accommodation reviews
        accommodation_df = self.filter_accommodation_reviews(df)
        
        if len(accommodation_df) == 0:
            print("No accommodation reviews found")
            return None
        
        # Run analyses
        accommodation_df, type_stats = self.analyze_accommodation_types(accommodation_df)
        accommodation_df, room_stats = self.analyze_room_features(accommodation_df)
        accommodation_df, property_stats = self.analyze_property_features(accommodation_df)
        accommodation_df, trip_stats = self.analyze_by_trip_purpose(accommodation_df)
        
        # Generate recommendations
        recommendations = self.generate_accommodation_recommendations(
            type_stats, room_stats, property_stats, trip_stats)
        
        # Compile results
        results = {
            'accommodation_types': type_stats,
            'room_features': room_stats,
            'property_features': property_stats,
            'traveler_preferences': trip_stats,
            'recommendations': recommendations
        }
        
        # Save overall results
        self._save_overall_results(results)
        
        print("\nAccommodation analysis complete.")
        
        return results
    
    def _save_overall_results(self, results):
        """
        Save overall analysis results to a file.
        
        Parameters:
        - results: Dictionary with all analysis results
        """
        # Save to JSON
        with open(os.path.join(self.output_dir, 'accommodation_analysis_summary.json'), 'w', encoding='utf-8') as f:
            # Create a simplified summary without large nested structures
            summary = {
                'data_volume': {
                    'accommodation_types': len(results['accommodation_types']),
                    'room_features': len(results['room_features']),
                    'property_features': len(results['property_features']),
                    'traveler_segments': len(results['traveler_preferences'])
                },
                'key_findings': {
                    'top_accommodation_type': max(results['accommodation_types'].items(), key=lambda x: x[1].get('mention_count', 0))[0] if results['accommodation_types'] else 'n/a',
                    'top_room_feature': max(results['room_features'].items(), key=lambda x: x[1].get('mention_count', 0))[0] if results['room_features'] else 'n/a',
                    'top_property_feature': max(results['property_features'].items(), key=lambda x: x[1].get('mention_count', 0))[0] if results['property_features'] else 'n/a'
                },
                'recommendation_count': {category: len(recs) for category, recs in results['recommendations'].items()}
            }
            
            json.dump(summary, f, indent=2)
    def run_island_analysis(self, df=None, island_col='island_category'):
        """
        Run accommodation analysis segmented by island groups.
        
        Parameters:
        - df: DataFrame with all reviews that includes island classification
        - island_col: Column name containing island categorization
        
        Returns:
        - Dictionary with analysis results by island
        """
        print("\n=== Running Island-Based Accommodation Analysis ===")
        
        # Check if we have data
        if df is None:
            if self.sentiment_analyzer is not None and hasattr(self.sentiment_analyzer, 'reviews_df'):
                df = self.sentiment_analyzer.reviews_df
            else:
                print("No data available for accommodation analysis")
                return None
        
        if df is None or len(df) == 0:
            print("No data available for accommodation analysis")
            return None
        
        # Check if we have the island column
        if island_col not in df.columns:
            print(f"Island column '{island_col}' not found in the data. Please run island analysis first.")
            return None
        
        # Filter to accommodation reviews
        all_accommodation_df = self.filter_accommodation_reviews(df)
        
        if len(all_accommodation_df) == 0:
            print("No accommodation reviews found")
            return None
        
        print(f"Found {len(all_accommodation_df)} accommodation reviews across all islands")
        
        # Get list of islands with enough accommodation reviews
        island_counts = all_accommodation_df[island_col].value_counts()
        valid_islands = island_counts[island_counts >= 10].index.tolist()
        
        if not valid_islands:
            print("No islands with sufficient accommodation reviews for analysis")
            return self.run_analysis(all_accommodation_df)  # Fall back to regular analysis
        
        print(f"Islands with sufficient accommodation reviews: {', '.join(valid_islands)}")
        
        # Create a directory for island-specific outputs
        island_output_dir = os.path.join(self.output_dir, 'island_analysis')
        if not os.path.exists(island_output_dir):
            os.makedirs(island_output_dir)
        
        # Store results for each island
        island_results = {}
        island_stats_comparison = {
            'accommodation_types': {},
            'room_features': {},
            'property_features': {},
            'avg_ratings': {}
        }
        
        # Run analysis for each island
        for island in valid_islands:
            print(f"\n--- Analyzing accommodations on {island} ---")
            
            # Create island-specific output directory
            island_dir = os.path.join(island_output_dir, island.replace(' ', '_').lower())
            if not os.path.exists(island_dir):
                os.makedirs(island_dir)
            
            # Save original output dir and temporarily set to island dir
            original_output_dir = self.output_dir
            self.output_dir = island_dir
            
            # Filter to this island's accommodation reviews
            island_df = all_accommodation_df[all_accommodation_df[island_col] == island].copy()
            
            print(f"Analyzing {len(island_df)} accommodation reviews for {island}")
            
            # Run analyses for this island
            island_df, type_stats = self.analyze_accommodation_types(island_df)
            island_df, room_stats = self.analyze_room_features(island_df)
            island_df, property_stats = self.analyze_property_features(island_df)
            island_df, trip_stats = self.analyze_by_trip_purpose(island_df)
            
            # Generate recommendations for this island
            recommendations = self.generate_accommodation_recommendations(
                type_stats, room_stats, property_stats, trip_stats)
            
            # Compile results for this island
            island_results[island] = {
                'accommodation_types': type_stats,
                'room_features': room_stats,
                'property_features': property_stats,
                'traveler_preferences': trip_stats,
                'recommendations': recommendations,
                'review_count': len(island_df),
                'avg_rating': float(island_df['rating'].mean()) if 'rating' in island_df.columns else None
            }
            
            # Collect data for comparison
            if 'rating' in island_df.columns:
                island_stats_comparison['avg_ratings'][island] = float(island_df['rating'].mean())
            
            # Collect top accommodation types
            for acc_type, stats in type_stats.items():
                if 'avg_sentiment' in stats:
                    if acc_type not in island_stats_comparison['accommodation_types']:
                        island_stats_comparison['accommodation_types'][acc_type] = {}
                    island_stats_comparison['accommodation_types'][acc_type][island] = stats['avg_sentiment']
            
            # Collect top room features
            for feature, stats in room_stats.items():
                if 'avg_sentiment' in stats:
                    if feature not in island_stats_comparison['room_features']:
                        island_stats_comparison['room_features'][feature] = {}
                    island_stats_comparison['room_features'][feature][island] = stats['avg_sentiment']
            
            # Collect top property features
            for feature, stats in property_stats.items():
                if 'avg_sentiment' in stats:
                    if feature not in island_stats_comparison['property_features']:
                        island_stats_comparison['property_features'][feature] = {}
                    island_stats_comparison['property_features'][feature][island] = stats['avg_sentiment']
            
            # Restore original output directory
            self.output_dir = original_output_dir
        
        # Generate cross-island comparison visualizations
        self._visualize_island_comparisons(island_stats_comparison, island_output_dir)
        
        # Save overall island results
        self._save_island_results(island_results, island_output_dir)
        
        print("\nIsland-based accommodation analysis complete.")
        
        self.visualize_island_accommodation_comparisons(island_results, island_output_dir)

        return island_results

    def _visualize_island_comparisons(self, comparison_data, output_dir):
        """
        Create visualizations comparing accommodation analysis across islands.
        
        Parameters:
        - comparison_data: Dictionary with cross-island comparisons
        - output_dir: Directory to save the visualizations
        """
        # Create visualizations directory
        viz_dir = os.path.join(output_dir, 'visualizations')
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        
        # 1. Compare average ratings by island
        if comparison_data['avg_ratings']:
            plt.figure(figsize=(12, 6))
            islands = list(comparison_data['avg_ratings'].keys())
            ratings = list(comparison_data['avg_ratings'].values())
            
            # Sort islands by average rating
            sorted_data = sorted(zip(islands, ratings), key=lambda x: x[1], reverse=True)
            sorted_islands, sorted_ratings = zip(*sorted_data)
            
            # Create bar chart
            bars = plt.bar(sorted_islands, sorted_ratings, color='skyblue')
            plt.title('Average Accommodation Ratings by Island', fontsize=14)
            plt.xlabel('Island')
            plt.ylabel('Average Rating (1-5)')
            plt.ylim(min(ratings) - 0.5 if min(ratings) > 0.5 else 0, 5.5)
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            
            # Add rating labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'island_accommodation_ratings.png'), dpi=300)
            plt.close()
        
        # 2. Compare room feature sentiment across islands
        if comparison_data['room_features']:
            # Convert to DataFrame for easier heatmap creation
            feature_data = {}
            
            for feature, island_values in comparison_data['room_features'].items():
                # Only include features with data from multiple islands and no None values
                if len(island_values) >= 2 and all(v is not None for v in island_values.values()):
                    feature_data[feature] = island_values
            
            if feature_data:
                df_features = pd.DataFrame(feature_data).T
                
                # Check if we have any valid data for the heatmap
                if not df_features.empty and not df_features.isna().all().all():
                    # Convert any remaining object dtypes to float
                    df_features = df_features.astype(float)
                    
                    # Create heatmap
                    plt.figure(figsize=(12, 8))
                    sns.heatmap(df_features, annot=True, cmap='RdYlGn', center=0, fmt='.2f')
                    plt.title('Room Feature Sentiment by Island', fontsize=14)
                    plt.ylabel('Room Feature')
                    plt.xlabel('Island')
                    plt.tight_layout()
                    plt.savefig(os.path.join(viz_dir, 'island_room_features_comparison.png'), dpi=300)
                    plt.close()
        
        # 3. Compare property feature sentiment across islands
        if comparison_data['property_features']:
            # Convert to DataFrame for easier heatmap creation
            feature_data = {}
            
            for feature, island_values in comparison_data['property_features'].items():
                # Only include features with data from multiple islands and no None values
                if len(island_values) >= 2 and all(v is not None for v in island_values.values()):
                    feature_data[feature] = island_values
            
            if feature_data:
                df_features = pd.DataFrame(feature_data).T
                
                # Check if we have any valid data for the heatmap
                if not df_features.empty and not df_features.isna().all().all():
                    # Convert any remaining object dtypes to float
                    df_features = df_features.astype(float)
                    
                    # Create heatmap
                    plt.figure(figsize=(12, 8))
                    sns.heatmap(df_features, annot=True, cmap='RdYlGn', center=0, fmt='.2f')
                    plt.title('Property Feature Sentiment by Island', fontsize=14)
                    plt.ylabel('Property Feature')
                    plt.xlabel('Island')
                    plt.tight_layout()
                    plt.savefig(os.path.join(viz_dir, 'island_property_features_comparison.png'), dpi=300)
                    plt.close()
        
        # 4. Compare accommodation types across islands
        if comparison_data['accommodation_types']:
            # Convert to DataFrame for easier heatmap creation
            type_data = {}
            
            for acc_type, island_values in comparison_data['accommodation_types'].items():
                # Only include types with data from multiple islands and no None values
                if len(island_values) >= 2 and all(v is not None for v in island_values.values()):
                    type_data[acc_type] = island_values
            
            if type_data:
                df_types = pd.DataFrame(type_data).T
                
                # Check if we have any valid data for the heatmap
                if not df_types.empty and not df_types.isna().all().all():
                    # Convert any remaining object dtypes to float
                    df_types = df_types.astype(float)
                    
                    # Create heatmap
                    plt.figure(figsize=(12, 6))
                    sns.heatmap(df_types, annot=True, cmap='RdYlGn', center=0, fmt='.2f')
                    plt.title('Accommodation Type Sentiment by Island', fontsize=14)
                    plt.ylabel('Accommodation Type')
                    plt.xlabel('Island')
                    plt.tight_layout()
                    plt.savefig(os.path.join(viz_dir, 'island_accommodation_types_comparison.png'), dpi=300)
                    plt.close()
        
        print(f"Island comparison visualizations saved to {viz_dir}")

    def _save_island_results(self, island_results, output_dir):
        """
        Save island-based analysis results.
        
        Parameters:
        - island_results: Dictionary with results by island
        - output_dir: Directory to save results
        """
        # Create a summary of island analysis
        summary = {}
        
        for island, results in island_results.items():
            summary[island] = {
                'review_count': results['review_count'],
                'avg_rating': results['avg_rating'],
                'top_accommodation_type': max(results['accommodation_types'].items(), 
                                            key=lambda x: x[1].get('mention_count', 0))[0] if results['accommodation_types'] else 'n/a',
                'top_room_feature': max(results['room_features'].items(), 
                                    key=lambda x: x[1].get('mention_count', 0))[0] if results['room_features'] else 'n/a',
                'top_property_feature': max(results['property_features'].items(), 
                                        key=lambda x: x[1].get('mention_count', 0))[0] if results['property_features'] else 'n/a',
                'key_recommendations': {
                    category: recs[:2] if len(recs) > 2 else recs 
                    for category, recs in results['recommendations'].items() if recs
                }
            }
        
        # Save summary to JSON
        with open(os.path.join(output_dir, 'island_accommodation_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=self._json_serialize)
        
        # Create a more detailed Excel report
        try:
            import openpyxl
            
            # Create Excel writer
            excel_path = os.path.join(output_dir, 'island_accommodation_comparison.xlsx')
            writer = pd.ExcelWriter(excel_path, engine='openpyxl')
            
            # Sheet 1: Overview
            overview_data = []
            for island, results in island_results.items():
                row = {
                    'Island': island,
                    'Reviews': results['review_count'],
                    'Avg Rating': results['avg_rating'] if results['avg_rating'] else 'N/A'
                }
                overview_data.append(row)
            
            overview_df = pd.DataFrame(overview_data)
            overview_df.to_excel(writer, sheet_name='Overview', index=False)
            
            # Sheet 2: Room Features
            room_data = []
            for island, results in island_results.items():
                for feature, stats in results['room_features'].items():
                    row = {
                        'Island': island,
                        'Feature': feature,
                        'Mentions': stats.get('mention_count', 0),
                        'Sentiment': stats.get('avg_sentiment', 'N/A'),
                        'Rating': stats.get('avg_rating', 'N/A')
                    }
                    room_data.append(row)
            
            if room_data:
                room_df = pd.DataFrame(room_data)
                room_df.to_excel(writer, sheet_name='Room Features', index=False)
            
            # Sheet 3: Property Features
            property_data = []
            for island, results in island_results.items():
                for feature, stats in results['property_features'].items():
                    row = {
                        'Island': island,
                        'Feature': feature,
                        'Mentions': stats.get('mention_count', 0),
                        'Sentiment': stats.get('avg_sentiment', 'N/A'),
                        'Rating': stats.get('avg_rating', 'N/A')
                    }
                    property_data.append(row)
            
            if property_data:
                property_df = pd.DataFrame(property_data)
                property_df.to_excel(writer, sheet_name='Property Features', index=False)
            
            # Sheet 4: Recommendations
            rec_data = []
            for island, results in island_results.items():
                for category, recs in results['recommendations'].items():
                    for i, rec in enumerate(recs):
                        row = {
                            'Island': island,
                            'Category': category,
                            'Recommendation': rec
                        }
                        rec_data.append(row)
            
            if rec_data:
                rec_df = pd.DataFrame(rec_data)
                rec_df.to_excel(writer, sheet_name='Recommendations', index=False)
            
            # Save and close
            writer.close()
            print(f"Island comparison Excel report saved to {excel_path}")
        
        except ImportError:
            print("openpyxl not found. Excel report not generated.")
            
            # Create CSV files instead
            csv_dir = os.path.join(output_dir, 'csv_reports')
            if not os.path.exists(csv_dir):
                os.makedirs(csv_dir)
            
            # Save summary as CSV
            summary_rows = []
            for island, data in summary.items():
                row = {'Island': island}
                row.update({k: v for k, v in data.items() if not isinstance(v, dict)})
                summary_rows.append(row)
            
            if summary_rows:
                pd.DataFrame(summary_rows).to_csv(os.path.join(csv_dir, 'island_summary.csv'), index=False)
                print(f"Island summary CSV saved to {csv_dir}")