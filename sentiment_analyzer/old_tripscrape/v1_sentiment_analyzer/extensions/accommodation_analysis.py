import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import re
import json

class AccommodationAnalysis:
    """
    Extension class for analyzing accommodation-specific reviews in Tonga tourism data.
    Focuses on hotels, resorts, guest houses, and other lodging experiences.
    """
    
    def __init__(self, base_analyzer, output_dir='accommodation_insights'):
        """
        Initialize with reference to the base analyzer.
        
        Parameters:
        - base_analyzer: Base TongaTourismAnalysis instance
        - output_dir: Directory to save accommodation-specific outputs
        """
        self.analyzer = base_analyzer
        
        # Create full path if output_dir is relative
        if not os.path.isabs(output_dir):
            self.output_dir = os.path.join(base_analyzer.output_dir, output_dir)
        else:
            self.output_dir = output_dir
            
        # Create output directory
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created accommodation analysis directory: {self.output_dir}")
            
        # Accommodation-specific categories setup
        self.setup_accommodation_categories()
        
    def setup_accommodation_categories(self):
        """
        Set up accommodation-specific categories for analysis.
        """
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
        
        # Define room and property features
        self.room_features = {
            'cleanliness': ['clean', 'dirt', 'dust', 'spotless', 'tidy', 'hygiene', 'sanitary', 'stain', 'smell'],
            'comfort': ['comfort', 'comfortable', 'bed', 'mattress', 'pillow', 'soft', 'hard', 'sleep', 'rest'],
            'size': ['size', 'space', 'spacious', 'tiny', 'large', 'small', 'big', 'cramped', 'roomy'],
            'bathroom': ['bathroom', 'shower', 'toilet', 'bath', 'hot water', 'pressure', 'sink', 'towels'],
            'amenities': ['amenity', 'tv', 'wifi', 'internet', 'fridge', 'minibar', 'kettle', 'coffee', 'tea', 'safe', 'closet'],
            'view': ['view', 'ocean', 'sea', 'beach', 'scenic', 'window', 'overlook', 'balcony', 'patio'],
            'noise': ['noise', 'quiet', 'peaceful', 'loud', 'disturb', 'sleep', 'soundproof', 'hear', 'traffic']
        }
        
        # Define property features
        self.property_features = {
            'location': ['location', 'central', 'convenient', 'beach', 'downtown', 'access', 'close', 'distance', 'walk', 'near', 'far'],
            'service': ['service', 'staff', 'reception', 'front desk', 'friendly', 'helpful', 'professional', 'manager', 'concierge'],
            'facilities': ['facility', 'pool', 'gym', 'fitness', 'spa', 'restaurant', 'bar', 'lounge', 'garden', 'parking', 'business'],
            'breakfast': ['breakfast', 'buffet', 'morning meal', 'continental', 'coffee', 'fruit', 'cereal', 'eggs'],
            'wifi': ['wifi', 'internet', 'connection', 'online', 'slow', 'fast', 'reliable'],
            'safety': ['safe', 'security', 'secure', 'lock', 'staff', 'camera', 'theft', 'steal', 'emergency'],
            'value': ['value', 'price', 'worth', 'expensive', 'cheap', 'affordable', 'cost', 'overpriced', 'budget']
        }
    
    def verify_accommodation_data(self, df):
        """
        Verify that we're dealing with accommodation reviews by checking the URL path.
        TripAdvisor uses Hotel_Review in the URL for accommodation.
        
        Parameters:
        - df: DataFrame with review data
        
        Returns:
        - Boolean indicating if reviews are for accommodations
        """
        # Check webUrl path for Hotel_Review
        if 'placeInfo.webUrl' in df.columns:
            has_hotel_url = df['placeInfo.webUrl'].str.contains('Hotel_Review', case=False, na=False)
            
            if has_hotel_url.any():
                # If any reviews have Hotel_Review in URL, verify how many
                hotel_count = has_hotel_url.sum()
                total_count = len(df)
                
                if hotel_count < total_count:
                    print(f"Warning: Only {hotel_count} out of {total_count} reviews are verified as accommodation")
                else:
                    print(f"Verified all {total_count} reviews are for accommodation")
                return True
                
        # If URL check fails, assume it's accommodation data since we're in the accommodation analyzer
        print("Warning: Could not verify accommodation reviews through URL pattern")
        return True
    
    def analyze_accommodation_types(self, df):
        """
        Analyze different types of accommodations mentioned in reviews.
        
        Parameters:
        - df: DataFrame with accommodation reviews
        
        Returns:
        - DataFrame with accommodation type analysis
        """
        print("Analyzing accommodation types mentioned in reviews...")
        
        # For each accommodation type, check if any keywords are present
        for accom_type, keywords in self.accommodation_types.items():
            pattern = r'\b(' + '|'.join(keywords) + r')\b'
            df[f'accom_{accom_type}'] = df['processed_text'].astype(str).str.contains(
                pattern, case=False, regex=True).astype(int)
        
        # Calculate accommodation type mention counts
        accom_cols = [f'accom_{accom_type}' for accom_type in self.accommodation_types.keys()]
        type_counts = df[accom_cols].sum().sort_values(ascending=False)
        
        print("Accommodation type mentions:")
        for accom_type, count in type_counts.items():
            type_name = accom_type.replace('accom_', '')
            print(f"  {type_name}: {count} mentions")
        
        # Create a bar chart of accommodation type mentions
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x=type_counts.index, y=type_counts.values, palette='viridis')
        plt.title('Accommodation Types Mentioned in Reviews')
        plt.xlabel('Accommodation Type')
        plt.ylabel('Number of Mentions')
        plt.xticks(rotation=45, ha='right')
        
        # Add count labels
        for i, count in enumerate(type_counts.values):
            ax.text(i, count + 0.5, str(count), ha='center')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/accommodation_type_mentions.png')
        print(f"Saved accommodation type chart to {self.output_dir}/accommodation_type_mentions.png")
        
        return df
    
    def analyze_room_features(self, df):
        """
        Analyze room features mentioned in accommodation reviews.
        
        Parameters:
        - df: DataFrame with accommodation reviews
        
        Returns:
        - DataFrame with room feature analysis
        """
        print("Analyzing room features mentioned in reviews...")
        
        # For each room feature, check if any keywords are present
        for feature, keywords in self.room_features.items():
            pattern = r'\b(' + '|'.join(keywords) + r')\b'
            df[f'room_{feature}'] = df['processed_text'].astype(str).str.contains(
                pattern, case=False, regex=True).astype(int)
        
        # Calculate room feature mention counts
        room_cols = [f'room_{feature}' for feature in self.room_features.keys()]
        feature_counts = df[room_cols].sum().sort_values(ascending=False)
        
        print("Room feature mentions:")
        for feature, count in feature_counts.items():
            feature_name = feature.replace('room_', '')
            print(f"  {feature_name}: {count} mentions")
        
        # Calculate average sentiment for each room feature
        feature_sentiment = {}
        for feature in self.room_features.keys():
            col = f'room_{feature}'
            # Only consider reviews that mention this feature
            feature_df = df[df[col] == 1]
            if len(feature_df) > 0:
                avg_sentiment = feature_df['sentiment_score'].mean()
                feature_sentiment[feature] = avg_sentiment
        
        if feature_sentiment:
            # Create a bar chart of room feature sentiment
            plt.figure(figsize=(12, 6))
            features = list(feature_sentiment.keys())
            sentiment_values = list(feature_sentiment.values())
            colors = ['green' if v > 0 else 'red' for v in sentiment_values]
            
            bars = plt.bar(features, sentiment_values, color=colors)
            plt.title('Average Sentiment for Room Features')
            plt.xlabel('Room Feature')
            plt.ylabel('Average Sentiment Score (-1 to 1)')
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            
            # Add sentiment value labels
            for bar, value in zip(bars, sentiment_values):
                label_position = value + 0.02 if value > 0 else value - 0.08
                plt.text(bar.get_x() + bar.get_width()/2., label_position,
                       f'{value:.2f}', ha='center', va='bottom', color='black', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/room_feature_sentiment.png')
            print(f"Saved room feature sentiment chart to {self.output_dir}/room_feature_sentiment.png")
        
        return df
    
    def analyze_property_features(self, df):
        """
        Analyze property-wide features mentioned in accommodation reviews.
        
        Parameters:
        - df: DataFrame with accommodation reviews
        
        Returns:
        - DataFrame with property feature analysis
        """
        print("Analyzing property features mentioned in reviews...")
        
        # For each property feature, check if any keywords are present
        for feature, keywords in self.property_features.items():
            pattern = r'\b(' + '|'.join(keywords) + r')\b'
            df[f'property_{feature}'] = df['processed_text'].astype(str).str.contains(
                pattern, case=False, regex=True).astype(int)
        
        # Calculate property feature mention counts
        property_cols = [f'property_{feature}' for feature in self.property_features.keys()]
        feature_counts = df[property_cols].sum().sort_values(ascending=False)
        
        print("Property feature mentions:")
        for feature, count in feature_counts.items():
            feature_name = feature.replace('property_', '')
            print(f"  {feature_name}: {count} mentions")
        
        # Calculate average sentiment for each property feature
        feature_sentiment = {}
        for feature in self.property_features.keys():
            col = f'property_{feature}'
            # Only consider reviews that mention this feature
            feature_df = df[df[col] == 1]
            if len(feature_df) > 0:
                avg_sentiment = feature_df['sentiment_score'].mean()
                feature_sentiment[feature] = avg_sentiment
        
        if feature_sentiment:
            # Create a bar chart of property feature sentiment
            plt.figure(figsize=(12, 6))
            features = list(feature_sentiment.keys())
            sentiment_values = list(feature_sentiment.values())
            colors = ['green' if v > 0 else 'red' for v in sentiment_values]
            
            bars = plt.bar(features, sentiment_values, color=colors)
            plt.title('Average Sentiment for Property Features')
            plt.xlabel('Property Feature')
            plt.ylabel('Average Sentiment Score (-1 to 1)')
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            
            # Add sentiment value labels
            for bar, value in zip(bars, sentiment_values):
                label_position = value + 0.02 if value > 0 else value - 0.08
                plt.text(bar.get_x() + bar.get_width()/2., label_position,
                       f'{value:.2f}', ha='center', va='bottom', color='black', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/property_feature_sentiment.png')
            print(f"Saved property feature sentiment chart to {self.output_dir}/property_feature_sentiment.png")
        
        return df
    
    def analyze_by_trip_purpose(self, df):
        """
        Analyze accommodation preferences and sentiment by trip purpose (trip type).
        
        Parameters:
        - df: DataFrame with accommodation reviews
        
        Returns:
        - DataFrame with trip purpose analysis
        """
        if 'trip_type_standard' not in df.columns:
            print("Trip type information not available for trip purpose analysis")
            return df
        
        print("Analyzing accommodation preferences by trip purpose...")
        
        # Get valid trip types with enough data
        trip_type_counts = df['trip_type_standard'].value_counts()
        valid_trip_types = trip_type_counts[trip_type_counts >= 3].index.tolist()
        
        if not valid_trip_types:
            print("Not enough data for trip purpose analysis")
            return df
            
        # Filter to segments with enough data
        segment_df = df[df['trip_type_standard'].isin(valid_trip_types)]
        
        # For each valid trip type, analyze what features are mentioned most
        all_feature_cols = []
        all_feature_cols.extend([f'room_{feature}' for feature in self.room_features.keys()])
        all_feature_cols.extend([f'property_{feature}' for feature in self.property_features.keys()])
        
        # Calculate the percentage of each segment that mentions each feature
        try:
            segment_preferences = segment_df.groupby('trip_type_standard')[all_feature_cols].mean()
            
            # Clean up column names for display
            new_cols = {}
            for col in segment_preferences.columns:
                if col.startswith('room_'):
                    new_cols[col] = 'Room: ' + col.replace('room_', '')
                elif col.startswith('property_'):
                    new_cols[col] = 'Property: ' + col.replace('property_', '')
                    
            segment_preferences = segment_preferences.rename(columns=new_cols)
            
            print("Feature preferences by trip purpose:")
            print(segment_preferences)
            
            # Create a heatmap of feature preferences by segment
            plt.figure(figsize=(16, 10))
            sns.heatmap(segment_preferences, annot=True, cmap='YlGnBu', fmt='.2f')
            plt.title('Accommodation Feature Preferences by Trip Purpose')
            plt.ylabel('Trip Purpose')
            plt.xlabel('Accommodation Feature')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/trip_purpose_preferences.png')
            print(f"Saved trip purpose preferences chart to {self.output_dir}/trip_purpose_preferences.png")
            
            # Get top features for each trip purpose
            trip_top_features = {}
            for trip_type in segment_preferences.index:
                trip_data = segment_preferences.loc[trip_type]
                top_features = trip_data.nlargest(3)
                trip_top_features[trip_type] = dict(top_features)
                
            print("\nTop accommodation features by trip purpose:")
            for trip_type, features in trip_top_features.items():
                print(f"\n  {trip_type}:")
                for feature, value in features.items():
                    print(f"    - {feature} ({value:.2f})")
            
            # Save to file
            segment_preferences.to_csv(f'{self.output_dir}/trip_purpose_preferences.csv')
            
            # Calculate average sentiment by trip purpose
            sentiment_by_trip = segment_df.groupby('trip_type_standard')['sentiment_score'].mean().reset_index()
            sentiment_by_trip = sentiment_by_trip.sort_values('sentiment_score', ascending=False)
            
            # Create bar chart of sentiment by trip purpose
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x='trip_type_standard', y='sentiment_score', data=sentiment_by_trip, palette='viridis')
            plt.title('Average Sentiment by Trip Purpose')
            plt.xlabel('Trip Purpose')
            plt.ylabel('Average Sentiment Score (-1 to 1)')
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            
            # Add sentiment value labels
            for i, row in enumerate(sentiment_by_trip.itertuples()):
                ax.text(i, row.sentiment_score + 0.01, f'{row.sentiment_score:.2f}', 
                      ha='center', va='bottom', color='black', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/sentiment_by_trip_purpose.png')
            print(f"Saved trip purpose sentiment chart to {self.output_dir}/sentiment_by_trip_purpose.png")
            
        except Exception as e:
            print(f"Error in trip purpose analysis: {str(e)}")
        
        return df
    
    def analyze_accommodation_satisfaction(self, df):
        """
        Analyze overall satisfaction with different accommodation types.
        
        Parameters:
        - df: DataFrame with accommodation reviews
        
        Returns:
        - DataFrame with accommodation satisfaction analysis
        """
        print("Analyzing satisfaction with different accommodation types...")
        
        accom_cols = [f'accom_{accom_type}' for accom_type in self.accommodation_types.keys()]
        
        # Calculate sentiment and rating (if available) for each accommodation type
        accom_metrics = []
        
        for col in accom_cols:
            accom_type = col.replace('accom_', '')
            accom_reviews = df[df[col] == 1]
            
            if len(accom_reviews) >= 3:  # Only analyze if we have enough data
                metrics = {
                    'accommodation_type': accom_type,
                    'mentions': len(accom_reviews),
                    'sentiment': accom_reviews['sentiment_score'].mean()
                }
                
                # Add rating if available
                if 'rating' in accom_reviews.columns:
                    metrics['rating'] = accom_reviews['rating'].mean()
                
                accom_metrics.append(metrics)
        
        if not accom_metrics:
            print("Not enough data for accommodation satisfaction analysis")
            return df
        
        # Convert to DataFrame and sort by sentiment
        metrics_df = pd.DataFrame(accom_metrics)
        metrics_df = metrics_df.sort_values('sentiment', ascending=False)
        
        print("Accommodation satisfaction metrics:")
        print(metrics_df)
        
        # Create visualization of accommodation satisfaction
        plt.figure(figsize=(12, 8))
        
        # Create a subplot for sentiment
        plt.subplot(2, 1, 1)
        bars = plt.barh(metrics_df['accommodation_type'], metrics_df['sentiment'], color='skyblue')
        plt.title('Average Sentiment by Accommodation Type')
        plt.xlabel('Average Sentiment Score (-1 to 1)')
        plt.axvline(x=0, color='r', linestyle='-', alpha=0.3)
        
        # Add sentiment labels
        for bar, value in zip(bars, metrics_df['sentiment']):
            plt.text(value + 0.02, bar.get_y() + bar.get_height()/2, 
                   f'{value:.2f}', va='center', color='black', fontweight='bold')
        
        # Create a subplot for rating if available
        if 'rating' in metrics_df.columns:
            plt.subplot(2, 1, 2)
            bars = plt.barh(metrics_df['accommodation_type'], metrics_df['rating'], color='lightgreen')
            plt.title('Average Rating by Accommodation Type')
            plt.xlabel('Average Rating (1-5)')
            plt.xlim(1, 5)
            
            # Add rating labels
            for bar, value in zip(bars, metrics_df['rating']):
                plt.text(value + 0.1, bar.get_y() + bar.get_height()/2, 
                       f'{value:.2f}', va='center', color='black', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/accommodation_satisfaction.png')
        print(f"Saved accommodation satisfaction chart to {self.output_dir}/accommodation_satisfaction.png")
        
        # Save metrics to file
        metrics_df.to_csv(f'{self.output_dir}/accommodation_satisfaction_metrics.csv', index=False)
        
        return df
    
    def generate_accommodation_recommendations(self, df):
        """
        Generate accommodation-specific recommendations based on the analysis.
        
        Parameters:
        - df: DataFrame with accommodation reviews and analysis
        
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
        accom_cols = [f'accom_{accom_type}' for accom_type in self.accommodation_types.keys()]
        if any(col in df.columns for col in accom_cols):
            # Calculate sentiment for each accommodation type
            sentiment_by_accom = {}
            for accom_type, col in zip(self.accommodation_types.keys(), accom_cols):
                if col in df.columns and df[col].sum() >= 3:
                    sentiment = df[df[col] == 1]['sentiment_score'].mean()
                    sentiment_by_accom[accom_type] = sentiment
            
            if sentiment_by_accom:
                # Identify most positive and negative types
                most_positive = max(sentiment_by_accom.items(), key=lambda x: x[1])
                most_negative = min(sentiment_by_accom.items(), key=lambda x: x[1])
                
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
        room_cols = [f'room_{feature}' for feature in self.room_features.keys()]
        if any(col in df.columns for col in room_cols):
            # Calculate sentiment for each room feature
            sentiment_by_feature = {}
            for feature, col in zip(self.room_features.keys(), room_cols):
                if col in df.columns and df[col].sum() >= 3:
                    sentiment = df[df[col] == 1]['sentiment_score'].mean()
                    sentiment_by_feature[feature] = sentiment
            
            if sentiment_by_feature:
                # Identify features with negative sentiment
                negative_features = {k: v for k, v in sentiment_by_feature.items() if v < 0}
                for feature, sentiment in negative_features.items():
                    recommendations["room_improvements"].append(
                        f"Address issues with {feature.replace('_', ' ')} which has negative sentiment ({sentiment:.2f})"
                    )
                
                # Identify features with very positive sentiment
                positive_features = {k: v for k, v in sentiment_by_feature.items() if v > 0.3}
                for feature, sentiment in positive_features.items():
                    recommendations["marketing"].append(
                        f"Emphasize {feature.replace('_', ' ')} in marketing as a strength ({sentiment:.2f})"
                    )
        
        # Property feature recommendations
        property_cols = [f'property_{feature}' for feature in self.property_features.keys()]
        if any(col in df.columns for col in property_cols):
            # Calculate sentiment for each property feature
            sentiment_by_feature = {}
            for feature, col in zip(self.property_features.keys(), property_cols):
                if col in df.columns and df[col].sum() >= 3:
                    sentiment = df[df[col] == 1]['sentiment_score'].mean()
                    sentiment_by_feature[feature] = sentiment
            
            if sentiment_by_feature:
                # Identify features with negative sentiment
                negative_features = {k: v for k, v in sentiment_by_feature.items() if v < 0}
                for feature, sentiment in negative_features.items():
                    recommendations["property_improvements"].append(
                        f"Improve {feature.replace('_', ' ')} which has negative sentiment ({sentiment:.2f})"
                    )
        
        # Visitor segment recommendations
        if 'trip_type_standard' in df.columns:
            try:
                # Calculate sentiment by trip type
                trip_sentiment = df.groupby('trip_type_standard')['sentiment_score'].mean()
                
                # Identify trip types with negative sentiment
                negative_trips = {k: v for k, v in trip_sentiment.items() if v < 0 and pd.notna(k) and k != 'unknown'}
                for trip, sentiment in negative_trips.items():
                    recommendations["visitor_segments"].append(
                        f"Focus on improving accommodations for {trip} travelers "
                        f"who report negative experiences ({sentiment:.2f})"
                    )
                
                # Identify trip types with positive sentiment
                positive_trips = {k: v for k, v in trip_sentiment.items() if v > 0.3 and pd.notna(k) and k != 'unknown'}
                for trip, sentiment in positive_trips.items():
                    recommendations["marketing"].append(
                        f"Target marketing towards {trip} travelers "
                        f"who particularly enjoy Tonga accommodations ({sentiment:.2f})"
                    )
                
                # Analyze feature preferences by trip type
                all_feature_cols = []
                all_feature_cols.extend([f'room_{feature}' for feature in self.room_features.keys()])
                all_feature_cols.extend([f'property_{feature}' for feature in self.property_features.keys()])
                
                if all_feature_cols:
                    segment_preferences = df.groupby('trip_type_standard')[all_feature_cols].mean()
                    
                    for trip_type in segment_preferences.index:
                        if pd.notna(trip_type) and trip_type != 'unknown':
                            trip_data = segment_preferences.loc[trip_type]
                            top_feature_col = trip_data.idxmax()
                            
                            if top_feature_col.startswith('room_'):
                                feature_type = 'room'
                                feature_name = top_feature_col.replace('room_', '')
                            else:
                                feature_type = 'property'
                                feature_name = top_feature_col.replace('property_', '')
                                
                            feature_score = trip_data.max()
                            
                            if feature_score > 0.2:  # If prominent
                                recommendations["visitor_segments"].append(
                                    f"For {trip_type} travelers, prioritize {feature_type} "
                                    f"{feature_name.replace('_', ' ')} features ({feature_score:.2f})"
                                )
            except Exception as e:
                print(f"Error generating visitor segment recommendations: {str(e)}")
        
        # Save recommendations
        with open(f'{self.output_dir}/accommodation_recommendations.json', 'w', encoding='utf-8') as f:
            json.dump(recommendations, f, indent=2)
        
        # Print key recommendations
        print("\nKey accommodation recommendations:")
        for category, recs in recommendations.items():
            if recs:
                print(f"\n{category.replace('_', ' ').title()}:")
                for rec in recs:
                    print(f"  - {rec}")
        
        return recommendations
    
    def run_analysis(self, df=None):
        """
        Run the complete accommodation analysis.
        
        Parameters:
        - df: DataFrame with review data (optional, will get from base analyzer if None)
        
        Returns:
        - DataFrame with accommodation analysis results
        """
        if df is None:
            # Get the processed data from the base analyzer
            df = self.analyzer.get_processed_data()
            
        if df is None or len(df) == 0:
            print("No data available for accommodation analysis")
            return None
        
        print("\n=== Running Accommodation Analysis ===")
        
        # Assuming df contains only accommodation-related data
        accommodation_df = df
        
        if len(accommodation_df) == 0:
            print("No accommodation reviews found in the data")
            return df
        
        # Run accommodation-specific analyses
        accommodation_df = self.analyze_accommodation_types(accommodation_df)
        accommodation_df = self.analyze_room_features(accommodation_df)
        accommodation_df = self.analyze_property_features(accommodation_df)
        accommodation_df = self.analyze_by_trip_purpose(accommodation_df)
        accommodation_df = self.analyze_accommodation_satisfaction(accommodation_df)
        
        # Generate accommodation-specific recommendations
        self.generate_accommodation_recommendations(accommodation_df)
        
        print("\nAccommodation analysis complete.")
        return accommodation_df
