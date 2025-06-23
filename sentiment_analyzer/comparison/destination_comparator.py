#!/usr/bin/env python3
"""
Destination Comparison System

This module provides functionality to compare sentiment analysis results
between different destinations using extracted TripAdvisor data.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import logging

# Import existing analyzers
import sys
RESEARCH_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'research', 'tonga_analysis')
if os.path.exists(RESEARCH_DIR):
    sys.path.append(RESEARCH_DIR)

logger = logging.getLogger(__name__)

class DestinationComparator:
    """
    Compare sentiment analysis results between multiple destinations.
    """
    
    def __init__(self, output_dir='outputs/destination_comparison'):
        """
        Initialize the destination comparator.
        
        Args:
            output_dir (str): Directory for comparison outputs
        """
        self.output_dir = output_dir
        self.destinations = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize analyzers (will import from existing codebase)
        self._setup_analyzers()
    
    def _setup_analyzers(self):
        """Set up the sentiment analyzers for different categories."""
        try:
            # Import your existing analyzers
            from accommodation_analyzer import AccommodationAnalyzer
            from attractions_analyzer import AttractionAnalyzer
            from sentiment_analyzer import SentimentAnalyzer
            
            self.sentiment_analyzer = SentimentAnalyzer(output_dir=self.output_dir)
            self.accommodation_analyzer = AccommodationAnalyzer(
                sentiment_analyzer=self.sentiment_analyzer,
                output_dir=os.path.join(self.output_dir, 'accommodations')
            )
            self.attraction_analyzer = AttractionAnalyzer(
                output_dir=os.path.join(self.output_dir, 'attractions')
            )
            
            logger.info("Successfully loaded existing analyzers")
            
        except ImportError as e:
            logger.warning(f"Could not import existing analyzers: {e}")
            # Create basic analyzer as fallback
            self.sentiment_analyzer = None
            self.accommodation_analyzer = None
            self.attraction_analyzer = None
    
    def add_destination(self, destination_name: str, tripadvisor_data: Dict):
        """
        Add a destination's TripAdvisor data for comparison.
        
        Args:
            destination_name (str): Name of the destination
            tripadvisor_data (Dict): Extracted TripAdvisor data from JSON
        """
        logger.info(f"Adding destination: {destination_name}")
        
        # Convert TripAdvisor data to DataFrame format
        reviews_df = self._convert_tripadvisor_to_dataframe(tripadvisor_data)
        
        # Store destination data
        self.destinations[destination_name] = {
            'raw_data': tripadvisor_data,
            'reviews_df': reviews_df,
            'business_info': tripadvisor_data.get('business_info', {}),
            'metadata': tripadvisor_data.get('extraction_metadata', {})
        }
        
        logger.info(f"Added {len(reviews_df)} reviews for {destination_name}")
    
    def load_destination_from_json(self, destination_name: str, json_file_path: str):
        """
        Load destination data from a TripAdvisor JSON file.
        
        Args:
            destination_name (str): Name to use for this destination
            json_file_path (str): Path to the TripAdvisor JSON file
        """
        logger.info(f"Loading destination {destination_name} from {json_file_path}")
        
        with open(json_file_path, 'r', encoding='utf-8') as f:
            tripadvisor_data = json.load(f)
        
        self.add_destination(destination_name, tripadvisor_data)
    
    def _convert_tripadvisor_to_dataframe(self, tripadvisor_data: Dict) -> pd.DataFrame:
        """
        Convert TripAdvisor JSON data to DataFrame format compatible with existing analyzers.
        
        Args:
            tripadvisor_data (Dict): Raw TripAdvisor data
            
        Returns:
            pd.DataFrame: Standardized DataFrame
        """
        reviews = tripadvisor_data.get('reviews', [])
        business_info = tripadvisor_data.get('business_info', {})
        
        # Convert to DataFrame
        df = pd.DataFrame(reviews)
        
        # Add business information to each review
        df['business_name'] = business_info.get('name', 'Unknown')
        df['business_category'] = business_info.get('category', 'other')
        df['business_location'] = business_info.get('location', 'Unknown')
        
        # Rename columns to match existing analyzer expectations
        column_mapping = {
            'text': 'text',
            'rating': 'rating',
            'date': 'date',
            'reviewer': 'reviewer',
            'trip_type': 'trip_type',
            'title': 'title',
            'id': 'id'
        }
        
        # Apply column mapping
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and old_col != new_col:
                df[new_col] = df[old_col]
        
        # Add required columns if missing
        if 'category' not in df.columns:
            df['category'] = business_info.get('category', 'other')
        
        # Add sentiment analysis if we have the analyzer
        if self.sentiment_analyzer:
            df = self._add_sentiment_analysis(df)
        
        return df
    
    def _add_sentiment_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add sentiment analysis to the DataFrame."""
        try:
            # Use existing sentiment analyzer
            df = self.sentiment_analyzer.analyze_sentiment(df)
            logger.info("Added sentiment analysis to reviews")
        except Exception as e:
            logger.warning(f"Could not add sentiment analysis: {e}")
            # Add basic sentiment analysis as fallback
            df = self._add_basic_sentiment(df)
        
        return df
    
    def _add_basic_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic sentiment analysis using TextBlob as fallback."""
        try:
            from textblob import TextBlob
            
            def get_sentiment(text):
                if pd.isna(text) or text == '':
                    return 0
                return TextBlob(str(text)).sentiment.polarity
            
            df['sentiment_score'] = df['text'].apply(get_sentiment)
            df['sentiment_category'] = pd.cut(
                df['sentiment_score'],
                bins=[-1, -0.1, 0.1, 1],
                labels=['negative', 'neutral', 'positive']
            )
            
            logger.info("Added basic sentiment analysis using TextBlob")
            
        except ImportError:
            logger.warning("TextBlob not available for sentiment analysis")
            # Create dummy sentiment scores based on ratings
            if 'rating' in df.columns:
                df['sentiment_score'] = (df['rating'] - 3) / 2
                df['sentiment_category'] = pd.cut(
                    df['sentiment_score'],
                    bins=[-1, -0.1, 0.1, 1],
                    labels=['negative', 'neutral', 'positive']
                )
        
        return df
    
    def compare_destinations(self, destination_names: List[str] = None) -> Dict:
        """
        Generate comprehensive comparison between destinations.
        
        Args:
            destination_names (List[str]): Specific destinations to compare (default: all)
            
        Returns:
            Dict: Comparison results
        """
        if destination_names is None:
            destination_names = list(self.destinations.keys())
        
        if len(destination_names) < 2:
            raise ValueError("Need at least 2 destinations for comparison")
        
        logger.info(f"Comparing destinations: {', '.join(destination_names)}")
        
        results = {
            'comparison_metadata': {
                'destinations': destination_names,
                'comparison_date': datetime.now().isoformat(),
                'total_reviews': sum(len(self.destinations[name]['reviews_df']) 
                                   for name in destination_names)
            },
            'overall_comparison': {},
            'category_comparison': {},
            'sentiment_comparison': {},
            'aspect_comparison': {}
        }
        
        # Overall comparison
        results['overall_comparison'] = self._compare_overall_metrics(destination_names)
        
        # Category-specific comparison
        results['category_comparison'] = self._compare_by_category(destination_names)
        
        # Sentiment comparison
        results['sentiment_comparison'] = self._compare_sentiment(destination_names)
        
        # Aspect-based comparison (if we have the analyzers)
        if self.accommodation_analyzer or self.attraction_analyzer:
            results['aspect_comparison'] = self._compare_aspects(destination_names)
        
        # Save results
        self._save_comparison_results(results, destination_names)
        
        # Generate visualizations
        self._create_comparison_visualizations(results, destination_names)
        
        return results
    
    def _compare_overall_metrics(self, destination_names: List[str]) -> Dict:
        """Compare overall metrics between destinations."""
        comparison = {}
        
        for name in destination_names:
            df = self.destinations[name]['reviews_df']
            business_info = self.destinations[name]['business_info']
            
            metrics = {
                'review_count': len(df),
                'avg_rating': df['rating'].mean() if 'rating' in df.columns else None,
                'business_name': business_info.get('name', 'Unknown'),
                'business_category': business_info.get('category', 'Unknown'),
                'location': business_info.get('location', 'Unknown')
            }
            
            # Add sentiment metrics if available
            if 'sentiment_score' in df.columns:
                metrics['avg_sentiment'] = df['sentiment_score'].mean()
                metrics['positive_percentage'] = (df['sentiment_category'] == 'positive').mean() * 100
                metrics['negative_percentage'] = (df['sentiment_category'] == 'negative').mean() * 100
            
            comparison[name] = metrics
        
        return comparison
    
    def _compare_by_category(self, destination_names: List[str]) -> Dict:
        """Compare destinations by business category."""
        category_comparison = {}
        
        # Group by category
        all_categories = set()
        for name in destination_names:
            df = self.destinations[name]['reviews_df']
            if 'category' in df.columns:
                all_categories.update(df['category'].unique())
        
        for category in all_categories:
            category_stats = {}
            
            for name in destination_names:
                df = self.destinations[name]['reviews_df']
                category_df = df[df['category'] == category] if 'category' in df.columns else df
                
                if len(category_df) > 0:
                    stats = {
                        'review_count': len(category_df),
                        'avg_rating': category_df['rating'].mean() if 'rating' in category_df.columns else None
                    }
                    
                    if 'sentiment_score' in category_df.columns:
                        stats['avg_sentiment'] = category_df['sentiment_score'].mean()
                    
                    category_stats[name] = stats
            
            if category_stats:
                category_comparison[category] = category_stats
        
        return category_comparison
    
    def _compare_sentiment(self, destination_names: List[str]) -> Dict:
        """Compare sentiment patterns between destinations."""
        sentiment_comparison = {}
        
        for name in destination_names:
            df = self.destinations[name]['reviews_df']
            
            if 'sentiment_score' in df.columns:
                sentiment_stats = {
                    'avg_sentiment': df['sentiment_score'].mean(),
                    'sentiment_std': df['sentiment_score'].std(),
                    'positive_count': (df['sentiment_category'] == 'positive').sum(),
                    'neutral_count': (df['sentiment_category'] == 'neutral').sum(),
                    'negative_count': (df['sentiment_category'] == 'negative').sum(),
                }
                
                # Sentiment by rating correlation
                if 'rating' in df.columns:
                    sentiment_stats['sentiment_rating_correlation'] = df['sentiment_score'].corr(df['rating'])
                
                sentiment_comparison[name] = sentiment_stats
        
        return sentiment_comparison
    
    def _compare_aspects(self, destination_names: List[str]) -> Dict:
        """Compare aspect-based sentiment between destinations."""
        aspect_comparison = {}
        
        for name in destination_names:
            df = self.destinations[name]['reviews_df']
            business_info = self.destinations[name]['business_info']
            category = business_info.get('category', 'other')
            
            try:
                if category == 'accommodation' and self.accommodation_analyzer:
                    # Use existing accommodation analyzer
                    aspect_results = self._analyze_accommodation_aspects(df)
                elif category == 'attraction' and self.attraction_analyzer:
                    # Use existing attraction analyzer
                    aspect_results = self._analyze_attraction_aspects(df)
                else:
                    # Basic aspect analysis
                    aspect_results = self._basic_aspect_analysis(df)
                
                aspect_comparison[name] = aspect_results
                
            except Exception as e:
                logger.warning(f"Could not analyze aspects for {name}: {e}")
                aspect_comparison[name] = {}
        
        return aspect_comparison
    
    def _analyze_accommodation_aspects(self, df: pd.DataFrame) -> Dict:
        """Analyze accommodation-specific aspects."""
        # Filter and analyze accommodation reviews
        accommodation_df = self.accommodation_analyzer.filter_accommodation_reviews(df)
        
        if len(accommodation_df) > 0:
            # Analyze room features
            _, room_stats = self.accommodation_analyzer.analyze_room_features(accommodation_df)
            
            # Analyze property features  
            _, property_stats = self.accommodation_analyzer.analyze_property_features(accommodation_df)
            
            return {
                'room_features': room_stats,
                'property_features': property_stats,
                'review_count': len(accommodation_df)
            }
        
        return {}
    
    def _analyze_attraction_aspects(self, df: pd.DataFrame) -> Dict:
        """Analyze attraction-specific aspects."""
        if len(df) > 0:
            # Use existing attraction analyzer methods
            activity_stats = self.attraction_analyzer.analyze_activity_types(df)
            experience_stats = self.attraction_analyzer.analyze_experience_aspects(df)
            
            return {
                'activities': activity_stats,
                'experiences': experience_stats,
                'review_count': len(df)
            }
        
        return {}
    
    def _basic_aspect_analysis(self, df: pd.DataFrame) -> Dict:
        """Basic aspect analysis for any category."""
        if 'text' not in df.columns or len(df) == 0:
            return {}
        
        # Define basic aspects
        basic_aspects = {
            'service': ['service', 'staff', 'friendly', 'helpful', 'professional'],
            'quality': ['quality', 'excellent', 'good', 'poor', 'bad'],
            'value': ['value', 'price', 'expensive', 'cheap', 'worth'],
            'location': ['location', 'convenient', 'central', 'close', 'far'],
            'cleanliness': ['clean', 'dirty', 'spotless', 'hygiene']
        }
        
        aspect_results = {}
        
        for aspect, keywords in basic_aspects.items():
            pattern = '|'.join(keywords)
            aspect_mask = df['text'].str.lower().str.contains(pattern, na=False)
            aspect_df = df[aspect_mask]
            
            if len(aspect_df) > 0:
                stats = {
                    'mention_count': len(aspect_df),
                    'mention_percentage': (len(aspect_df) / len(df)) * 100
                }
                
                if 'sentiment_score' in aspect_df.columns:
                    stats['avg_sentiment'] = aspect_df['sentiment_score'].mean()
                
                if 'rating' in aspect_df.columns:
                    stats['avg_rating'] = aspect_df['rating'].mean()
                
                aspect_results[aspect] = stats
        
        return aspect_results
    
    def _create_comparison_visualizations(self, results: Dict, destination_names: List[str]):
        """Create visualization charts for destination comparison."""
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create comparison charts
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Destination Comparison: {" vs ".join(destination_names)}', fontsize=16)
        
        # 1. Overall ratings comparison
        self._plot_ratings_comparison(axes[0, 0], results['overall_comparison'])
        
        # 2. Sentiment distribution
        self._plot_sentiment_comparison(axes[0, 1], results['sentiment_comparison'])
        
        # 3. Review count by category
        self._plot_category_comparison(axes[1, 0], results['category_comparison'])
        
        # 4. Top aspects comparison
        self._plot_aspects_comparison(axes[1, 1], results['aspect_comparison'])
        
        plt.tight_layout()
        
        # Save the comparison chart
        chart_path = os.path.join(self.output_dir, f'comparison_{"_vs_".join(destination_names)}.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Comparison visualizations saved to {chart_path}")
    
    def _plot_ratings_comparison(self, ax, overall_comparison: Dict):
        """Plot average ratings comparison."""
        destinations = list(overall_comparison.keys())
        ratings = [overall_comparison[dest].get('avg_rating', 0) for dest in destinations]
        
        bars = ax.bar(destinations, ratings, alpha=0.7)
        ax.set_title('Average Ratings Comparison')
        ax.set_ylabel('Average Rating')
        ax.set_ylim(0, 5)
        
        # Add value labels on bars
        for bar, rating in zip(bars, ratings):
            if rating > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                       f'{rating:.1f}', ha='center', va='bottom')
    
    def _plot_sentiment_comparison(self, ax, sentiment_comparison: Dict):
        """Plot sentiment score comparison."""
        destinations = list(sentiment_comparison.keys())
        sentiments = [sentiment_comparison[dest].get('avg_sentiment', 0) for dest in destinations]
        
        bars = ax.bar(destinations, sentiments, alpha=0.7, color='green')
        ax.set_title('Average Sentiment Comparison')
        ax.set_ylabel('Sentiment Score')
        ax.set_ylim(-1, 1)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        
        # Add value labels on bars
        for bar, sentiment in zip(bars, sentiments):
            if sentiment != 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{sentiment:.2f}', ha='center', va='bottom')
    
    def _plot_category_comparison(self, ax, category_comparison: Dict):
        """Plot review count by category."""
        if not category_comparison:
            ax.text(0.5, 0.5, 'No category data available', ha='center', va='center')
            ax.set_title('Category Comparison')
            return
        
        categories = list(category_comparison.keys())
        destinations = list(next(iter(category_comparison.values())).keys())
        
        x = np.arange(len(categories))
        width = 0.35
        
        for i, dest in enumerate(destinations):
            counts = [category_comparison[cat].get(dest, {}).get('review_count', 0) for cat in categories]
            ax.bar(x + i * width, counts, width, label=dest, alpha=0.7)
        
        ax.set_title('Review Count by Category')
        ax.set_xlabel('Category')
        ax.set_ylabel('Review Count')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(categories)
        ax.legend()
    
    def _plot_aspects_comparison(self, ax, aspect_comparison: Dict):
        """Plot top aspects comparison."""
        if not aspect_comparison:
            ax.text(0.5, 0.5, 'No aspect data available', ha='center', va='center')
            ax.set_title('Aspect Comparison')
            return
        
        # Get common aspects across destinations
        all_aspects = set()
        for dest_aspects in aspect_comparison.values():
            for category_aspects in dest_aspects.values():
                if isinstance(category_aspects, dict):
                    all_aspects.update(category_aspects.keys())
        
        if not all_aspects:
            ax.text(0.5, 0.5, 'No common aspects found', ha='center', va='center')
            ax.set_title('Aspect Comparison')
            return
        
        # Take top 5 most common aspects
        top_aspects = list(all_aspects)[:5]
        destinations = list(aspect_comparison.keys())
        
        x = np.arange(len(top_aspects))
        width = 0.35
        
        for i, dest in enumerate(destinations):
            scores = []
            for aspect in top_aspects:
                score = 0
                # Look for the aspect in any category for this destination
                for category_aspects in aspect_comparison[dest].values():
                    if isinstance(category_aspects, dict) and aspect in category_aspects:
                        score = category_aspects[aspect].get('avg_sentiment', 0)
                        break
                scores.append(score)
            
            ax.bar(x + i * width, scores, width, label=dest, alpha=0.7)
        
        ax.set_title('Top Aspects Sentiment Comparison')
        ax.set_xlabel('Aspect')
        ax.set_ylabel('Sentiment Score')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(top_aspects, rotation=45)
        ax.legend()
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    def _save_comparison_results(self, results: Dict, destination_names: List[str]):
        """Save comparison results to JSON file."""
        filename = f"comparison_{'_vs_'.join(destination_names)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        results_serializable = convert_numpy(results)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Comparison results saved to {filepath}")

# Convenience function for quick comparison
def compare_destinations_from_files(json_files: Dict[str, str], output_dir: str = 'outputs/comparison') -> Dict:
    """
    Quick function to compare destinations from TripAdvisor JSON files.
    
    Args:
        json_files (Dict[str, str]): Dictionary mapping destination names to JSON file paths
        output_dir (str): Output directory for comparison results
        
    Returns:
        Dict: Comparison results
    """
    comparator = DestinationComparator(output_dir=output_dir)
    
    # Load all destinations
    for dest_name, json_path in json_files.items():
        comparator.load_destination_from_json(dest_name, json_path)
    
    # Generate comparison
    return comparator.compare_destinations()

if __name__ == "__main__":
    # Example usage
    print("Destination Comparison Tool")
    print("==========================")
    
    # Get file paths
    files = {}
    while True:
        dest_name = input("Enter destination name (or 'done' to finish): ").strip()
        if dest_name.lower() == 'done':
            break
        
        json_path = input(f"Enter JSON file path for {dest_name}: ").strip()
        if os.path.exists(json_path):
            files[dest_name] = json_path
        else:
            print(f"File not found: {json_path}")
    
    if len(files) >= 2:
        try:
            results = compare_destinations_from_files(files)
            print(f"\nComparison complete!")
            print(f"Results saved to outputs/comparison/")
            
        except Exception as e:
            print(f"Error during comparison: {str(e)}")
    else:
        print("Need at least 2 destinations for comparison.") 