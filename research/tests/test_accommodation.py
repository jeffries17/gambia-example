import sys
import os

# Add parent directory to the path to allow imports from the main project
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tonga_analysis.review_analyzer import TongaReviewAnalyzer
from tonga_analysis.sentiment_analyzer import SentimentAnalyzer
from tonga_analysis.accommodation_analyzer import AccommodationAnalyzer

def main():
    """
    Test the accommodation analysis module.
    """
    # Initialize base analyzer and load data
    base_analyzer = TongaReviewAnalyzer(
        data_dir='data',
        output_dir='outputs'
    )
    base_analyzer.load_data()
    
    if base_analyzer.all_reviews_df is None:
        print("Error: No review data loaded!")
        return
    
    # Initialize sentiment analyzer
    sentiment_analyzer = SentimentAnalyzer(output_dir='outputs')
    
    # Add sentiment analysis to the data
    df_with_sentiment, _ = sentiment_analyzer.run_sentiment_analysis(
        base_analyzer.all_reviews_df
    )
    
    # Initialize and run accommodation analyzer
    accommodation_analyzer = AccommodationAnalyzer(
        sentiment_analyzer=sentiment_analyzer,
        output_dir='outputs/accommodation_analysis'
    )
    
    results = accommodation_analyzer.run_analysis(df_with_sentiment)
    
    # Print key findings
    print("\nAccommodation Analysis Results:")
    
    print("\nRoom Features Analysis:")
    for feature, stats in results['room_features'].items():
        print(f"\n{feature.replace('_', ' ').title()}:")
        print(f"  Reviews: {stats['review_count']}")
        print(f"  Average Sentiment: {stats['avg_sentiment']:.3f}")
        print(f"  Positive Reviews: {stats['positive_reviews']}")
        print(f"  Negative Reviews: {stats['negative_reviews']}")
        if stats['avg_rating']:
            print(f"  Average Rating: {stats['avg_rating']:.2f}")
    
    print("\nProperty Features Analysis:")
    for feature, stats in results['property_features'].items():
        print(f"\n{feature.replace('_', ' ').title()}:")
        print(f"  Reviews: {stats['review_count']}")
        print(f"  Average Sentiment: {stats['avg_sentiment']:.3f}")
        print(f"  Positive Reviews: {stats['positive_reviews']}")
        print(f"  Negative Reviews: {stats['negative_reviews']}")
        if stats['avg_rating']:
            print(f"  Average Rating: {stats['avg_rating']:.2f}")
        
        print("  Most Common Phrases:")
        common_phrases = sorted(
            stats['common_phrases'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        for phrase, count in common_phrases:
            print(f"    - {phrase}: {count}")
    
    print("\nAccommodation Types Analysis:")
    for acc_type, stats in results['accommodation_types'].items():
        print(f"\n{acc_type.replace('_', ' ').title()}:")
        print(f"  Reviews: {stats['review_count']}")
        print(f"  Average Sentiment: {stats['avg_sentiment']:.3f}")
        if stats['avg_rating']:
            print(f"  Average Rating: {stats['avg_rating']:.2f}")
        
        print("  Top Features:")
        for feature, count in list(stats['top_features'].items())[:5]:
            print(f"    - {feature.replace('_', ' ').title()}: {count}")
    
    print("\nTraveler Preferences:")
    for trip_type, stats in results['traveler_preferences'].items():
        print(f"\n{trip_type}:")
        print(f"  Reviews: {stats['review_count']}")
        print(f"  Average Sentiment: {stats['avg_sentiment']:.3f}")
        if stats['avg_rating']:
            print(f"  Average Rating: {stats['avg_rating']:.2f}")
        
        print("  Preferred Features:")
        for feature, count in list(stats['preferred_features'].items())[:3]:
            print(f"    - {feature.replace('_', ' ').title()}: {count}")
    
    print("\nAnalysis outputs have been saved to:")
    print("- Detailed results: outputs/accommodation_analysis/accommodation_analysis.json")
    print("- Visualizations: outputs/accommodation_analysis/visualizations/")

if __name__ == "__main__":
    main()