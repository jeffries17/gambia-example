import sys
import os

# Add parent directory to the path to allow imports from the main project
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tonga_analysis.review_analyzer import TongaReviewAnalyzer
from tonga_analysis.sentiment_analyzer import SentimentAnalyzer
from tonga_analysis.restaurant_analyzer import RestaurantAnalyzer

def main():
    """
    Test the restaurant analysis module.
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
    
    # Initialize and run restaurant analyzer
    restaurant_analyzer = RestaurantAnalyzer(
        sentiment_analyzer=sentiment_analyzer,
        output_dir='outputs/restaurant_analysis'
    )
    
    results = restaurant_analyzer.run_analysis(df_with_sentiment)
    
    # Print key findings
    print("\nRestaurant Analysis Results:")
    
    print("\nCuisine Preferences:")
    for cuisine, stats in results['cuisine_preferences'].items():
        print(f"\n{cuisine.replace('_', ' ').title()}:")
        print(f"  Reviews: {stats['review_count']}")
        print(f"  Average Sentiment: {stats['avg_sentiment']:.3f}")
        print(f"  Positive Reviews: {stats['positive_reviews']}")
        if stats['avg_rating']:
            print(f"  Average Rating: {stats['avg_rating']:.2f}")
    
    print("\nCultural Elements Analysis:")
    cultural = results['cultural_elements']
    print(f"Total Cultural Mentions: {cultural['review_count']}")
    print(f"Average Sentiment: {cultural['avg_sentiment']:.3f}")
    print(f"Positive Mentions: {cultural['positive_sentiment_count']}")
    print(f"Negative Mentions: {cultural['negative_sentiment_count']}")
    
    print("\nMost Common Cultural Phrases:")
    common_phrases = sorted(
        cultural['common_phrases'].items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    for phrase, count in common_phrases:
        print(f"  {phrase}: {count}")
    
    print("\nMeal-specific Analysis:")
    for meal_type, stats in results['meal_analysis'].items():
        print(f"\n{meal_type.title()}:")
        print(f"  Reviews: {stats['review_count']}")
        print(f"  Average Sentiment: {stats['avg_sentiment']:.3f}")
        if stats['avg_rating']:
            print(f"  Average Rating: {stats['avg_rating']:.2f}")
        print("  Top Aspects Mentioned:")
        for aspect, count in stats['top_aspects'].items():
            print(f"    - {aspect}: {count}")
    
    print("\nAnalysis outputs have been saved to:")
    print("- Detailed results: outputs/restaurant_analysis/restaurant_analysis.json")
    print("- Visualizations: outputs/restaurant_analysis/visualizations/")

if __name__ == "__main__":
    main()