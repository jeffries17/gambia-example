import sys
import os

# Add parent directory to the path to allow imports from the main project
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tonga_analysis.review_analyzer import TongaReviewAnalyzer
from tonga_analysis.sentiment_analyzer import SentimentAnalyzer
from tonga_analysis.attractions_analyzer import AttractionAnalyzer

def main():
    """
    Test the attraction analysis module.
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
    
    # Initialize and run attraction analyzer
    attraction_analyzer = AttractionAnalyzer(
        sentiment_analyzer=sentiment_analyzer,
        output_dir='outputs/attraction_analysis'
    )
    
    results = attraction_analyzer.run_analysis(df_with_sentiment)
    
    # Print key findings
    print("\nAttraction Analysis Results:")
    
    for key, value in results.items():
        print(f"\n{key.replace('_', ' ').title()} Analysis:")
        for subkey, stats in value.items():
            print(f"\n{subkey.replace('_', ' ').title()}:")
            print(f"  Reviews: {stats.get('review_count', 0)}")
            print(f"  Average Sentiment: {stats.get('avg_sentiment', 0.0):.3f}")
            print(f"  Positive Reviews: {stats.get('positive_reviews', 0)}")
            print(f"  Negative Reviews: {stats.get('negative_reviews', 0)}")

            avg_rating = stats.get('avg_rating')
            if avg_rating is not None:
                print(f"  Average Rating: {avg_rating:.2f}")

            common_phrases = stats.get('common_phrases', {})
            if common_phrases:
                print("  Most Common Phrases:")
                sorted_phrases = sorted(common_phrases.items(), key=lambda x: x[1], reverse=True)[:5]
                for phrase, count in sorted_phrases:
                    print(f"    - {phrase}: {count}")

    print("\nAnalysis outputs have been saved to:")
    print("- Detailed results: outputs/attraction_analysis/attraction_analysis.json")
    print("- Visualizations: outputs/attraction_analysis/visualizations/")

if __name__ == "__main__":
    main()
