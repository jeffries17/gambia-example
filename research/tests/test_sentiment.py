import sys
import os

# Add parent directory to the path to allow imports from the main project
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tonga_analysis.review_analyzer import TongaReviewAnalyzer
from tonga_analysis.sentiment_analyzer import SentimentAnalyzer

def main():
    """
    Test the sentiment analysis module with our review data.
    """
    # First load the data using our base analyzer
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
    
    # Run sentiment analysis
    df_with_sentiment, results = sentiment_analyzer.run_sentiment_analysis(
        base_analyzer.all_reviews_df
    )
    
    # Print key findings
    print("\nSentiment Analysis Results:")
    print(f"\nOverall Statistics:")
    print(f"Average Sentiment Score: {results['summary_stats']['average_sentiment']:.3f}")
    print(f"Standard Deviation: {results['summary_stats']['sentiment_std']:.3f}")
    print("\nReview Counts by Sentiment:")
    print(f"Positive Reviews: {results['summary_stats']['positive_reviews']}")
    print(f"Neutral Reviews: {results['summary_stats']['neutral_reviews']}")
    print(f"Negative Reviews: {results['summary_stats']['negative_reviews']}")
    
    if 'trends' in results and 'category' in results['trends']:
        print("\nSentiment by Category:")
        for stats in results['trends']['category']:
            print(f"- {stats['name']}: {stats['mean_sentiment']:.3f} "
                  f"(from {stats['review_count']} reviews)")
            print(f"  Positive words: {stats['positive_words']}, "
                  f"Negative words: {stats['negative_words']}")
    
    print("\nCommon Phrases in Positive Reviews:")
    if 'positive' in results['common_phrases']:
        top_positive = sorted(
            results['common_phrases']['positive'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        for word, count in top_positive:
            print(f"- {word}: {count}")
    
    print("\nCommon Phrases in Negative Reviews:")
    if 'negative' in results['common_phrases']:
        top_negative = sorted(
            results['common_phrases']['negative'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        for word, count in top_negative:
            print(f"- {word}: {count}")
    
    print("\nAnalysis outputs have been saved to:")
    print("- Visualizations: outputs/sentiment_analysis/visualizations/")

if __name__ == "__main__":
    main()