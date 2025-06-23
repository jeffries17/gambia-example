import os
from island_review_count import IslandBasedAnalyzer
from sentiment_analyzer import SentimentAnalyzer
from island_attraction_analyzer import IslandAttractionAnalyzer

# Use standardized output directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
output_dir = os.path.join(parent_dir, "outputs")

# First run the island analysis to get island classification
island_analyzer = IslandBasedAnalyzer(
    data_dir='data',  # Use local data directory
    output_dir=output_dir
)
island_analyzer.load_data()
island_analyzer.analyze_islands(top_n=10)

# The all_reviews_df now has island classification
enriched_df = island_analyzer.all_reviews_df

# Initialize sentiment analyzer and run sentiment analysis on the enriched DataFrame
sentiment_analyzer = SentimentAnalyzer(output_dir=output_dir)
enriched_df, sentiment_results = sentiment_analyzer.run_sentiment_analysis(enriched_df)

# Save the sentiment-analyzed dataframe to the sentiment analyzer
sentiment_analyzer.reviews_df = enriched_df  

# Create and run the attraction analyzer with island segmentation
attraction_analyzer = IslandAttractionAnalyzer(
    sentiment_analyzer, 
    output_dir=os.path.join(output_dir, 'attraction_analysis')
)
island_results = attraction_analyzer.run_island_analysis(df=enriched_df)

print(f"Island-based attraction analysis complete! Results saved to: {os.path.join(output_dir, 'attraction_analysis')}")