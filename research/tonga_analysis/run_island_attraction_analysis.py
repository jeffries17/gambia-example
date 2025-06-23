import os
import argparse
from island_review_count import IslandBasedAnalyzer
from attractions_analyzer import AttractionAnalyzer
from sentiment_analyzer import SentimentAnalyzer

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run island-based attraction analysis')
    
    # Use new directory structure
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    default_output_dir = os.path.join(parent_dir, "outputs", "by_sector", "attractions")
    
    parser.add_argument('--data-dir', default='tonga_data',
                      help='Directory containing Tonga tourism data files (default: tonga_data)')
    parser.add_argument('--output-dir', default=default_output_dir,
                      help=f'Directory to save output files (default: {default_output_dir})')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Running island-based attraction analysis...")
    
    # First run the island analysis to get island classification
    island_analyzer = IslandBasedAnalyzer(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    try:
        island_analyzer.load_data()
        island_analyzer.analyze_islands(top_n=10)
        
        # The all_reviews_df now has island classification
        enriched_df = island_analyzer.all_reviews_df
        
        if enriched_df is None or len(enriched_df) == 0:
            print("No data loaded for analysis. Check your data directory.")
            return
            
        # Initialize sentiment analyzer and run sentiment analysis on the enriched DataFrame
        sentiment_analyzer = SentimentAnalyzer(output_dir=args.output_dir)
        enriched_df, sentiment_results = sentiment_analyzer.run_sentiment_analysis(enriched_df)
        
        # Save the sentiment-analyzed dataframe to the sentiment analyzer
        sentiment_analyzer.reviews_df = enriched_df  
        
        # Create and run the attraction analyzer with island segmentation
        attraction_analyzer = AttractionAnalyzer(
            sentiment_analyzer, 
            output_dir=args.output_dir
        )
        island_results = attraction_analyzer.run_island_analysis(df=enriched_df)
        
        print(f"Island-based attraction analysis complete! Results saved to: {args.output_dir}")
    
    except Exception as e:
        print(f"Error during attraction analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()