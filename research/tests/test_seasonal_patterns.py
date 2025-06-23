import sys
import os

# Add parent directory to the path to allow imports from the main project
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tonga_analysis.review_analyzer import TongaReviewAnalyzer
from tonga_analysis.seasonal_patterns import SeasonalPatterns

def main():
    """
    Test the seasonal analysis of Tonga tourism reviews.
    """
    # Initialize the review analyzer
    data_dir = 'data'  # Ensure this directory has your JSON data files
    output_dir = 'outputs'
    review_analyzer = TongaReviewAnalyzer(data_dir=data_dir, output_dir=output_dir)

    # Load data
    review_analyzer.load_data()
    
    # Check if data is loaded
    if review_analyzer.all_reviews_df is None or review_analyzer.all_reviews_df.empty:
        print("No review data available for analysis.")
        return

    # Initialize seasonal analysis with the loaded reviews DataFrame
    seasonal_analysis = SeasonalPatterns(review_analyzer.all_reviews_df)

    # Assign seasons to reviews
    seasonal_analysis.assign_seasons()

    # Calculate and print seasonal statistics
    seasonal_analysis.calculate_seasonal_stats()

    # Generate and display visualizations for seasonal trends
    seasonal_analysis.visualize_seasonal_trends()

    # Generate and display word clouds for seasonal keywords
    seasonal_analysis.extract_seasonal_keywords()

if __name__ == "__main__":
    main()
