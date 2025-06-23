import os
import pandas as pd
from traveler_segments import TravelerSegmentAnalyzer
from review_analyzer import TongaReviewAnalyzer

def main():
    """
    Run the traveler segment analysis with specific focus on top 5 countries:
    New Zealand, Australia, United States, United Kingdom, and Tonga.
    """
    # Use new directory structure
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_dir = os.path.join(parent_dir, "outputs", "by_traveler_segment")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # First load the data using the main review analyzer
    print("Loading review data...")
    analyzer = TongaReviewAnalyzer(
        data_dir=os.path.join(parent_dir, 'tonga_data'),
        output_dir=output_dir
    )
    analyzer.load_data()
    
    if analyzer.all_reviews_df is None or len(analyzer.all_reviews_df) == 0:
        print("No review data available for analysis!")
        return
    
    # Initialize traveler segment analyzer with the data
    print("\nInitializing traveler segment analysis...")
    segment_analyzer = TravelerSegmentAnalyzer(
        reviews_df=analyzer.all_reviews_df, 
        output_dir=output_dir
    )
    
    # Preprocess the data
    segment_analyzer.preprocess_data()
    
    # Run the focused top 5 countries analysis
    print("\nRunning analysis on target countries...")
    segment_analyzer.analyze_top5_countries()
    
    print("\nTraveler segment analysis complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Visualizations: {output_dir}/top5_countries/visualizations/")

if __name__ == "__main__":
    main()