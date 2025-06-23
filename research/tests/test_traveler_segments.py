import sys
import os

# Add parent directory to the path to allow imports from the main project
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tonga_analysis.traveler_segments import TravelerSegmentAnalyzer
from tonga_analysis.review_analyzer import TongaReviewAnalyzer

def main():
    data_dir = 'data'
    output_dir = 'outputs'
    review_analyzer = TongaReviewAnalyzer(data_dir=data_dir, output_dir=output_dir)
    review_analyzer.load_data()

    if review_analyzer.all_reviews_df is None or review_analyzer.all_reviews_df.empty:
        print("No review data available for analysis.")
        return

    # Segment analysis and visualization
    traveler_segments = TravelerSegmentAnalyzer(review_analyzer.all_reviews_df)
    traveler_segments.preprocess_data()
    trip_type_stats, country_stats = traveler_segments.analyze_by_segment()

    print("Traveler Segment Analysis Stats by Trip Type:")
    print(trip_type_stats)
    print("Traveler Segment Analysis Stats by Country:")
    print(country_stats)

    traveler_segments.visualize_by_segment(trip_type_stats, 'trip_type')
    traveler_segments.visualize_by_segment(country_stats, 'country')

if __name__ == "__main__":
    main()
