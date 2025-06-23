import sys
import os

# Add parent directory to the path to allow imports from the main project
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tonga_analysis.visitor_expectations import VisitorExpectations
from tonga_analysis.review_analyzer import TongaReviewAnalyzer

def main():
    data_dir = 'data'
    output_dir = 'outputs'
    review_analyzer = TongaReviewAnalyzer(data_dir=data_dir, output_dir=output_dir)
    review_analyzer.load_data()

    if review_analyzer.all_reviews_df is None or review_analyzer.all_reviews_df.empty:
        print("No review data available for analysis.")
        return

    visitor_expectations = VisitorExpectations(review_analyzer.all_reviews_df)
    visitor_expectations.identify_expectation_keywords()
    expectation_stats = visitor_expectations.analyze_expectation_discrepancies()
    print("Expectation Analysis Stats:")
    print(expectation_stats)
    
    visitor_expectations.visualize_expectation_discrepancies()

if __name__ == "__main__":
    main()
