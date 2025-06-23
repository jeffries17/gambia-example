import os
import json
import argparse
import pandas as pd  # Import pandas to handle DataFrame operations
from datetime import datetime
from review_analysis import TongaTourismAnalysis

# Assuming the below imports point to the actual locations of these classes
from extensions.restaurant_analysis import RestaurantAnalysis
from extensions.attraction_analysis import AttractionAnalysis
from extensions.accommodation_analysis import AccommodationAnalysis
from insights.seasonal_patterns import SeasonalPatternAnalyzer
from insights.traveler_segments import TravelerSegmentAnalyzer
from insights.price_value_perception import PriceValueAnalyzer
from insights.visitor_expectations import ExpectationAnalyzer
from insights.competitive_intelligence import CompetitiveAnalyzer

def setup_argparse():
    """Set up command line argument parsing."""
    parser = argparse.ArgumentParser(description='Run tourism review analysis on TripAdvisor data.')
    parser.add_argument('--analyzer', type=str, choices=['restaurant', 'attraction', 'accommodation', 'all'],
                        help='Specify which analyzer to run (restaurant, attraction, accommodation, or all)')
    parser.add_argument('--insights', action='store_true',
                        help='Include additional insight analysis (seasonal, traveler, etc.)')
    parser.add_argument('--data-dir', type=str, default='tonga_data',
                        help='Directory containing JSON review data (default: tonga_data)')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory for analysis outputs (default: results)')
    return parser.parse_args()

def create_analyzers(base_analyzer, output_dir):
    """Create and return specific analyzers based on the base analysis class."""
    return {
        'restaurant': RestaurantAnalysis(base_analyzer, output_dir=os.path.join(output_dir, 'restaurant')),
        'attraction': AttractionAnalysis(base_analyzer, output_dir=os.path.join(output_dir, 'attraction')),
        'accommodation': AccommodationAnalysis(base_analyzer, output_dir=os.path.join(output_dir, 'accommodation'))
    }

def create_insight_analyzers(output_dir):
    """Create and return insight analyzers."""
    return {
        'seasonal': SeasonalPatternAnalyzer(output_dir=os.path.join(output_dir, 'seasonal')),
        'traveler': TravelerSegmentAnalyzer(output_dir=os.path.join(output_dir, 'traveler')),
        'price_value': PriceValueAnalyzer(output_dir=os.path.join(output_dir, 'price_value')),
        'expectations': ExpectationAnalyzer(output_dir=os.path.join(output_dir, 'expectations')),
        'competitive': CompetitiveAnalyzer(output_dir=os.path.join(output_dir, 'competitive'))
    }

def main():
    args = setup_argparse()

    # Initialize the base analyzer with the specified output directory
    base_analyzer = TongaTourismAnalysis(output_dir=args.output_dir)

    # Create analyzers based on user input
    analyzers = create_analyzers(base_analyzer, args.output_dir)
    insights = create_insight_analyzers(args.output_dir) if args.insights else {}

    # Load and preprocess files based on the type of analysis requested
    json_files = [os.path.join(args.data_dir, file) for file in os.listdir(args.data_dir) if file.endswith('.json')]
    if not json_files:
        print(f"No JSON files found in {args.data_dir}. Please check your data directory and try again.")
        return

    for json_file in json_files:
        df = pd.read_json(json_file)
        df['processed_text'] = df['text'].apply(base_analyzer.preprocess_text)  # Preprocess using method in base analyzer

        if args.analyzer in ['restaurant', 'attraction', 'accommodation']:
            specific_analyzer = analyzers[args.analyzer]
            print(f"Running {args.analyzer} analysis on {json_file}")
            specific_analyzer.run_analysis(df)
        elif args.analyzer == 'all':
            for analyzer_name, analyzer in analyzers.items():
                print(f"Running {analyzer_name} analysis on {json_file}")
                analyzer.run_analysis(df)

        if args.insights:
            for insight_name, insight in insights.items():
                print(f"Running {insight_name} insights on {json_file}")
                insight.run_analysis(df)

    print("Analysis complete. Results are stored in the specified output directory.")

if __name__ == "__main__":
    main()