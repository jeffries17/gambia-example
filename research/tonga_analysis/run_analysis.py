from review_analyzer import TongaReviewAnalyzer
import os
import argparse

def main():
    """
    Run the Tonga tourism review analysis with the new directory structure.
    """
    # Setup command line arguments
    parser = argparse.ArgumentParser(description='Run Tonga tourism review analysis')
    
    # Use standardized output directory by default
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    default_output_dir = os.path.join(parent_dir, "outputs", "consolidated_reports")
    
    parser.add_argument('--data-dir', default='tonga_data',
                      help='Directory containing Tonga tourism data files (default: tonga_data)')
    parser.add_argument('--output-dir', default=default_output_dir,
                      help=f'Directory to save output files (default: {default_output_dir})')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize analyzer
    analyzer = TongaReviewAnalyzer(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    # Run analysis
    stats = analyzer.run_analysis()
    
    # Print key findings
    if stats:
        print("\nKey Findings:")
        print(f"Total Reviews Analyzed: {stats['review_counts']['total']}")
        print("\nBreakdown by Category:")
        print(f"- Accommodations: {stats['review_counts']['accommodations']}")
        print(f"- Restaurants: {stats['review_counts']['restaurants']}")
        print(f"- Attractions: {stats['review_counts']['attractions']}")
        
        if 'rating_stats' in stats:
            print(f"\nRatings:")
            print(f"Average Rating: {stats['rating_stats']['mean']:.2f}")
            print(f"Median Rating: {stats['rating_stats']['median']:.2f}")
            print(f"Rating Range: {stats['rating_stats']['min']} - {stats['rating_stats']['max']}")
            print(f"Standard Deviation: {stats['rating_stats']['std']:.2f}")
        
        if 'temporal_stats' in stats:
            print(f"\nTemporal Coverage:")
            print(f"Date Range: {stats['temporal_stats']['earliest']} to {stats['temporal_stats']['latest']}")
            print(f"Time Span: {stats['temporal_stats']['date_range_days']} days")
        
        if 'trip_type_stats' in stats:
            print("\nTop Trip Types:")
            sorted_trips = sorted(stats['trip_type_stats'].items(), key=lambda x: x[1], reverse=True)
            for trip_type, count in sorted_trips[:5]:  # Show top 5
                print(f"- {trip_type}: {count}")
        
        if 'location_stats' in stats:
            print("\nTop Visitor Locations:")
            sorted_locations = sorted(stats['location_stats'].items(), key=lambda x: x[1], reverse=True)
            for location, count in sorted_locations[:5]:  # Show top 5
                if location and location.lower() != 'nan':
                    print(f"- {location}: {count}")
        
        print(f"\nAnalysis outputs have been saved to: {args.output_dir}")
        print(f"- Basic statistics: {os.path.join(args.output_dir, 'basic_stats.json')}")
        print(f"- Visualizations: {os.path.join(args.output_dir, 'visualizations')}")

if __name__ == "__main__":
    main()