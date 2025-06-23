import os
import argparse
from island_review_count import IslandBasedAnalyzer
from accommodation_analyzer import AccommodationAnalyzer

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run island-based accommodation analysis')
    
    # Use new directory structure
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    default_output_dir = os.path.join(parent_dir, "outputs", "by_sector", "accommodations")
    
    parser.add_argument('--data-dir', default='tonga_data',
                      help='Directory containing Tonga tourism data files (default: tonga_data)')
    parser.add_argument('--output-dir', default=default_output_dir,
                      help=f'Directory to save output files (default: {default_output_dir})')
    parser.add_argument('--top-islands', type=int, default=10,
                       help='Number of top islands to analyze (default: 10)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Running island-based accommodation analysis...")
    
    # First run the island analysis
    island_analyzer = IslandBasedAnalyzer(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    island_analyzer.load_data()
    island_analyzer.analyze_islands(top_n=args.top_islands)
    
    # The all_reviews_df now has island classification
    enriched_df = island_analyzer.all_reviews_df
    
    # Create and run the accommodation analyzer with island segmentation
    accommodation_analyzer = AccommodationAnalyzer(
        output_dir=args.output_dir
    )
    island_results = accommodation_analyzer.run_island_analysis(df=enriched_df)
    
    print(f"Analysis complete. Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()