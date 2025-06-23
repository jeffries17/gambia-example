#!/usr/bin/env python3
"""
Run script for the consolidated island analyzer.
This script replaces both:
- run_island_accommodation_analysis.py
- run_island_property_analysis.py

It performs a comprehensive analysis of tourism data across different
islands in Tonga, including both review-based and property-based analyses.
"""

import os
import argparse
from island_analyzer import IslandAnalyzer

def main():
    """Run the comprehensive island analysis."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run comprehensive island analysis for Tonga tourism data')
    
    # Use new directory structure for outputs
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    default_output_dir = os.path.join(parent_dir, "outputs", "by_island")
    
    parser.add_argument('--data-dir', default='tonga_data',
                      help='Directory containing Tonga tourism data files (default: tonga_data)')
    parser.add_argument('--output-dir', default=default_output_dir,
                      help=f'Directory to save output files (default: {default_output_dir})')
    parser.add_argument('--top-islands', type=int, default=5,
                       help='Number of top islands to analyze (default: 5)')
    
    args = parser.parse_args()
    
    # Create and run the analyzer
    print(f"Running island analysis with data from {args.data_dir}")
    analyzer = IslandAnalyzer(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    # Run the analysis
    island_stats, property_stats, overall_stats = analyzer.run_analysis(top_n=args.top_islands)
    
    # Print summary of results
    if island_stats and overall_stats:
        # Print overall statistics
        print("\n--- ANALYSIS SUMMARY ---")
        print(f"Total reviews analyzed: {overall_stats['total_reviews']}")
        if 'total_properties' in overall_stats:
            print(f"Total unique properties: {overall_stats['total_properties']}")
        
        # Print top islands by review count
        print("\nTop islands by review count:")
        island_review_counts = {k: v for k, v in overall_stats['reviews_by_island'].items()}
        sorted_islands = sorted(island_review_counts.items(), key=lambda x: x[1], reverse=True)
        for i, (island, count) in enumerate(sorted_islands[:args.top_islands], 1):
            print(f"{i}. {island}: {count} reviews")
            
            # Print breakdown by category
            if island in island_stats:
                stats = island_stats[island]
                for category in ['accommodation', 'restaurant', 'attraction']:
                    if category in stats.get('category_counts', {}):
                        count = stats['category_counts'][category]
                        avg_rating = stats.get('avg_ratings_by_category', {}).get(category, 0)
                        print(f"   - {category.title()}: {count} reviews, avg rating: {avg_rating:.2f}")
        
        print(f"\nResults saved to: {args.output_dir}")
        print(f"Visualizations available at: {os.path.join(args.output_dir, 'visualizations')}")
    else:
        print("\nAnalysis did not complete successfully.")

if __name__ == "__main__":
    main()