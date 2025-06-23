#!/usr/bin/env python3
"""
Script to run regional comparison analysis on tourism review data.
Compares Tonga with regional competitors (Fiji, Samoa, Tahiti).
"""

import os
import sys
import argparse
from regional_comparison import RegionalComparisonAnalyzer

def main():
    """Run the regional comparison analysis."""
    parser = argparse.ArgumentParser(description='Run regional tourism data comparison analysis.')
    parser.add_argument('--countries', nargs='+', default=['tonga', 'fiji', 'samoa', 'tahiti'],
                      help='List of countries to include in the comparison (default: tonga, fiji, samoa, tahiti)')
    parser.add_argument('--data-dir', default='regional_data',
                      help='Base directory containing country data folders (default: regional_data)')
    parser.add_argument('--use-tonga-data', action='store_true',
                      help='Copy Tonga data from the main data directory instead of looking in regional_data/tonga')
    # Use new directory structure
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    default_output_dir = os.path.join(parent_dir, "outputs", "regional_comparison")
    parser.add_argument('--output-dir', default=default_output_dir,
                      help=f'Directory to save output files (default: {default_output_dir})')
    parser.add_argument('--sectors', nargs='+', default=['all'],
                       help='Sectors to analyze: accommodations, attractions, restaurants, all (default: all)')
    
    args = parser.parse_args()
    
    # Validate countries
    if not args.countries:
        print("Error: No countries specified")
        sys.exit(1)
    
    # Create data directory if it doesn't exist
    if not os.path.exists(args.data_dir):
        print(f"Creating data directory '{args.data_dir}'")
        os.makedirs(args.data_dir, exist_ok=True)
        
    # Create country subdirectories if needed
    for country in args.countries:
        country_dir = os.path.join(args.data_dir, country)
        if not os.path.exists(country_dir):
            print(f"Creating subdirectory for {country}")
            os.makedirs(country_dir, exist_ok=True)
            
    # Copy Tonga data from main data directory if requested
    if args.use_tonga_data and 'tonga' in args.countries:
        import shutil
        tonga_dir = os.path.join(args.data_dir, 'tonga')
        main_data_dir = 'tonga_data'  # Changed from 'data' to 'tonga_data'
        
        # Create the tonga directory if it doesn't exist
        if not os.path.exists(tonga_dir):
            os.makedirs(tonga_dir, exist_ok=True)
            print(f"Created {tonga_dir} directory")
        
        for file_name in ['tonga_accommodations.json', 'tonga_attractions.json', 'tonga_restaurants.json']:
            src_path = os.path.join(main_data_dir, file_name)
            dst_path = os.path.join(tonga_dir, file_name)
            
            if os.path.exists(src_path):
                print(f"Copying {file_name} from main data directory to regional directory")
                shutil.copy2(src_path, dst_path)
            else:
                print(f"Warning: Could not find {file_name} in {main_data_dir}")
        
        print("Tonga data copied successfully")
    
    # Create analyzer
    analyzer = RegionalComparisonAnalyzer(
        countries=args.countries,
        base_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    # Load data
    print("\nLoading country data:")
    for country in args.countries:
        print(f"- Looking for {country} data in {args.data_dir}/{country}/")
    
    analyzer.load_country_data()
    
    # Print what data was loaded
    print("\nData loaded:")
    for country in args.countries:
        accommodations = "Loaded" if analyzer.data[country]["accommodations"] is not None else "Not loaded"
        attractions = "Loaded" if analyzer.data[country]["attractions"] is not None else "Not loaded"
        restaurants = "Loaded" if analyzer.data[country]["restaurants"] is not None else "Not loaded"
        print(f"- {country}: Accommodations: {accommodations}, Attractions: {attractions}, Restaurants: {restaurants}")
    
    # Apply sentiment analysis
    analyzer.apply_sentiment_analysis()
    
    # Run sector-specific analyses based on user input
    sectors_to_analyze = []
    if 'all' in args.sectors:
        sectors_to_analyze = ['accommodations', 'attractions', 'restaurants']
    else:
        valid_sectors = {'accommodations', 'attractions', 'restaurants'}
        sectors_to_analyze = [s for s in args.sectors if s in valid_sectors]
        if not sectors_to_analyze:
            print("Error: No valid sectors specified")
            sys.exit(1)
    
    # Run analyses
    if 'accommodations' in sectors_to_analyze:
        print("Analyzing accommodations...")
        analyzer.analyze_accommodations()
    
    if 'attractions' in sectors_to_analyze:
        print("Analyzing attractions...")
        analyzer.analyze_attractions()
    
    if 'restaurants' in sectors_to_analyze:
        print("Analyzing restaurants...")
        analyzer.analyze_restaurants()
    
    # Create cross-sector comparisons
    print("Creating cross-sector comparisons...")
    analyzer.create_cross_sector_comparisons()
    
    # Generate competitive insights
    print("Generating competitive insights...")
    insights = analyzer.generate_competitive_insights()
    
    print("\nAnalysis complete!")
    print(f"Results saved to: {args.output_dir}")
    
    # Print key insights about Tonga's position
    if 'overall' in insights and 'rating_comparison' in insights['overall']:
        rating_comp = insights['overall']['rating_comparison']
        print("\nOverall Position:")
        print(f"- Tonga's average rating: {rating_comp['tonga_avg_rating']:.2f}")
        print(f"- Tonga ranks #{rating_comp['tonga_position']} out of {rating_comp['total_countries']} countries")
        if rating_comp['better_than_tonga']:
            print(f"- Countries rating higher than Tonga: {', '.join(rating_comp['better_than_tonga'])}")
    
    if 'overall' in insights and 'tonga_sector_comparison' in insights['overall']:
        sector_comp = insights['overall']['tonga_sector_comparison']
        print("\nTonga's Strengths and Weaknesses:")
        print(f"- Strongest sector: {sector_comp['strongest_sector']} (rating: {sector_comp['strongest_sector_rating']:.2f})")
        print(f"- Weakest sector: {sector_comp['weakest_sector']} (rating: {sector_comp['weakest_sector_rating']:.2f})")

if __name__ == "__main__":
    main()