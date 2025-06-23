#!/usr/bin/env python3
"""
Run script for Island Sentiment Comparison Analysis

This script runs the island sentiment comparison analysis to analyze
sentiment scores across Tonga's islands, broken down by category
(accommodations, attractions, restaurants).
"""

from island_sentiment_comparison import IslandSentimentComparison

def main():
    """
    Main function to run the island sentiment comparison analysis.
    """
    print("=== Tonga Island Sentiment Comparison Analysis ===")
    print("Analyzing sentiment scores across island groups by category...")
    
    # Create the analyzer with correct data directory path
    analyzer = IslandSentimentComparison(data_dir='../tonga_data')
    
    # Run the analysis
    results = analyzer.run_analysis()
    
    if results:
        print("\nAnalysis completed successfully!")
        print("Visualizations generated:")
        print("1. Bar chart of sentiment by island and category")
        print("2. Heatmap of sentiment scores")
        print("3. Combined chart with sentiment scores and review counts")
        print("4. Radar chart showing sentiment patterns across islands")
    else:
        print("\nAnalysis failed.")

if __name__ == "__main__":
    main()