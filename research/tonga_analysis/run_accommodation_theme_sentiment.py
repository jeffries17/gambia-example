#!/usr/bin/env python3
"""
Run script for Accommodation Theme Sentiment Analysis

This script runs the accommodation theme sentiment analysis, which analyzes
sentiment scores for specific themes in accommodation reviews across
different islands in Tonga.
"""

from accommodation_theme_sentiment import run_accommodation_theme_sentiment_analysis

def main():
    """
    Main function to run the accommodation theme sentiment analysis.
    """
    print("=== Tonga Accommodation Theme Sentiment Analysis ===")
    print("Analyzing accommodation sentiment by theme and island...")
    
    # Run the analysis
    sentiment_df = run_accommodation_theme_sentiment_analysis()
    
    if sentiment_df is not None and not sentiment_df.empty:
        print("\nAnalysis completed successfully!")
        print("Generated visualizations:")
        print("1. Bar chart of theme sentiment by island")
        print("2. Theme sentiment heatmap")
        print("3. Bubble chart showing sentiment and mention frequency")
        print("4. Individual radar charts for each island")
        print("5. Combined radar chart for all islands")
        
        print("\nThese visualizations provide a more detailed view of sentiment")
        print("by focusing on specific themes like cleanliness, service, etc.")
        print("This approach gives stakeholders a much more actionable understanding")
        print("than overall average sentiment scores.")
    else:
        print("\nAnalysis failed or produced no results.")

if __name__ == "__main__":
    main()