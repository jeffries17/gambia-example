#!/usr/bin/env python3
"""
Accommodation Theme Sentiment Analysis Script

This script analyzes accommodation reviews across different islands in Tonga,
identifying sentiment for different themes (cleanliness, service, location, etc.)
to provide a more detailed analysis than overall sentiment scores.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Import modules with fallback options
try:
    from island_review_count import IslandBasedAnalyzer
except ImportError:
    try:
        from .island_review_count import IslandBasedAnalyzer
    except ImportError:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from tonga_analysis.island_review_count import IslandBasedAnalyzer

# Import SentimentAnalyzer with fallback
try:
    from sentiment_analyzer import SentimentAnalyzer
except ImportError:
    from .sentiment_analyzer import SentimentAnalyzer
except:
    # Try a last resort without the relative import
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tonga_analysis.sentiment_analyzer import SentimentAnalyzer
import json
from matplotlib.colors import LinearSegmentedColormap

# Use standardized output directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
output_dir = os.path.join(parent_dir, "outputs")

def run_accommodation_theme_sentiment_analysis():
    """
    Run the accommodation theme sentiment analysis and generate visualizations.
    """
    print("Starting accommodation theme sentiment analysis...")
    
    # First run the island analysis to get island classification
    island_analyzer = IslandBasedAnalyzer(
        data_dir=os.path.join(parent_dir, 'tonga_data'),  # Use correct data directory path
        output_dir=output_dir
    )
    island_analyzer.load_data()
    island_analyzer.analyze_islands(top_n=10)
    
    # The all_reviews_df now has island classification
    enriched_df = island_analyzer.all_reviews_df
    
    # Initialize sentiment analyzer and run sentiment analysis on the enriched DataFrame
    sentiment_analyzer = SentimentAnalyzer(output_dir=output_dir)
    enriched_df, sentiment_results = sentiment_analyzer.run_sentiment_analysis(enriched_df)
    
    # Filter to accommodation reviews only
    accommodation_df = enriched_df[enriched_df['category'].str.lower() == 'accommodation'].copy()
    print(f"Analyzing {len(accommodation_df)} accommodation reviews")
    
    # Define accommodation themes to analyze
    accommodation_themes = {
        'cleanliness': [
            'clean', 'dirty', 'spotless', 'tidy', 'dust', 'stain', 'hygiene',
            'maintenance', 'housekeeping', 'sanitary', 'smell'
        ],
        'staff_service': [
            'staff', 'service', 'reception', 'front desk', 'friendly', 'helpful', 
            'professional', 'manager', 'concierge', 'attentive', 'welcoming',
            'host', 'hospitality'
        ],
        'location': [
            'location', 'central', 'convenient', 'beach', 'downtown', 'access', 
            'close', 'distance', 'walk', 'near', 'far', 'town', 'accessible', 
            'remote', 'quiet'
        ],
        'value': [
            'value', 'price', 'expensive', 'affordable', 'overpriced', 'cheap', 
            'cost', 'reasonable', 'worth', 'budget'
        ],
        'room_quality': [
            'room', 'bed', 'comfortable', 'spacious', 'small', 'cramped', 'mattress', 
            'pillow', 'air conditioning', 'shower', 'bathroom', 'view', 'balcony'
        ],
        'facilities': [
            'facility', 'pool', 'restaurant', 'bar', 'garden', 'spa', 'gym', 
            'parking', 'reception', 'lounge', 'common area', 'fitness', 'business',
            'wifi', 'internet', 'breakfast', 'amenities'
        ]
    }
    
    # Islands to compare (focusing on the main islands)
    islands = ['Tongatapu', "Vava'u", "Ha'apai", "'Eua"]
    
    # Analyze sentiment for each theme on each island
    theme_sentiment_data = []
    
    for island in islands:
        # Filter to this island
        island_df = accommodation_df[accommodation_df['island_category'] == island].copy()
        
        if len(island_df) < 10:
            print(f"Not enough reviews for {island}")
            continue
            
        # Get overall sentiment for this island
        overall_sentiment = island_df['sentiment_score'].mean()
        theme_sentiment_data.append({
            'Island': island,
            'Theme': 'Overall',
            'Sentiment': overall_sentiment,
            'Reviews': len(island_df),
            'Percentage': 100
        })
        
        # Analyze each theme
        for theme, keywords in accommodation_themes.items():
            pattern = '|'.join(keywords)
            # Find reviews mentioning this theme
            mask = island_df['text'].str.lower().str.contains(pattern, na=False)
            theme_reviews = island_df[mask]
            
            if len(theme_reviews) < 5:
                # Not enough mentions
                continue
            
            # Calculate sentiment for this theme
            theme_sentiment = theme_reviews['sentiment_score'].mean()
            
            # Save data for visualization
            theme_sentiment_data.append({
                'Island': island,
                'Theme': theme.replace('_', ' ').title(),
                'Sentiment': theme_sentiment,
                'Reviews': len(theme_reviews),
                'Percentage': (len(theme_reviews) / len(island_df)) * 100
            })
    
    # Create DataFrame for visualization
    sentiment_df = pd.DataFrame(theme_sentiment_data)
    
    # Create output directory
    vis_dir = os.path.join(output_dir, 'accommodation_analysis', 'island_analysis', 'theme_sentiment')
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    
    # Set visualization style
    sns.set_style('whitegrid')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    
    # Set color palette for islands
    island_colors = {
        'Tongatapu': '#1f77b4',  # Blue
        "Vava'u": '#ff7f0e',     # Orange
        "Ha'apai": '#2ca02c',    # Green
        "'Eua": '#d62728'        # Red
    }
    
    # Save data to JSON for future reference
    output_json = os.path.join(vis_dir, 'accommodation_theme_sentiment.json')
    sentiment_df.to_json(output_json, orient='records')
    print(f"Saved theme sentiment data to {output_json}")
    
    # Create a bar chart of theme sentiment by island
    plt.figure(figsize=(14, 8))
    
    # Find the specific themes that have been mentioned
    themes = sentiment_df[sentiment_df['Theme'] != 'Overall']['Theme'].unique()
    
    # Plot each theme as a group of bars by island
    for theme in themes:
        theme_data = sentiment_df[sentiment_df['Theme'] == theme]
        x_positions = []
        colors = []
        heights = []
        islands_present = []
        
        for i, island in enumerate(islands):
            island_data = theme_data[theme_data['Island'] == island]
            if not island_data.empty:
                x_positions.append(i)
                colors.append(island_colors[island])
                heights.append(island_data['Sentiment'].values[0])
                islands_present.append(island)
        
        plt.bar(
            [theme + ' - ' + island for island in islands_present],
            heights,
            color=colors,
            alpha=0.7
        )
    
    plt.title('Accommodation Theme Sentiment by Island', fontsize=16, pad=20)
    plt.xlabel('Theme and Island', fontsize=14)
    plt.ylabel('Sentiment Score', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add island color legend
    island_patches = [plt.Rectangle((0, 0), 1, 1, color=color) for color in island_colors.values()]
    plt.legend(island_patches, island_colors.keys(), title='Island')
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'accommodation_theme_sentiment_by_island.png'), dpi=300)
    plt.close()
    
    # Create a heatmap of theme sentiment by island
    pivot_df = sentiment_df.pivot(index='Island', columns='Theme', values='Sentiment')
    
    plt.figure(figsize=(14, 8))
    
    # Create a colormap from red to green
    cmap = LinearSegmentedColormap.from_list(
        'sentiment_cmap',
        ['#d7191c', '#fdae61', '#ffffbf', '#a6d96a', '#1a9641'],
        N=100
    )
    
    # Create the heatmap
    sns.heatmap(
        pivot_df,
        annot=True,
        fmt='.2f',
        cmap=cmap,
        linewidths=.5,
        cbar_kws={'label': 'Sentiment Score'}
    )
    
    plt.title('Accommodation Theme Sentiment Heatmap', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'accommodation_theme_sentiment_heatmap.png'), dpi=300)
    plt.close()
    
    # Create a bubble chart showing sentiment and mention percentage
    plt.figure(figsize=(14, 10))
    
    # Exclude the Overall theme for this visualization
    bubble_df = sentiment_df[sentiment_df['Theme'] != 'Overall']
    
    # Plot each island separately
    for i, island in enumerate(islands):
        island_data = bubble_df[bubble_df['Island'] == island]
        if island_data.empty:
            continue
            
        plt.scatter(
            island_data['Theme'],
            island_data['Sentiment'],
            s=island_data['Percentage'] * 20,  # Size based on percentage of reviews
            color=island_colors[island],
            alpha=0.7,
            label=island
        )
        
        # Add labels
        for _, row in island_data.iterrows():
            plt.annotate(
                f"{row['Sentiment']:.2f}",
                (row['Theme'], row['Sentiment']),
                ha='center',
                va='bottom',
                fontsize=9
            )
    
    plt.title('Accommodation Theme Sentiment and Mention Frequency', fontsize=16, pad=20)
    plt.xlabel('Theme', fontsize=14)
    plt.ylabel('Sentiment Score', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(alpha=0.3)
    plt.legend(title='Island')
    
    # Add a legend for bubble size
    sizes = [10, 25, 40]  # Example sizes for the legend
    labels = ['10%', '25%', '40%']  # Corresponding percentages
    
    # Add size legend
    for size, label in zip(sizes, labels):
        plt.scatter([], [], s=size*20, color='gray', alpha=0.7, label=label)
    
    plt.legend(title='Island & Mention %')
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'accommodation_theme_sentiment_bubble.png'), dpi=300)
    plt.close()
    
    # Create radar charts for each island
    # Define theme order for consistent radar charts
    theme_order = ['Cleanliness', 'Staff Service', 'Location', 'Value', 'Room Quality', 'Facilities']
    
    # Only include themes that have data
    available_themes = [t for t in theme_order if t in bubble_df['Theme'].unique()]
    
    # Create a radar chart for each island
    for island in islands:
        island_data = bubble_df[bubble_df['Island'] == island]
        if island_data.empty or len(island_data) < 3:  # Need at least 3 themes for a meaningful radar chart
            continue
            
        # Number of themes
        N = len(available_themes)
        
        # Create angle for each theme
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the polygon
        
        # Set up the plot
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Get sentiment values for available themes, in order
        values = []
        for theme in available_themes:
            theme_data = island_data[island_data['Theme'] == theme]
            if not theme_data.empty:
                values.append(theme_data['Sentiment'].values[0])
            else:
                # If this island doesn't have data for this theme, use a placeholder
                values.append(0)
        
        # Close the polygon
        values += values[:1]
        
        # Draw the radar chart
        ax.plot(angles, values, linewidth=2, linestyle='solid', color=island_colors[island])
        ax.fill(angles, values, color=island_colors[island], alpha=0.25)
        
        # Add labels
        plt.xticks(angles[:-1], available_themes, size=12)
        
        # Add sentiment scale
        ax.set_rlabel_position(0)
        plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], 
                   ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7'], 
                   color='grey', size=10)
        plt.ylim(0, 0.7)
        
        # Add a title
        plt.title(f'Accommodation Theme Sentiment for {island}', size=15, y=1.1)
        
        # Save the radar chart
        safe_island = island.replace("'", "").replace('"', '').lower()
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'accommodation_theme_radar_{safe_island}.png'), dpi=300)
        plt.close()
    
    # Create a combined radar chart for all islands
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))
    
    # Number of themes
    N = len(available_themes)
    
    # Create angle for each theme
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the polygon
    
    # For each island
    for island in islands:
        island_data = bubble_df[bubble_df['Island'] == island]
        if island_data.empty or len(island_data) < 3:
            continue
        
        # Get sentiment values for available themes, in order
        values = []
        for theme in available_themes:
            theme_data = island_data[island_data['Theme'] == theme]
            if not theme_data.empty:
                values.append(theme_data['Sentiment'].values[0])
            else:
                # If this island doesn't have data for this theme, use a placeholder
                values.append(0)
        
        # Close the polygon
        values += values[:1]
        
        # Draw the radar chart
        ax.plot(angles, values, linewidth=2, linestyle='solid', color=island_colors[island], label=island)
        ax.fill(angles, values, color=island_colors[island], alpha=0.1)
    
    # Add labels
    plt.xticks(angles[:-1], available_themes, size=12)
    
    # Add sentiment scale
    ax.set_rlabel_position(0)
    plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], 
               ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7'], 
               color='grey', size=10)
    plt.ylim(0, 0.7)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Add a title
    plt.title('Accommodation Theme Sentiment Comparison Across Islands', size=15, y=1.1)
    
    # Save the radar chart
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'accommodation_theme_radar_comparison.png'), dpi=300)
    plt.close()
    
    print(f"Accommodation theme sentiment analysis completed. Visualizations saved to {vis_dir}")
    return sentiment_df

if __name__ == "__main__":
    run_accommodation_theme_sentiment_analysis()