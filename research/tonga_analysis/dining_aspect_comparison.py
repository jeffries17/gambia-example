import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from island_review_count import IslandBasedAnalyzer
from sentiment_analyzer import SentimentAnalyzer
import re

# Use standardized output directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
output_dir = os.path.join(parent_dir, "outputs")

# First run the island analysis to get island classification
island_analyzer = IslandBasedAnalyzer(
    data_dir='data',  # Use local data directory
    output_dir=output_dir
)
island_analyzer.load_data()
island_analyzer.analyze_islands(top_n=10)

# The all_reviews_df now has island classification
enriched_df = island_analyzer.all_reviews_df

# Initialize sentiment analyzer and run sentiment analysis on the enriched DataFrame
sentiment_analyzer = SentimentAnalyzer(output_dir=output_dir)
enriched_df, sentiment_results = sentiment_analyzer.run_sentiment_analysis(enriched_df)

# Filter to restaurant reviews only
restaurant_df = enriched_df[enriched_df['category'].str.lower() == 'restaurant'].copy()
print(f"Analyzing {len(restaurant_df)} restaurant reviews")

# Define dining aspects to analyze
dining_aspects = {
    'food_quality': ['tasty', 'delicious', 'fresh', 'quality', 'flavor', 'flavour', 'taste', 'cooked', 'cuisine'],
    'service': ['service', 'staff', 'friendly', 'attentive', 'helpful', 'quick', 'slow', 'wait', 'waiter', 'waitress'],
    'value': ['price', 'value', 'expensive', 'cheap', 'worth', 'affordable', 'overpriced', 'reasonable', 'cost'],
    'atmosphere': ['atmosphere', 'ambiance', 'ambience', 'decor', 'view', 'setting', 'music', 'romantic', 'quiet', 'loud'],
    'cleanliness': ['clean', 'spotless', 'hygiene', 'dirty', 'mess', 'tidy', 'neat'],
    'menu_variety': ['menu', 'options', 'variety', 'choice', 'selection', 'diverse', 'limited', 'specials']
}

# Islands to compare
islands = ['Tongatapu', "Vava'u", "Ha'apai"]

# Function to analyze aspects
def analyze_dining_aspects(df, island):
    """Analyze dining aspects for a specific island"""
    # Filter to this island
    island_df = df[df['island_category'] == island].copy()
    
    if len(island_df) < 10:
        print(f"Not enough reviews for {island}")
        return None
    
    # Analyze each aspect
    aspect_data = {}
    
    for aspect, keywords in dining_aspects.items():
        pattern = '|'.join(keywords)
        # Find reviews mentioning this aspect
        mask = island_df['text'].str.lower().str.contains(pattern, na=False)
        aspect_reviews = island_df[mask]
        
        if len(aspect_reviews) < 5:
            # Not enough mentions
            aspect_data[aspect] = {
                'mentions': 0,
                'sentiment': np.nan,
                'percentage': 0
            }
            continue
            
        # Calculate statistics
        aspect_data[aspect] = {
            'mentions': len(aspect_reviews),
            'sentiment': aspect_reviews['sentiment_score'].mean(),
            'percentage': (len(aspect_reviews) / len(island_df)) * 100
        }
    
    return aspect_data

# Analyze aspects for each island
island_aspect_data = {}
for island in islands:
    island_aspect_data[island] = analyze_dining_aspects(restaurant_df, island)
    
# Create output directory
comparison_dir = os.path.join(output_dir, 'restaurant_analysis', 'island_analysis', 'comparisons')
if not os.path.exists(comparison_dir):
    os.makedirs(comparison_dir)

# Create DataFrame for visualization
aspect_rows = []
for island, aspects in island_aspect_data.items():
    if aspects is None:
        continue
        
    for aspect, data in aspects.items():
        aspect_rows.append({
            'Island': island,
            'Aspect': aspect.replace('_', ' ').title(),
            'Mentions': data['mentions'],
            'Sentiment': data['sentiment'] if not np.isnan(data['sentiment']) else 0,
            'Percentage': data['percentage']
        })

aspect_df = pd.DataFrame(aspect_rows)

# Visualize mentions by aspect across islands
plt.figure(figsize=(14, 8))
sns.set_style('whitegrid')

# Set color palette for islands
island_colors = {
    'Tongatapu': '#FFB703',  # Yellow/Gold
    "Vava'u": '#219EBC',    # Blue
    "Ha'apai": '#8ECAE6',   # Light Blue
    "'Eua": '#90BE6D'      # Green
}

# Create bar chart
ax = sns.barplot(
    x='Aspect', 
    y='Mentions',
    hue='Island',
    data=aspect_df,
    palette=[island_colors.get(island, '#AAAAAA') for island in aspect_df['Island'].unique()]
)

# Customize plot
plt.title('Dining Experience Aspects Mentioned by Island', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Dining Aspect', fontsize=14)
plt.ylabel('Number of Mentions', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Island', frameon=True)
plt.grid(axis='y', alpha=0.3)

# Add value labels
for container in ax.containers:
    ax.bar_label(container, fmt='%d', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(comparison_dir, 'dining_aspects_by_island.png'), dpi=300, bbox_inches='tight')
plt.close()

# Visualize sentiment by aspect across islands
plt.figure(figsize=(14, 8))

# Create bar chart for sentiment
sentiment_df = aspect_df[aspect_df['Mentions'] > 0].copy()  # Filter out zero mentions
ax = sns.barplot(
    x='Aspect', 
    y='Sentiment',
    hue='Island',
    data=sentiment_df,
    palette=[island_colors.get(island, '#AAAAAA') for island in sentiment_df['Island'].unique()]
)

# Customize plot
plt.title('Sentiment Towards Dining Experience Aspects by Island', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Dining Aspect', fontsize=14)
plt.ylabel('Average Sentiment Score', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Island', frameon=True)
plt.grid(axis='y', alpha=0.3)

# Add value labels
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(comparison_dir, 'dining_aspects_sentiment_by_island.png'), dpi=300, bbox_inches='tight')
plt.close()

# Create a heatmap of sentiment by aspect and island
pivot_df = aspect_df.pivot(index='Aspect', columns='Island', values='Sentiment')

plt.figure(figsize=(12, 8))
sns.heatmap(
    pivot_df,
    annot=True,
    fmt='.2f',
    cmap='RdYlGn',
    center=0,
    linewidths=0.5,
    cbar_kws={'label': 'Sentiment Score'}
)
plt.title('Dining Aspect Sentiment Heatmap by Island', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(comparison_dir, 'dining_aspects_sentiment_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()

print("Dining aspect comparison visualizations generated!")
print(f"Visualizations saved to {comparison_dir}")