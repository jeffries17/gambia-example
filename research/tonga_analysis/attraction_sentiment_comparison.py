import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from island_review_count import IslandBasedAnalyzer
from sentiment_analyzer import SentimentAnalyzer

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

# Filter to attraction reviews only
attraction_df = enriched_df[enriched_df['category'].str.lower() == 'attraction'].copy()
print(f"Analyzing {len(attraction_df)} attraction reviews")

# Define attraction aspects to analyze
attraction_aspects = {
    'guide_quality': ['guide', 'tour guide', 'leader', 'instructor', 'captain', 'staff'],
    'safety': ['safe', 'safety', 'equipment', 'secure', 'comfort', 'comfortable'],
    'value': ['price', 'value', 'worth', 'money', 'expensive', 'cheap', 'cost'],
    'scenery': ['beautiful', 'view', 'scenery', 'landscape', 'picture', 'photo'],
    'learning': ['learn', 'educational', 'interesting', 'information', 'history', 'knowledge'],
    'organization': ['organized', 'professional', 'time', 'schedule', 'punctual'],
    'activities': ['activity', 'swimming', 'snorkeling', 'diving', 'hiking', 'tour']
}

# Islands to compare (using only those with enough reviews)
islands = ['Tongatapu', "Vava'u", "Ha'apai"]

# Analyze sentiment for each aspect on each island
aspect_sentiment_data = []

for island in islands:
    # Filter to this island
    island_df = attraction_df[attraction_df['island_category'] == island].copy()
    
    if len(island_df) < 10:
        print(f"Not enough reviews for {island}")
        continue
        
    # Get overall sentiment for this island
    overall_sentiment = island_df['sentiment_score'].mean()
    aspect_sentiment_data.append({
        'Island': island,
        'Aspect': 'Overall',
        'Sentiment': overall_sentiment,
        'Reviews': len(island_df)
    })
    
    # Analyze each aspect
    for aspect, keywords in attraction_aspects.items():
        pattern = '|'.join(keywords)
        # Find reviews mentioning this aspect
        mask = island_df['text'].str.lower().str.contains(pattern, na=False)
        aspect_reviews = island_df[mask]
        
        if len(aspect_reviews) < 5:
            # Not enough mentions
            continue
        
        # Calculate sentiment for this aspect
        aspect_sentiment = aspect_reviews['sentiment_score'].mean()
        
        # Save data for visualization
        aspect_sentiment_data.append({
            'Island': island,
            'Aspect': aspect.replace('_', ' ').title(),
            'Sentiment': aspect_sentiment,
            'Reviews': len(aspect_reviews)
        })

# Create DataFrame for visualization
sentiment_df = pd.DataFrame(aspect_sentiment_data)

# Create output directory
vis_dir = os.path.join(output_dir, 'attraction_analysis', 'island_analysis', 'visualizations')
if not os.path.exists(vis_dir):
    os.makedirs(vis_dir)

# Set visualization style
sns.set_style('whitegrid')

# Set color palette for islands
island_colors = {
    'Tongatapu': '#FFB703',  # Yellow/Gold
    "Vava'u": '#219EBC',    # Blue
    "Ha'apai": '#8ECAE6',   # Light Blue
    "'Eua": '#90BE6D'      # Green
}

# 1. Bar chart comparison of aspect sentiment by island
plt.figure(figsize=(14, 8))
ax = sns.barplot(
    x='Aspect', 
    y='Sentiment',
    hue='Island',
    data=sentiment_df,
    palette=[island_colors.get(island, '#AAAAAA') for island in sentiment_df['Island'].unique()]
)

# Add horizontal line at zero (neutral sentiment)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)

# Customize plot
plt.title('Sentiment Towards Attraction Aspects by Island', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Aspect', fontsize=14)
plt.ylabel('Average Sentiment Score', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Island', frameon=True)
plt.grid(axis='y', alpha=0.3)

# Add value labels
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(vis_dir, 'attraction_aspect_sentiment_by_island.png'), dpi=300)
plt.close()

# 2. Heatmap of aspect sentiment by island
# Pivot the data
pivot_df = sentiment_df.pivot(index='Aspect', columns='Island', values='Sentiment')

plt.figure(figsize=(12, 8))
sns.heatmap(
    pivot_df,
    annot=True,
    fmt='.2f',
    cmap='RdYlGn',
    center=0,
    vmin=-0.5,
    vmax=0.5,
    linewidths=0.5,
    cbar_kws={'label': 'Sentiment Score'}
)
plt.title('Attraction Aspect Sentiment Heatmap by Island', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(vis_dir, 'attraction_aspect_sentiment_heatmap.png'), dpi=300)
plt.close()

# 3. Bubble chart showing sentiment and review count
plt.figure(figsize=(14, 8))

# Create a bubble chart where:
# - X-axis is Island
# - Y-axis is Sentiment
# - Bubble size is review count
# - Color is Aspect
aspects = sentiment_df['Aspect'].unique()
colors = sns.color_palette('tab10', len(aspects))
aspect_color_map = {aspect: color for aspect, color in zip(aspects, colors)}

for aspect in aspects:
    aspect_data = sentiment_df[sentiment_df['Aspect'] == aspect]
    plt.scatter(
        aspect_data['Island'], 
        aspect_data['Sentiment'],
        s=aspect_data['Reviews'] * 2,  # Scale size for visibility
        color=aspect_color_map[aspect],
        alpha=0.7,
        label=aspect
    )

# Customize plot
plt.title('Attraction Sentiment Analysis by Island (Bubble Size = Number of Reviews)', 
         fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Island', fontsize=14)
plt.ylabel('Sentiment Score', fontsize=14)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
plt.grid(True, alpha=0.3)
plt.legend(title='Aspect', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(vis_dir, 'attraction_sentiment_bubble.png'), dpi=300)
plt.close()

print("Attraction sentiment comparison visualizations generated!")
print(f"Visualizations saved to {vis_dir}")