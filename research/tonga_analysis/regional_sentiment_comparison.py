import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set up paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = os.path.join(base_dir, "outputs", "regional_comparison")
sentiment_path = os.path.join(output_dir, "theme_sentiment", "regional_theme_comparison.json")
viz_dir = os.path.join(output_dir, "overall", "visualizations")
os.makedirs(viz_dir, exist_ok=True)

# Load sentiment data
if not os.path.exists(sentiment_path):
    print(f"Error: Sentiment data file not found at {sentiment_path}")
    exit(1)

with open(sentiment_path, 'r') as f:
    sentiment_data = json.load(f)

# Initialize data structure to hold calculated sentiment scores
categories = ["accommodation", "attraction", "restaurant"]
countries = ["tonga", "fiji", "samoa", "tahiti"]

# Create a DataFrame to store sentiment scores
results = []

# Extract and average sentiment scores for each category and country
for category in categories:
    category_key = category
    if category_key == "attraction":
        category_key = "attractions"  # Handle different naming in the data
    if category_key == "restaurant":
        category_key = "restaurants"  # Handle different naming in the data
    
    if category in sentiment_data.get("by_category", {}):
        themes = sentiment_data["by_category"][category]["by_theme"]
        
        for country in countries:
            country_sentiments = []
            country_weights = []
            
            for theme, theme_data in themes.items():
                if country in theme_data.get("by_country", {}):
                    country_data = theme_data["by_country"][country]
                    sentiment = country_data.get("avg_sentiment", 0)
                    mentions = country_data.get("mention_count", 0)
                    
                    # Add to weighted average calculation
                    if mentions > 0:
                        country_sentiments.append(sentiment)
                        country_weights.append(mentions)
            
            # Calculate weighted average sentiment for this country and category
            if sum(country_weights) > 0:
                avg_sentiment = sum(s * w for s, w in zip(country_sentiments, country_weights)) / sum(country_weights)
            else:
                avg_sentiment = 0
                
            # Add to results
            results.append({
                "country": country,
                "category": category,
                "sentiment": avg_sentiment
            })

# Convert to DataFrame
df = pd.DataFrame(results)

# Calculate overall sentiment for each country
overall_sentiment = df.groupby('country')['sentiment'].mean().reset_index()
overall_sentiment['category'] = 'overall'

# Combine with category data
df_combined = pd.concat([df, overall_sentiment], ignore_index=True)

# Map category names to more readable versions
category_mapping = {
    'overall': 'Overall',
    'accommodation': 'Accommodations',
    'attraction': 'Attractions',
    'restaurant': 'Restaurants'
}
df_combined['display_category'] = df_combined['category'].map(category_mapping)

# Define the order of categories for x-axis
category_order = ['Overall', 'Accommodations', 'Attractions', 'Restaurants']

# Define a custom color palette for countries
country_colors = {
    'tonga': '#E57373',   # Red
    'fiji': '#64B5F6',    # Blue
    'samoa': '#81C784',   # Green
    'tahiti': '#FFD54F'   # Yellow
}

# Set up plot with improved styling
plt.figure(figsize=(14, 8))

# Create grouped bar chart
ax = sns.barplot(x='display_category', y='sentiment', hue='country', data=df_combined, 
                order=category_order, palette=country_colors)

# Customize plot appearance
plt.title('Comparison of Average Sentiment Scores: Tonga vs Other Pacific Islands', fontsize=22, fontweight='bold')
plt.xlabel('Category', fontsize=20)
plt.ylabel('Average Sentiment Score', fontsize=20)
plt.ylim(0, 0.5)  # Sentiment scores typically range from -1 to 1, but our data seems to be 0 to ~0.35
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on top of each bar
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', padding=3, fontsize=14, fontweight='bold')

# Customize legend
plt.legend(title='Country', fontsize=14, title_fontsize=16)

# Enhance tick labels
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

# Save the figure
output_path = os.path.join(viz_dir, "regional_true_sentiment_comparison.png")
plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.close()

print(f"Regional TRUE sentiment comparison visualization created at {output_path}")

# Also save the data for reference
df_combined.to_csv(os.path.join(output_dir, "overall", "sentiment_by_country_category.csv"), index=False)