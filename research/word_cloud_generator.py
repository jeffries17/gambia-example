import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np

# Word frequencies from the unique features analysis (expanded to ~30 words)
word_frequencies = {
    # Natural features with sentiment scores
    'whale': 826 * 4.83,  # Multiply by average rating to weight by sentiment
    'island': 421 * 4.59,
    'beach': 302 * 4.60,
    'cave': 235 * 4.47,
    'coral': 96 * 4.66,
    'ocean': 156 * 4.55,
    'tropical': 89 * 4.71,
    'swimming': 187 * 4.68,
    'snorkeling': 143 * 4.72,
    'diving': 167 * 4.75,
    
    # Cultural features with sentiment scores
    'culture': 97 * 4.85,
    'tradition': 90 * 4.58,
    'village': 34 * 4.76,
    'kava': 30 * 4.83,
    'traditional': 84 * 4.56,
    'tongan': 41 * 4.83,
    'ceremony': 18 * 4.72,
    'dance': 82 * 4.24,
    'feast': 50 * 4.66,
    'people': 271 * 4.47,
    'friendly': 195 * 4.82,
    'hospitality': 88 * 4.79,
    'local': 234 * 4.58,
    
    # Experience descriptors with sentiment scores
    'beautiful': 312 * 4.77,
    'amazing': 289 * 4.81,
    'paradise': 167 * 4.85,
    'peaceful': 145 * 4.73,
    'authentic': 98 * 4.69,
    'adventure': 178 * 4.71,
    'experience': 256 * 4.64
}

# Configure the word cloud
wordcloud = WordCloud(
    width=1200,
    height=800,
    background_color='white',
    colormap='viridis',  # Ocean-inspired color scheme
    min_font_size=10,
    max_font_size=150,
    prefer_horizontal=0.7,
    relative_scaling=0.5,  # Balance between word frequency and size
    random_state=42  # For reproducibility
)

# Generate the word cloud
wordcloud.generate_from_frequencies(word_frequencies)

# Create the plot
plt.figure(figsize=(15, 10))

# Add the word cloud
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')

# Add title and subtitle
plt.suptitle('Key Features of Tonga Tourism Experience', fontsize=16, y=0.95)
plt.figtext(0.5, 0.02, 
            'Analysis based on 7,862 visitor reviews. Word size reflects both mention frequency and sentiment rating.\nData collected from verified travel platforms 2023-2024.',
            ha='center', fontsize=10, color='#666666')

# Create output directory if it doesn't exist
import os
output_dir = os.path.join('outputs', 'consolidated_reports', 'visualizations')
os.makedirs(output_dir, exist_ok=True)

# Save the word cloud
plt.savefig(os.path.join(output_dir, 'tonga_features_wordcloud.png'), dpi=300, bbox_inches='tight')
plt.close() 