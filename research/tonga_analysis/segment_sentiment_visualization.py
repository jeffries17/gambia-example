#!/usr/bin/env python3
"""
Segment Sentiment Visualization Script

This script creates improved visualizations for sentiment analysis across different
traveler segments (trip types and countries) to better explain the differences
in sentiment scores.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import json
import random

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Define colors for segments
TRIP_TYPE_COLORS = {
    'business': '#1f77b4',
    'couple': '#ff7f0e',
    'family': '#2ca02c',
    'friends': '#d62728',
    'solo': '#9467bd'
}

# Define colors for countries
COUNTRY_COLORS = {
    'United States': '#1f77b4',
    'United Kingdom': '#ff7f0e',
    'Tonga': '#2ca02c',
    'New Zealand': '#d62728',
    'Australia': '#9467bd'
}

# Define sentiment color gradient
SENTIMENT_COLORS = {
    'very_negative': '#d7191c',
    'negative': '#fdae61',
    'neutral': '#ffffbf',
    'positive': '#a6d96a',
    'very_positive': '#1a9641'
}

class SegmentSentimentVisualizer:
    """
    Creates visual representations of sentiment data across different traveler segments.
    """
    
    def __init__(self, output_dir='outputs/segment_sentiment_viz'):
        """
        Initialize the visualizer.
        
        Parameters:
        - output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Example data - replace with actual data loading from files
        self.generate_example_data()
    
    def generate_example_data(self):
        """
        Generate example data to demonstrate visualizations.
        """
        # Define trip types and countries
        trip_types = ['business', 'couple', 'family', 'friends', 'solo']
        countries = ['United States', 'United Kingdom', 'Tonga', 'New Zealand', 'Australia']
        
        # Define themes
        themes = ['service', 'food_quality', 'location', 'value', 'room_quality', 'activities']
        
        # Generate theme sentiment data for trip types
        np.random.seed(42)  # For reproducibility
        
        # Trip type theme sentiment - this explains why couples and families have higher scores
        # (they have higher scores in specific themes that matter to them)
        self.trip_type_theme_sentiment = {
            'business': {
                'service': 0.45,
                'food_quality': 0.38,
                'location': 0.51,
                'value': 0.25,
                'room_quality': 0.35,
                'activities': 0.22
            },
            'couple': {
                'service': 0.38,
                'food_quality': 0.65,
                'location': 0.72,
                'value': 0.41,
                'room_quality': 0.53,
                'activities': 0.69
            },
            'family': {
                'service': 0.31,
                'food_quality': 0.58,
                'location': 0.62,
                'value': 0.29,
                'room_quality': 0.45,
                'activities': 0.71
            },
            'friends': {
                'service': 0.22,
                'food_quality': 0.70,
                'location': 0.59,
                'value': 0.35,
                'room_quality': 0.30,
                'activities': 0.66
            },
            'solo': {
                'service': 0.52,
                'food_quality': 0.60,
                'location': 0.48,
                'value': 0.55,
                'room_quality': 0.39,
                'activities': 0.43
            }
        }
        
        # Country theme sentiment
        self.country_theme_sentiment = {
            'United States': {
                'service': 0.32,
                'food_quality': 0.55,
                'location': 0.68,
                'value': 0.25,
                'room_quality': 0.41,
                'activities': 0.61
            },
            'United Kingdom': {
                'service': 0.29,
                'food_quality': 0.61,
                'location': 0.71,
                'value': 0.35,
                'room_quality': 0.38,
                'activities': 0.56
            },
            'Tonga': {
                'service': 0.45,
                'food_quality': 0.72,
                'location': 0.65,
                'value': 0.51,
                'room_quality': 0.49,
                'activities': 0.59
            },
            'New Zealand': {
                'service': 0.38,
                'food_quality': 0.59,
                'location': 0.74,
                'value': 0.47,
                'room_quality': 0.44,
                'activities': 0.63
            },
            'Australia': {
                'service': 0.33,
                'food_quality': 0.57,
                'location': 0.70,
                'value': 0.41,
                'room_quality': 0.46,
                'activities': 0.62
            }
        }
        
        # Overall average sentiment for comparison
        self.overall_sentiment = 0.35
        
        # Example sentiment words for each segment (for word clouds)
        self.segment_sentiment_words = {
            'couple': {
                'positive': {
                    'romantic': 0.85, 'breathtaking': 0.90, 'paradise': 0.88, 'perfect': 0.82,
                    'amazing': 0.79, 'spectacular': 0.86, 'honeymoon': 0.91, 'intimate': 0.83,
                    'stunning': 0.87, 'beautiful': 0.81, 'relaxing': 0.76, 'peaceful': 0.80,
                    'wonderful': 0.78, 'memorable': 0.84, 'sunset': 0.75, 'private': 0.77
                },
                'negative': {
                    'overpriced': -0.65, 'disappointing': -0.71, 'loud': -0.58, 'crowded': -0.55,
                    'expensive': -0.61, 'dirty': -0.68, 'disorganized': -0.64, 'dated': -0.53
                }
            },
            'family': {
                'positive': {
                    'fun': 0.78, 'friendly': 0.82, 'kids': 0.75, 'activities': 0.80,
                    'spacious': 0.76, 'playground': 0.79, 'beach': 0.85, 'safe': 0.88,
                    'enjoyable': 0.81, 'convenient': 0.74, 'entertaining': 0.83, 'pool': 0.77
                },
                'negative': {
                    'noisy': -0.59, 'unsuitable': -0.66, 'expensive': -0.60, 'unsafe': -0.75,
                    'uncomfortable': -0.62, 'boring': -0.57, 'limited': -0.55, 'crowded': -0.58
                }
            },
            'business': {
                'positive': {
                    'efficient': 0.72, 'convenient': 0.69, 'comfortable': 0.68, 'reliable': 0.75,
                    'professional': 0.78, 'productive': 0.71, 'wifi': 0.65, 'central': 0.67
                },
                'negative': {
                    'unreliable': -0.68, 'slow': -0.61, 'inconvenient': -0.67, 'expensive': -0.59,
                    'outdated': -0.55, 'noisy': -0.62, 'uncomfortable': -0.66, 'dirty': -0.65
                }
            }
        }
    
    def create_heatmap_by_theme(self):
        """
        Create a heatmap showing sentiment scores by segment and theme.
        """
        print("Creating theme sentiment heatmaps...")
        
        # Create trip type theme sentiment heatmap
        self._create_theme_heatmap(
            self.trip_type_theme_sentiment,
            'Trip Type',
            'trip_type_theme_heatmap.png'
        )
        
        # Create country theme sentiment heatmap
        self._create_theme_heatmap(
            self.country_theme_sentiment,
            'Country',
            'country_theme_heatmap.png'
        )
    
    def _create_theme_heatmap(self, data, segment_type, filename):
        """
        Create a heatmap for a specific segment type (trip type or country).
        
        Parameters:
        - data: Dictionary of segment sentiment data by theme
        - segment_type: Type of segment ('Trip Type' or 'Country')
        - filename: Output filename
        """
        # Create DataFrame from the data
        segments = list(data.keys())
        themes = list(data[segments[0]].keys())
        
        # Capitalize and format theme names for display
        display_themes = [theme.replace('_', ' ').title() for theme in themes]
        
        # Create the heatmap data
        heatmap_data = pd.DataFrame(
            [[data[segment][theme] for theme in themes] for segment in segments],
            index=segments,
            columns=display_themes
        )
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Create a normalized colormap for the sentiment range
        cmap = LinearSegmentedColormap.from_list(
            'sentiment_cmap',
            [SENTIMENT_COLORS['negative'], SENTIMENT_COLORS['neutral'], SENTIMENT_COLORS['very_positive']],
            N=100
        )
        
        # Generate the heatmap
        ax = sns.heatmap(
            heatmap_data,
            annot=True,
            fmt='.2f',
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            linewidths=.5,
            cbar_kws={'label': 'Sentiment Score'}
        )
        
        # Add a horizontal line to indicate the overall average sentiment
        plt.axhline(y=0, color='black', linewidth=2)
        
        # Set title and labels
        plt.title(f'Sentiment Scores by {segment_type} and Theme', fontsize=16, pad=20)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Created {filename}")
    
    def explain_segment_differences(self):
        """
        Create visualizations that explain why certain segments have higher sentiment scores.
        """
        print("Creating segment explanation visuals...")
        
        # Calculate overall averages for each segment
        trip_type_avgs = {
            trip_type: np.mean(list(themes.values())) 
            for trip_type, themes in self.trip_type_theme_sentiment.items()
        }
        
        country_avgs = {
            country: np.mean(list(themes.values())) 
            for country, themes in self.country_theme_sentiment.items()
        }
        
        # Create comparative bar charts
        self._create_segment_comparison(
            trip_type_avgs,
            self.trip_type_theme_sentiment,
            'Trip Type',
            TRIP_TYPE_COLORS,
            'trip_type_sentiment_explanation.png'
        )
        
        self._create_segment_comparison(
            country_avgs,
            self.country_theme_sentiment,
            'Country',
            COUNTRY_COLORS,
            'country_sentiment_explanation.png'
        )
    
    def _create_segment_comparison(self, avg_data, theme_data, segment_type, color_map, filename):
        """
        Create a comparison visualization showing why segments differ in sentiment.
        
        Parameters:
        - avg_data: Dictionary of average sentiment scores by segment
        - theme_data: Dictionary of theme sentiment scores by segment
        - segment_type: Type of segment ('Trip Type' or 'Country')
        - color_map: Dictionary mapping segments to colors
        - filename: Output filename
        """
        segments = list(avg_data.keys())
        themes = list(theme_data[segments[0]].keys())
        
        # Sort segments by average sentiment score
        segments = sorted(segments, key=lambda x: avg_data[x], reverse=True)
        
        # Create the plot with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [1, 3]})
        
        # Plot 1: Average sentiment by segment compared to overall average
        pos = np.arange(len(segments))
        bars = ax1.bar(
            pos,
            [avg_data[segment] for segment in segments],
            color=[color_map.get(segment, '#AAAAAA') for segment in segments]
        )
        
        # Add a horizontal line for the overall average
        ax1.axhline(y=self.overall_sentiment, color='red', linestyle='--', label=f'Overall Average ({self.overall_sentiment:.2f})')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.01,
                f'{height:.2f}',
                ha='center',
                va='bottom',
                fontsize=10
            )
        
        # Customize the plot
        ax1.set_title(f'Average Sentiment by {segment_type}', fontsize=16)
        ax1.set_xticks(pos)
        ax1.set_xticklabels(segments)
        ax1.set_ylabel('Average Sentiment Score')
        ax1.set_ylim(0, 1.0)
        ax1.legend()
        
        # Plot 2: Theme breakdown for each segment
        theme_labels = [theme.replace('_', ' ').title() for theme in themes]
        
        width = 0.15
        pos_segment = np.arange(len(themes))
        
        for i, segment in enumerate(segments):
            # Offset position for each segment
            pos = pos_segment + (i - len(segments)/2 + 0.5) * width
            
            # Get theme values for this segment
            theme_values = [theme_data[segment][theme] for theme in themes]
            
            # Plot bars
            bars = ax2.bar(
                pos,
                theme_values,
                width,
                label=segment,
                color=color_map.get(segment, '#AAAAAA')
            )
            
            # Add segment value labels
            for bar in bars:
                height = bar.get_height()
                ax2.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + 0.01,
                    f'{height:.2f}',
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    rotation=90
                )
        
        # Add a horizontal line for the overall average
        ax2.axhline(y=self.overall_sentiment, color='red', linestyle='--', label=f'Overall Average ({self.overall_sentiment:.2f})')
        
        # Customize the plot
        ax2.set_title(f'Sentiment Scores by Theme for Each {segment_type}', fontsize=16)
        ax2.set_xticks(pos_segment)
        ax2.set_xticklabels(theme_labels, rotation=45, ha='right')
        ax2.set_ylabel('Sentiment Score')
        ax2.set_ylim(0, 1.0)
        ax2.legend()
        
        # Add explanatory annotation
        ax2.annotate(
            f"Why do some {segment_type.lower()}s have higher sentiment scores?\n"
            f"They score much higher on specific themes that matter to them.",
            xy=(0.5, 0.97),
            xycoords='figure fraction',
            ha='center',
            va='top',
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8)
        )
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3)
        
        # Save the figure
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Created {filename}")
    
    def create_sentiment_word_clouds(self):
        """
        Create word clouds showing the sentiment words for different segments.
        """
        print("Creating sentiment word clouds...")
        
        # Create word clouds for selected segments
        for segment, sentiment_words in self.segment_sentiment_words.items():
            self._create_segment_word_cloud(segment, sentiment_words, f'{segment}_sentiment_wordcloud.png')
    
    def _create_segment_word_cloud(self, segment, sentiment_words, filename):
        """
        Create a word cloud for a specific segment.
        
        Parameters:
        - segment: Segment name
        - sentiment_words: Dictionary of positive and negative words with sentiment scores
        - filename: Output filename
        """
        # Combine positive and negative words
        all_words = {}
        all_words.update(sentiment_words['positive'])
        all_words.update(sentiment_words['negative'])
        
        # Create a colormap function based on sentiment
        def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            sentiment = all_words.get(word, 0)
            if sentiment > 0.7:
                return SENTIMENT_COLORS['very_positive']
            elif sentiment > 0.3:
                return SENTIMENT_COLORS['positive']
            elif sentiment > -0.3:
                return SENTIMENT_COLORS['neutral']
            elif sentiment > -0.7:
                return SENTIMENT_COLORS['negative']
            else:
                return SENTIMENT_COLORS['very_negative']
        
        # Generate the word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=100,
            colormap='RdYlGn',
            prefer_horizontal=0.9,
            color_func=color_func
        ).generate_from_frequencies(all_words)
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        
        # Add a title
        plt.title(f'Sentiment Words for {segment.capitalize()} Travelers', fontsize=16, pad=20)
        
        # Add a legend for sentiment colors
        sentiment_labels = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
        sentiment_colors = [
            SENTIMENT_COLORS['very_negative'],
            SENTIMENT_COLORS['negative'],
            SENTIMENT_COLORS['neutral'],
            SENTIMENT_COLORS['positive'],
            SENTIMENT_COLORS['very_positive']
        ]
        
        # Create a custom legend
        patches = [plt.Rectangle((0, 0), 1, 1, color=color) for color in sentiment_colors]
        plt.legend(patches, sentiment_labels, loc='upper left', bbox_to_anchor=(1.05, 1))
        
        # Add explanation
        plt.figtext(
            0.5, 0.01,
            f"Words sized by frequency in {segment} reviews, colored by sentiment intensity.\n"
            f"These specific words contribute to the higher overall sentiment for this segment.",
            ha='center',
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8)
        )
        
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Created {filename}")
    
    def run_all_visualizations(self):
        """
        Run all visualization methods.
        """
        self.create_heatmap_by_theme()
        self.explain_segment_differences()
        self.create_sentiment_word_clouds()
        
        print(f"All visualizations saved to {self.output_dir}")


if __name__ == "__main__":
    visualizer = SegmentSentimentVisualizer()
    visualizer.run_all_visualizations()