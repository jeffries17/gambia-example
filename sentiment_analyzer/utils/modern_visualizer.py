#!/usr/bin/env python3
"""
Modern Tourism Visualization Module

Professional, clean, and modern visualizations for tourism analysis
with contemporary design, typography, and color schemes.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.patches as patches
from matplotlib import font_manager
import os
from typing import Dict, List, Optional, Tuple
from collections import Counter
import json

# Modern color palettes
MODERN_COLORS = {
    'primary': '#2E86C1',      # Professional blue
    'secondary': '#F39C12',     # Warm orange
    'success': '#27AE60',       # Fresh green
    'warning': '#E67E22',       # Vibrant orange
    'danger': '#E74C3C',        # Clean red
    'info': '#8E44AD',          # Modern purple
    'dark': '#2C3E50',          # Deep navy
    'light': '#ECF0F1',         # Clean light gray
    'accent': '#1ABC9C',        # Teal accent
    'muted': '#95A5A6'          # Sophisticated gray
}

SENTIMENT_COLORS = {
    'positive': '#27AE60',      # Fresh green
    'negative': '#E74C3C',      # Clean red  
    'neutral': '#95A5A6',       # Sophisticated gray
    'mixed': '#F39C12'          # Warm orange
}

CATEGORY_COLORS = {
    'accommodation': '#3498DB',  # Sky blue
    'restaurants': '#E67E22',    # Carrot orange
    'attractions': '#9B59B6',    # Amethyst purple
    'cultural_heritage': '#1ABC9C', # Turquoise
    'infrastructure': '#E74C3C',  # Alizarin red
    'service_quality': '#F1C40F', # Sun yellow
    'accessibility': '#34495E'    # Wet asphalt
}

class ModernTourismVisualizer:
    """
    Modern, professional visualization class for tourism analysis.
    """
    
    def __init__(self, style='modern'):
        """Initialize with modern styling."""
        self.setup_modern_style()
        
    def setup_modern_style(self):
        """Set up modern matplotlib and seaborn styling."""
        # Set modern matplotlib style
        plt.style.use('default')
        
        # Modern font configuration
        plt.rcParams.update({
            'font.family': ['SF Pro Display', 'Helvetica Neue', 'Arial', 'sans-serif'],
            'font.size': 11,
            'font.weight': 'normal',
            'axes.titlesize': 16,
            'axes.titleweight': '600',
            'axes.labelsize': 12,
            'axes.labelweight': '500',
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 20,
            'figure.titleweight': '700',
            
            # Modern colors and styling
            'axes.facecolor': '#FFFFFF',
            'figure.facecolor': '#FFFFFF',
            'axes.edgecolor': '#E5E5E5',
            'axes.linewidth': 0.8,
            'axes.grid': True,
            'grid.color': '#F5F5F5',
            'grid.linewidth': 0.5,
            'grid.alpha': 0.7,
            
            # Remove spines for cleaner look
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.spines.left': True,
            'axes.spines.bottom': True,
            
            # Modern legend styling
            'legend.frameon': False,
            'legend.loc': 'best',
            
            # High DPI for crisp visuals
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.2
        })
        
        # Set seaborn style
        sns.set_style("whitegrid", {
            'axes.grid': True,
            'grid.color': '#F5F5F5',
            'axes.edgecolor': '#E5E5E5',
            'axes.linewidth': 0.8
        })
        
        # Custom color palette
        sns.set_palette([MODERN_COLORS['primary'], MODERN_COLORS['secondary'], 
                        MODERN_COLORS['success'], MODERN_COLORS['warning'],
                        MODERN_COLORS['info'], MODERN_COLORS['accent']])
    
    def create_modern_dashboard(self, analysis_data: Dict, destination_name: str, 
                              output_path: str = None) -> str:
        """Create a comprehensive modern dashboard."""
        
        # Extract data
        overall = analysis_data.get('overall_sentiment', {})
        aspects = analysis_data.get('aspect_sentiment', {})
        themes = analysis_data.get('recurring_themes', {})
        
        # Create figure with modern layout
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle(f'{destination_name} Tourism Analysis Dashboard', 
                    fontsize=24, fontweight='700', color=MODERN_COLORS['dark'], y=0.96)
        
        # Create grid layout
        gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.3, 
                             left=0.05, right=0.95, top=0.90, bottom=0.05)
        
        # 1. Overall Metrics (Top Row)
        self._create_metric_cards(fig, gs, overall, destination_name)
        
        # 2. Sentiment Distribution (Enhanced Donut Chart)
        ax2 = fig.add_subplot(gs[1, 0])
        self._create_modern_sentiment_donut(ax2, overall)
        
        # 3. Aspect Performance (Modern Bar Chart)
        ax3 = fig.add_subplot(gs[1, 1:3])
        self._create_modern_aspect_bars(ax3, aspects)
        
        # 4. Rating Distribution (Elegant Histogram)
        ax4 = fig.add_subplot(gs[1, 3])
        self._create_modern_rating_dist(ax4, analysis_data)
        
        # 5. Theme Analysis (Modern Horizontal Bars)
        ax5 = fig.add_subplot(gs[2, :2])
        self._create_modern_theme_analysis(ax5, themes)
        
        # 6. Temporal Analysis (Modern Line Chart)
        ax6 = fig.add_subplot(gs[2, 2:])
        self._create_modern_temporal_analysis(ax6, analysis_data)
        
        # 7. Key Insights Panel (Modern Text Box)
        ax7 = fig.add_subplot(gs[3, :])
        self._create_modern_insights_panel(ax7, analysis_data)
        
        # Add modern footer
        self._add_modern_footer(fig)
        
        # Save with high quality
        if not output_path:
            output_path = f'outputs/gambia_insights/modern_dashboard_{destination_name.replace(" ", "_")}.png'
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return output_path
    
    def _create_metric_cards(self, fig, gs, overall: Dict, destination: str):
        """Create modern metric cards at the top."""
        metrics = [
            ('Total Reviews', overall.get('total_reviews', 0), MODERN_COLORS['primary']),
            ('Avg Rating', f"{overall.get('average_rating', 0):.1f}/5", MODERN_COLORS['success']),
            ('Sentiment Score', f"{overall.get('overall_score', 0):.3f}", MODERN_COLORS['info']),
            ('Positive Rate', f"{overall.get('sentiment_distribution', {}).get('positive_percentage', 0):.1f}%", MODERN_COLORS['accent'])
        ]
        
        for i, (title, value, color) in enumerate(metrics):
            ax = fig.add_subplot(gs[0, i])
            
            # Create modern card background
            card = patches.Rectangle((0, 0), 1, 1, linewidth=0, 
                                   facecolor=color, alpha=0.1, transform=ax.transAxes)
            ax.add_patch(card)
            
            # Add border
            border = patches.Rectangle((0, 0), 1, 1, linewidth=2, 
                                     edgecolor=color, facecolor='none', transform=ax.transAxes)
            ax.add_patch(border)
            
            # Add text
            ax.text(0.5, 0.7, str(value), ha='center', va='center', 
                   fontsize=20, fontweight='700', color=color, transform=ax.transAxes)
            ax.text(0.5, 0.3, title, ha='center', va='center', 
                   fontsize=11, fontweight='500', color=MODERN_COLORS['dark'], transform=ax.transAxes)
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
    
    def _create_modern_sentiment_donut(self, ax, overall: Dict):
        """Create modern donut chart for sentiment distribution."""
        dist = overall.get('sentiment_distribution', {})
        
        # Data for donut
        sizes = [dist.get('positive_percentage', 0), 
                dist.get('neutral_percentage', 0),
                dist.get('negative_percentage', 0)]
        labels = ['Positive', 'Neutral', 'Negative']
        colors = [SENTIMENT_COLORS['positive'], SENTIMENT_COLORS['neutral'], SENTIMENT_COLORS['negative']]
        
        # Create donut chart
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                         startangle=90, pctdistance=0.85,
                                         textprops={'fontsize': 10, 'fontweight': '500'})
        
        # Create donut hole
        centre_circle = plt.Circle((0,0), 0.60, fc='white', linewidth=2, edgecolor=MODERN_COLORS['light'])
        ax.add_artist(centre_circle)
        
        # Add center text
        ax.text(0, 0, 'Sentiment\nDistribution', ha='center', va='center',
               fontsize=12, fontweight='600', color=MODERN_COLORS['dark'])
        
        ax.set_title('Sentiment Breakdown', fontsize=14, fontweight='600', 
                    color=MODERN_COLORS['dark'], pad=20)
    
    def _create_modern_aspect_bars(self, ax, aspects: Dict):
        """Create modern horizontal bar chart for aspects."""
        if not aspects:
            ax.text(0.5, 0.5, 'No aspect data available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, color=MODERN_COLORS['muted'])
            ax.axis('off')
            return
        
        # Prepare data
        aspect_names = []
        sentiment_scores = []
        mention_rates = []
        
        for aspect, data in aspects.items():
            aspect_names.append(aspect.replace('_', ' ').title())
            sentiment_scores.append(data.get('average_sentiment', 0))
            mention_rates.append(data.get('mention_percentage', 0))
        
        # Create modern bars
        y_pos = np.arange(len(aspect_names))
        
        # Create bars with gradient effect
        bars = ax.barh(y_pos, sentiment_scores, height=0.6,
                      color=[CATEGORY_COLORS.get(aspect.lower().replace(' ', '_'), MODERN_COLORS['primary']) 
                            for aspect in aspect_names], alpha=0.8)
        
        # Add value labels
        for i, (bar, score, mention) in enumerate(zip(bars, sentiment_scores, mention_rates)):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{score:.3f} ({mention:.1f}%)', 
                   ha='left', va='center', fontsize=9, fontweight='500')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(aspect_names, fontsize=11, fontweight='500')
        ax.set_xlabel('Sentiment Score (Mention Rate)', fontsize=11, fontweight='500')
        ax.set_title('Aspect Performance', fontsize=14, fontweight='600', 
                    color=MODERN_COLORS['dark'], pad=15)
        
        # Modern styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3)
    
    def _create_modern_rating_dist(self, ax, analysis_data: Dict):
        """Create modern rating distribution."""
        # Simulated rating distribution (you can replace with actual data)
        ratings = [1, 2, 3, 4, 5]
        counts = [1, 2, 4, 12, 5]  # Example data
        
        # Create modern bars
        bars = ax.bar(ratings, counts, color=MODERN_COLORS['primary'], alpha=0.8, width=0.7)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{int(height)}', ha='center', va='bottom', 
                   fontsize=10, fontweight='500')
        
        ax.set_xlabel('Rating', fontsize=11, fontweight='500')
        ax.set_ylabel('Count', fontsize=11, fontweight='500')
        ax.set_title('Rating Distribution', fontsize=14, fontweight='600', 
                    color=MODERN_COLORS['dark'], pad=15)
        ax.set_xticks(ratings)
        
        # Modern styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3, axis='y')
    
    def _create_modern_theme_analysis(self, ax, themes: Dict):
        """Create modern theme analysis visualization."""
        if not themes:
            ax.text(0.5, 0.5, 'No theme data available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, color=MODERN_COLORS['muted'])
            ax.axis('off')
            return
        
        # Prepare data
        theme_names = []
        percentages = []
        sentiments = []
        
        for theme, data in list(themes.items())[:6]:  # Top 6 themes
            theme_names.append(theme.replace('_', ' ').title())
            percentages.append(data.get('percentage', 0))
            sentiments.append(data.get('average_sentiment', 0))
        
        # Create horizontal bars with sentiment color coding
        y_pos = np.arange(len(theme_names))
        
        # Color bars based on sentiment
        colors = []
        for sentiment in sentiments:
            if sentiment > 0.1:
                colors.append(SENTIMENT_COLORS['positive'])
            elif sentiment < -0.1:
                colors.append(SENTIMENT_COLORS['negative'])
            else:
                colors.append(SENTIMENT_COLORS['neutral'])
        
        bars = ax.barh(y_pos, percentages, height=0.6, color=colors, alpha=0.8)
        
        # Add labels
        for i, (bar, pct, sent) in enumerate(zip(bars, percentages, sentiments)):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                   f'{pct:.1f}% (±{sent:.2f})', 
                   ha='left', va='center', fontsize=9, fontweight='500')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(theme_names, fontsize=10, fontweight='500')
        ax.set_xlabel('Mention Percentage', fontsize=11, fontweight='500')
        ax.set_title('Key Themes Analysis', fontsize=14, fontweight='600', 
                    color=MODERN_COLORS['dark'], pad=15)
        
        # Modern styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3)
    
    def _create_modern_temporal_analysis(self, ax, analysis_data: Dict):
        """Create modern temporal analysis."""
        # Example temporal data (replace with actual data if available)
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        review_counts = [3, 4, 6, 5, 4, 2]
        sentiment_trend = [0.15, 0.18, 0.12, 0.20, 0.16, 0.14]
        
        # Create dual axis for modern look
        ax2 = ax.twinx()
        
        # Bar chart for review counts
        bars = ax.bar(months, review_counts, alpha=0.6, color=MODERN_COLORS['light'], 
                     edgecolor=MODERN_COLORS['primary'], linewidth=2, label='Review Count')
        
        # Line chart for sentiment trend
        line = ax2.plot(months, sentiment_trend, color=MODERN_COLORS['accent'], 
                       linewidth=3, marker='o', markersize=8, markerfacecolor='white',
                       markeredgecolor=MODERN_COLORS['accent'], markeredgewidth=2,
                       label='Sentiment Trend')
        
        # Styling
        ax.set_ylabel('Review Count', fontsize=11, fontweight='500', color=MODERN_COLORS['primary'])
        ax2.set_ylabel('Sentiment Score', fontsize=11, fontweight='500', color=MODERN_COLORS['accent'])
        ax.set_title('Temporal Analysis', fontsize=14, fontweight='600', 
                    color=MODERN_COLORS['dark'], pad=15)
        
        # Modern styling
        ax.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax.grid(True, alpha=0.3)
        
        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', frameon=False)
    
    def _create_modern_insights_panel(self, ax, analysis_data: Dict):
        """Create modern insights panel."""
        executive_summary = analysis_data.get('executive_summary', {})
        
        # Panel background
        panel = patches.Rectangle((0, 0), 1, 1, linewidth=2, 
                                edgecolor=MODERN_COLORS['primary'], 
                                facecolor=MODERN_COLORS['primary'], alpha=0.05, 
                                transform=ax.transAxes)
        ax.add_patch(panel)
        
        # Title
        ax.text(0.02, 0.9, '🎯 Key Insights & Recommendations', 
               fontsize=16, fontweight='700', color=MODERN_COLORS['dark'], 
               transform=ax.transAxes)
        
        # Strengths
        strengths = executive_summary.get('strengths', [])[:3]
        if strengths:
            ax.text(0.02, 0.75, '✅ Strengths:', fontsize=12, fontweight='600', 
                   color=SENTIMENT_COLORS['positive'], transform=ax.transAxes)
            for i, strength in enumerate(strengths):
                ax.text(0.04, 0.68 - i*0.08, f"• {strength}", fontsize=10, 
                       color=MODERN_COLORS['dark'], transform=ax.transAxes)
        
        # Improvements
        improvements = executive_summary.get('areas_for_improvement', [])[:3]
        if improvements:
            ax.text(0.52, 0.75, '⚠️ Areas for Improvement:', fontsize=12, fontweight='600', 
                   color=SENTIMENT_COLORS['negative'], transform=ax.transAxes)
            for i, improvement in enumerate(improvements):
                ax.text(0.54, 0.68 - i*0.08, f"• {improvement}", fontsize=10, 
                       color=MODERN_COLORS['dark'], transform=ax.transAxes)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    def _add_modern_footer(self, fig):
        """Add modern footer to the dashboard."""
        fig.text(0.99, 0.01, 'Generated by Tourism Analytics AI • Powered by Advanced Sentiment Analysis', 
                ha='right', va='bottom', fontsize=8, style='italic', 
                color=MODERN_COLORS['muted'], alpha=0.8)
    
    def create_modern_wordcloud(self, text_data: Dict, title: str, 
                               output_path: str = None, aspect_type: str = 'general') -> str:
        """Create modern, professional word clouds."""
        
        # Modern color schemes by aspect
        color_schemes = {
            'cultural_heritage': ['#8E44AD', '#9B59B6', '#AF7AC5', '#C39BD3'],
            'infrastructure': ['#E74C3C', '#EC7063', '#F1948A', '#F5B7B1'],
            'service_tourism': ['#3498DB', '#5DADE2', '#85C1E9', '#AED6F1'],
            'accessibility': ['#27AE60', '#58D68D', '#82E0AA', '#A9DFBF'],
            'general': ['#2E86C1', '#F39C12', '#27AE60', '#8E44AD', '#E74C3C']
        }
        
        colors = color_schemes.get(aspect_type, color_schemes['general'])
        
        # Create word cloud
        wordcloud = WordCloud(
            width=1200, height=600,
            background_color='white',
            max_words=50,
            colormap=None,
            color_func=lambda *args, **kwargs: np.random.choice(colors),
            font_path=None,  # Use system default
            prefer_horizontal=0.7,
            min_font_size=12,
            max_font_size=80,
            relative_scaling=0.5,
            margin=10
        ).generate_from_frequencies(text_data)
        
        # Create modern figure
        fig, ax = plt.subplots(figsize=(15, 8), dpi=300)
        fig.patch.set_facecolor('white')
        
        # Display word cloud
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        
        # Modern title
        ax.set_title(f'{title} Word Cloud', fontsize=20, fontweight='700', 
                    color=MODERN_COLORS['dark'], pad=20)
        
        # Modern border
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Save
        if not output_path:
            safe_title = title.lower().replace(' ', '_').replace('&', 'and')
            output_path = f'outputs/gambia_insights/modern_wordcloud_{safe_title}.png'
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return output_path

# Convenience function
def create_modern_tourism_visuals(analysis_data: Dict, destination_name: str, 
                                 output_dir: str = 'outputs') -> Dict[str, str]:
    """
    Create complete set of modern tourism visualizations.
    
    Returns:
        Dict of visualization paths created
    """
    visualizer = ModernTourismVisualizer()
    
    visuals = {}
    
    # Main dashboard
    dashboard_path = visualizer.create_modern_dashboard(
        analysis_data, destination_name, 
        f'{output_dir}/modern_dashboard_{destination_name.replace(" ", "_")}.png'
    )
    visuals['dashboard'] = dashboard_path
    
    # Word clouds for different aspects
    aspects_data = analysis_data.get('aspect_sentiment', {})
    for aspect, data in aspects_data.items():
        if 'keywords' in data:  # Assuming keywords are available
            wordcloud_path = visualizer.create_modern_wordcloud(
                data['keywords'], f'{aspect.replace("_", " ").title()}',
                f'{output_dir}/modern_wordcloud_{aspect}.png', aspect
            )
            visuals[f'wordcloud_{aspect}'] = wordcloud_path
    
    return visuals 