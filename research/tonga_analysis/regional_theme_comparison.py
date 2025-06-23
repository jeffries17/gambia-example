"""
Regional Theme Comparison - Generates comparative visualizations for theme sentiment across countries
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import re
from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
from visualization_styles import REGION_COLORS, get_standard_sentiment_cmap, SENTIMENT_MIN, SENTIMENT_MID_LOW, SENTIMENT_MID_HIGH, SENTIMENT_MAX

class RegionalThemeComparison:
    """
    Analyzes and compares theme sentiment across different countries in the region.
    Generates visualizations showing how different countries compare on specific themes.
    """
    
    def __init__(self, output_dir='outputs/regional_comparison/theme_sentiment'):
        """
        Initialize the regional theme comparison.
        
        Parameters:
        - output_dir: Directory to save comparison outputs
        """
        self.output_dir = output_dir
        self.data = {}
        self.themes = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize sentiment thresholds
        self.pos_threshold = 0.2
        self.neg_threshold = -0.1
        
        # Download required NLTK data
        self._download_nltk_data()
        
    def _download_nltk_data(self):
        """Download required NLTK data packages."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
    def load_data(self):
        """
        Load regional comparison data from files.
        """
        data_sources = [
            {
                'category': 'accommodation',
                'path': 'outputs/regional_comparison/accommodations/regional_comparison.json'
            },
            {
                'category': 'attraction',
                'path': 'outputs/regional_comparison/attractions/regional_comparison.json'
            },
            {
                'category': 'restaurant',
                'path': 'outputs/regional_comparison/restaurants/regional_comparison.json'
            }
        ]
        
        for source in data_sources:
            try:
                if os.path.exists(source['path']):
                    with open(source['path'], 'r', encoding='utf-8') as f:
                        self.data[source['category']] = json.load(f)
            except Exception as e:
                print(f"Error loading {source['path']}: {str(e)}")
                
        if not self.data:
            print("Warning: No regional comparison data found.")
            
    def analyze_themes(self):
        """
        Analyze sentiment by themes across countries and categories.
        
        Returns:
        - Dictionary with theme analysis results
        """
        if not self.data:
            print("No data to analyze. Please load data first.")
            return {}
            
        print("Analyzing themes across regions...")
        
        # Base structure for theme analysis
        results = {
            'by_category': {},
            'overall': {}
        }
        
        # Define common themes across categories
        category_themes = {
            'accommodation': [
                'cleanliness', 'staff', 'location', 'value', 
                'room_quality', 'facilities'
            ],
            'attraction': [
                'activity_quality', 'natural_beauty', 'guides', 
                'value', 'uniqueness', 'accessibility'
            ],
            'restaurant': [
                'food_quality', 'service', 'value', 'ambiance',
                'menu_variety', 'local_cuisine'
            ]
        }
        
        # Process each category's data
        for category, category_data in self.data.items():
            if not category_data or 'countries' not in category_data:
                continue
                
            results['by_category'][category] = {
                'by_theme': {}
            }
            
            themes = category_themes.get(category, [])
            
            # Analyze themes
            for theme in themes:
                theme_results = self._analyze_theme_by_country(
                    category_data, theme, category=category
                )
                if theme_results:
                    results['by_category'][category]['by_theme'][theme] = theme_results
                    
        # Calculate cross-category insights
        results['overall'] = self._calculate_overall_insights(results['by_category'])
        
        # Save results
        self._save_results(results)
        
        return results
        
    def _analyze_theme_by_country(self, data, theme, category=None):
        """
        Analyze a specific theme across countries.
        
        Parameters:
        - data: Category data
        - theme: Theme to analyze
        - category: Category name
        
        Returns:
        - Theme analysis results
        """
        if not data or 'countries' not in data:
            return {}
            
        # Structure for theme results
        theme_results = {
            'overall': {
                'avg_sentiment': 0,
                'mention_count': 0,
                'mention_percentage': 0
            },
            'by_country': {}
        }
        
        # Track total mentions and reviews for percentage calculation
        total_reviews = 0
        total_mentions = 0
        total_sentiment_sum = 0
        
        # Process each country's data
        for country, country_data in data['countries'].items():
            if 'reviews' not in country_data:
                continue
                
            # Get review texts and check for theme mentions
            reviews = country_data['reviews']
            country_review_count = len(reviews)
            
            # Collect mentions of the theme
            mentions = []
            sentences_with_theme = []
            
            for review in reviews:
                if 'text' not in review:
                    continue
                    
                # Check if theme is mentioned in the review
                review_text = review['text'].lower()
                theme_terms = self._get_theme_terms(theme)
                
                if any(term in review_text for term in theme_terms):
                    # Review mentions theme
                    sentences = sent_tokenize(review_text)
                    
                    # Find sentences containing theme terms
                    theme_sentences = []
                    for sentence in sentences:
                        if any(term in sentence for term in theme_terms):
                            theme_sentences.append(sentence)
                            
                    if theme_sentences:
                        # Add to mentions
                        mentions.append({
                            'review_id': review.get('review_id'),
                            'sentences': theme_sentences,
                            'rating': review.get('rating', 0)
                        })
                        sentences_with_theme.extend(theme_sentences)
            
            # Calculate stats for this country
            mention_count = len(mentions)
            mention_percentage = (
                (mention_count / country_review_count) * 100 
                if country_review_count > 0 else 0
            )
            
            # Calculate average sentiment of mentions
            sentiment_scores = []
            
            for sentence in sentences_with_theme:
                # Calculate TextBlob sentiment
                blob = TextBlob(sentence)
                sentiment = blob.sentiment.polarity
                sentiment_scores.append(sentiment)
                
            avg_sentiment = (
                sum(sentiment_scores) / len(sentiment_scores) 
                if sentiment_scores else 0
            )
            
            # Store country results
            theme_results['by_country'][country] = {
                'review_count': country_review_count,
                'mention_count': mention_count,
                'mention_percentage': mention_percentage,
                'avg_sentiment': avg_sentiment
            }
            
            # Update overall totals
            total_reviews += country_review_count
            total_mentions += mention_count
            total_sentiment_sum += avg_sentiment * mention_count
            
        # Calculate overall stats
        theme_results['overall']['mention_count'] = total_mentions
        theme_results['overall']['mention_percentage'] = (
            (total_mentions / total_reviews) * 100 if total_reviews > 0 else 0
        )
        theme_results['overall']['avg_sentiment'] = (
            total_sentiment_sum / total_mentions if total_mentions > 0 else 0
        )
        
        return theme_results
        
    def _get_theme_terms(self, theme):
        """
        Get alternative terms for a theme.
        
        Parameters:
        - theme: Base theme name
        
        Returns:
        - List of terms to search for
        """
        theme_term_mapping = {
            # Accommodation themes
            'cleanliness': ['clean', 'cleanliness', 'dirty', 'spotless', 'hygiene', 'tidy'],
            'staff': ['staff', 'service', 'helpful', 'friendly', 'reception', 'hospitality'],
            'location': ['location', 'located', 'central', 'convenient', 'walking distance'],
            'value': ['value', 'price', 'worth', 'expensive', 'cheap', 'affordable', 'cost'],
            'room_quality': ['room', 'bed', 'comfortable', 'spacious', 'quiet', 'noisy'],
            'facilities': ['facilities', 'pool', 'wifi', 'breakfast', 'amenities', 'restaurant'],
            
            # Attraction themes
            'activity_quality': ['activity', 'experience', 'tour', 'enjoyable', 'fun'],
            'natural_beauty': ['beautiful', 'scenery', 'views', 'landscape', 'nature'],
            'guides': ['guide', 'informed', 'knowledgeable', 'explained', 'informative'],
            'uniqueness': ['unique', 'special', 'different', 'one of a kind', 'memorable'],
            'accessibility': ['accessible', 'easy', 'difficult', 'pathway', 'walking'],
            
            # Restaurant themes
            'food_quality': ['food', 'delicious', 'tasty', 'flavors', 'quality'],
            'service': ['service', 'staff', 'attentive', 'slow', 'quick', 'waiter'],
            'ambiance': ['ambiance', 'atmosphere', 'decor', 'setting', 'cozy', 'lively'],
            'menu_variety': ['menu', 'variety', 'options', 'choices', 'selection'],
            'local_cuisine': ['local', 'traditional', 'authentic', 'tongan', 'specialty']
        }
        
        # Default to the theme itself if no mapping exists
        return theme_term_mapping.get(theme, [theme])
        
    def _calculate_overall_insights(self, category_results):
        """
        Calculate cross-category insights.
        
        Parameters:
        - category_results: Results by category
        
        Returns:
        - Overall insights dictionary
        """
        insights = {
            'top_themes_by_country': {},
            'country_rankings': {},
            'regional_comparisons': {}
        }
        
        # Process themes by country to get top performing themes
        country_themes = defaultdict(dict)
        
        for category, category_data in category_results.items():
            if 'by_theme' not in category_data:
                continue
                
            for theme, theme_data in category_data['by_theme'].items():
                if 'by_country' not in theme_data:
                    continue
                    
                for country, country_data in theme_data['by_country'].items():
                    # Store theme data for this country
                    theme_key = f"{category}_{theme}"
                    country_themes[country][theme_key] = {
                        'category': category,
                        'theme': theme,
                        'sentiment': country_data['avg_sentiment'],
                        'mention_percentage': country_data['mention_percentage']
                    }
                    
        # Find top themes for each country
        for country, themes in country_themes.items():
            # Sort themes by sentiment
            sorted_themes = sorted(
                themes.items(), 
                key=lambda x: x[1]['sentiment'], 
                reverse=True
            )
            
            # Get top and bottom themes
            top_themes = sorted_themes[:3] if len(sorted_themes) >= 3 else sorted_themes
            bottom_themes = sorted_themes[-3:] if len(sorted_themes) >= 3 else sorted_themes
            
            insights['top_themes_by_country'][country] = {
                'top_themes': [
                    {
                        'category': item[1]['category'],
                        'theme': item[1]['theme'],
                        'sentiment': item[1]['sentiment']
                    } 
                    for item in top_themes
                ],
                'bottom_themes': [
                    {
                        'category': item[1]['category'],
                        'theme': item[1]['theme'],
                        'sentiment': item[1]['sentiment']
                    } 
                    for item in reversed(bottom_themes)
                ]
            }
            
        # Calculate country rankings by category
        for category, category_data in category_results.items():
            if 'by_theme' not in category_data:
                continue
                
            # Gather country sentiment data
            country_scores = defaultdict(list)
            
            for theme_data in category_data['by_theme'].values():
                if 'by_country' not in theme_data:
                    continue
                    
                for country, data in theme_data['by_country'].items():
                    country_scores[country].append(data['avg_sentiment'])
                    
            # Calculate average sentiment across all themes
            country_avg_scores = {
                country: sum(scores) / len(scores) if scores else 0
                for country, scores in country_scores.items()
            }
            
            # Rank countries by average sentiment
            sorted_countries = sorted(
                country_avg_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            insights['country_rankings'][category] = [
                {'country': country, 'avg_sentiment': score}
                for country, score in sorted_countries
            ]
            
        return insights
        
    def _save_results(self, results):
        """
        Save analysis results to a JSON file.
        
        Parameters:
        - results: Analysis results
        """
        output_file = os.path.join(self.output_dir, 'regional_theme_comparison.json')
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            print(f"Error saving results: {str(e)}")
            
    def create_visualizations(self, results):
        """
        Create visualizations for theme comparisons.
        
        Parameters:
        - results: Theme analysis results
        """
        if not results or 'by_category' not in results:
            print("No results to visualize.")
            return
            
        print("Creating theme comparison visualizations...")
        
        # Create visualizations directory
        viz_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Process each category
        for category, category_data in results['by_category'].items():
            if 'by_theme' not in category_data:
                continue
                
            # Create theme comparisons for each category
            self._create_theme_comparisons(category, category_data['by_theme'], viz_dir)
            
            # Create country-specific theme charts
            self._create_country_theme_charts(category, category_data['by_theme'], viz_dir)
            
        print("Visualization creation complete.")
        
    def _create_theme_comparisons(self, category, theme_data, viz_dir):
        """
        Create comparative visualizations for themes.
        
        Parameters:
        - category: Category name
        - theme_data: Theme data dictionary
        - viz_dir: Visualization directory
        """
        # Define visualization tasks
        comparison_tasks = [
            {
                'theme': theme,
                'metric': 'sentiment',
                'title_suffix': 'Sentiment Comparison'
            },
            {
                'theme': theme,
                'metric': 'mention',
                'title_suffix': 'Mention Comparison'
            },
            {
                'theme': theme,
                'metric': 'combined',
                'title_suffix': 'Combined Analysis'
            }
        ]
        
        # Create each theme comparison
        for theme, theme_results in theme_data.items():
            if 'by_country' not in theme_results:
                continue
                
            theme_display = theme.replace('_', ' ').title()
            
            # Prepare data for visualization
            countries = []
            sentiments = []
            mentions = []
            
            for country, data in theme_results['by_country'].items():
                countries.append(country)
                sentiments.append(data['avg_sentiment'])
                mentions.append(data['mention_percentage'])
                
            # Create a DataFrame for plotting
            df = pd.DataFrame({
                'Country': [c.title() for c in countries],
                'Sentiment': sentiments,
                'Mention %': mentions
            })
            
            # Sort by sentiment
            df = df.sort_values('Sentiment', ascending=False)
            
            # Get colors for countries
            sorted_countries = [c.lower() for c in df['Country']]
            sorted_colors = [REGION_COLORS.get(c.lower(), '#AAAAAA') for c in sorted_countries]
            
            # 1. Create sentiment comparison
            plt.figure(figsize=(12, 6))
            
            # Color by sentiment value using standard scale
            cmap, norm = get_standard_sentiment_cmap()
            colors = cmap(norm(df['Sentiment']))
            
            bars = plt.bar(df['Country'], df['Sentiment'], color=sorted_colors)
            
            # Add value labels
            for bar, value in zip(bars, df['Sentiment']):
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + 0.02 if height >= 0 else height - 0.08,
                    f'{value:.2f}',
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontweight='bold'
                )
                
            # Format for display
            category_display = category.title()
            
            plt.title(f'{theme_display} Sentiment by Country - {category_display}', fontsize=14, pad=20)
            plt.ylabel('Sentiment Score', fontsize=12)
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.2)
            plt.grid(axis='y', alpha=0.3)
            plt.ylim(SENTIMENT_MIN, SENTIMENT_MAX)  # Use standardized scale limits
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join(viz_dir, f'{category}_{theme}_sentiment_comparison.png'), dpi=300)
            plt.close()
            
            # 2. Create mention comparison
            plt.figure(figsize=(12, 6))
            
            bars = plt.bar(df['Country'], df['Mention %'], color=sorted_colors)
            
            # Add value labels
            for bar, value in zip(bars, df['Mention %']):
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + 1,
                    f'{value:.1f}%',
                    ha='center', va='bottom',
                    fontweight='bold'
                )
                
            plt.title(f'{theme_display} Mention Percentage by Country - {category_display}', fontsize=14, pad=20)
            plt.ylabel('Mention Percentage', fontsize=12)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join(viz_dir, f'{category}_{theme}_mention_comparison.png'), dpi=300)
            plt.close()
            
            # 3. Create combined chart
            plt.figure(figsize=(12, 6))
            
            # Primary axis for sentiment
            ax1 = plt.gca()
            bars = ax1.bar(df['Country'], df['Sentiment'], color=sorted_colors, alpha=0.7)
            
            # Add sentiment labels
            for bar, value in zip(bars, df['Sentiment']):
                height = bar.get_height()
                ax1.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + 0.05 if height >= 0 else height - 0.1,
                    f'{value:.2f}',
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontweight='bold'
                )
                
            ax1.set_ylabel('Sentiment Score', fontsize=12)
            ax1.set_ylim(SENTIMENT_MIN, SENTIMENT_MAX)  # Use standardized scale limits
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.2)
            ax1.grid(axis='y', alpha=0.3)
            
            # Secondary axis for mention percentage
            ax2 = ax1.twinx()
            ax2.plot(df['Country'], df['Mention %'], 'ro-', linewidth=2, markersize=8)
            
            # Add mention percentage labels
            for i, value in enumerate(df['Mention %']):
                ax2.text(
                    i,
                    value + 2,
                    f'{value:.1f}%',
                    ha='center', va='bottom',
                    color='red',
                    fontweight='bold'
                )
                
            ax2.set_ylabel('Mention Percentage', fontsize=12, color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            
            plt.title(f'{theme_display} Combined Analysis - {category_display}', fontsize=14, pad=20)
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join(viz_dir, f'{category}_{theme}_combined_analysis.png'), dpi=300)
            plt.close()
            
    def _create_country_theme_charts(self, category, theme_data, viz_dir):
        """
        Create charts comparing themes for each country.
        
        Parameters:
        - category: Category name
        - theme_data: Theme data dictionary
        - viz_dir: Visualization directory
        """
        # Prepare data for each country
        country_theme_data = defaultdict(dict)
        
        # Collect theme data by country
        for theme, theme_results in theme_data.items():
            if 'by_country' not in theme_results:
                continue
                
            for country, data in theme_results['by_country'].items():
                theme_display = theme.replace('_', ' ').title()
                country_theme_data[country][theme_display] = data
                
        # Process each country
        for country, themes in country_theme_data.items():
            if not themes:
                continue
                
            # Gather data for chart
            theme_names = []
            sentiments = []
            mentions = []
            
            for theme, data in themes.items():
                theme_names.append(theme)
                sentiments.append(data['avg_sentiment'])
                mentions.append(data['mention_percentage'])
                
            # Create a DataFrame for easier manipulation
            df = pd.DataFrame({
                'Theme': theme_names,
                'Sentiment': sentiments,
                'Mention %': mentions
            })
            
            # Sort by sentiment
            df = df.sort_values('Sentiment', ascending=False)
            
            # Color by sentiment value using standard scale
            cmap, norm = get_standard_sentiment_cmap()
            colors = cmap(norm(df['Sentiment']))
            
            # Create sentiment bar chart
            plt.figure(figsize=(12, 6))
            bars = plt.bar(df['Theme'], df['Sentiment'], color=colors)
            
            # Add value labels
            for bar, value in zip(bars, df['Sentiment']):
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + 0.02 if height >= 0 else height - 0.08,
                    f'{value:.2f}',
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontweight='bold'
                )
                
            # Format for display
            country_display = country.title()
            category_display = category.title()
            
            plt.title(f'{category_display} Theme Sentiment - {country_display}', fontsize=14, pad=20)
            plt.ylabel('Sentiment Score', fontsize=12)
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.2)
            plt.grid(axis='y', alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            plt.ylim(SENTIMENT_MIN, SENTIMENT_MAX)  # Use standardized scale limits
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join(viz_dir, f'{category}_{country}_theme_sentiment.png'), dpi=300)
            plt.close()
            
            # Create combined chart
            plt.figure(figsize=(13, 6))
            
            # Primary axis for sentiment
            ax1 = plt.gca()
            bars = ax1.bar(df['Theme'], df['Sentiment'], color=colors, alpha=0.8)
            
            # Add sentiment labels
            for bar, value in zip(bars, df['Sentiment']):
                height = bar.get_height()
                ax1.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + 0.05 if height >= 0 else height - 0.1,
                    f'{value:.2f}',
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontweight='bold'
                )
                
            ax1.set_ylabel('Sentiment Score', fontsize=12)
            ax1.set_ylim(SENTIMENT_MIN, SENTIMENT_MAX)  # Use standardized scale limits
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.2)
            ax1.grid(axis='y', alpha=0.3)
            ax1.set_xticklabels(df['Theme'], rotation=45, ha='right')
            
            # Secondary axis for mention percentage
            ax2 = ax1.twinx()
            ax2.plot(range(len(df)), df['Mention %'], 'ro-', linewidth=2, markersize=8)
            
            # Add mention percentage labels
            for i, value in enumerate(df['Mention %']):
                ax2.text(
                    i,
                    value + 2,
                    f'{value:.1f}%',
                    ha='center', va='bottom',
                    color='red',
                    fontweight='bold'
                )
                
            ax2.set_ylabel('Mention Percentage', fontsize=12, color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            
            plt.title(f'{category_display} Theme Analysis - {country_display}', fontsize=14, pad=20)
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join(viz_dir, f'{category}_{country}_theme_combined.png'), dpi=300)
            plt.close()
            
    def create_heatmaps(self, results):
        """
        Create heatmaps comparing sentiment and mentions across countries and themes.
        
        Parameters:
        - results: Dictionary with theme analysis results
        """
        if not results or 'by_category' not in results:
            print("No results to visualize.")
            return
            
        print("Creating theme comparison heatmaps...")
        
        # Create visualizations directory
        viz_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Process each category
        for category, category_data in results['by_category'].items():
            if 'by_theme' not in category_data:
                continue
                
            # Create sentiment heatmap
            self._create_sentiment_heatmap(category, category_data['by_theme'], viz_dir)
            
            # Create mention heatmap
            self._create_mention_heatmap(category, category_data['by_theme'], viz_dir)
            
        print("Heatmap creation complete.")
        
    def _create_sentiment_heatmap(self, category, theme_data, viz_dir):
        """
        Create a heatmap of theme sentiment by country.
        
        Parameters:
        - category: Category name
        - theme_data: Theme data dictionary
        - viz_dir: Visualization directory
        """
        # Prepare data for heatmap
        heatmap_data = []
        
        # Collect theme sentiment data
        for theme, theme_results in theme_data.items():
            if 'by_country' not in theme_results:
                continue
                
            for country, data in theme_results['by_country'].items():
                theme_display = theme.replace('_', ' ').title()
                heatmap_data.append({
                    'Country': country.title(),
                    'Theme': theme_display,
                    'Sentiment': data['avg_sentiment']
                })
                
        if not heatmap_data:
            return
            
        # Create DataFrame for heatmap
        df = pd.DataFrame(heatmap_data)
        
        # Create pivot table
        pivot = df.pivot(index='Country', columns='Theme', values='Sentiment')
        
        # Create heatmap with standardized scale
        plt.figure(figsize=(12, 8))
        
        # Get standard sentiment colormap and normalization
        cmap, norm = get_standard_sentiment_cmap()
        
        # Create heatmap with fixed scale for consistent interpretation
        sns.heatmap(pivot, annot=True, cmap=cmap, norm=norm, fmt='.2f',
                   cbar_kws={'label': 'Sentiment Score'})
        
        # Format for display
        category_display = category.title()
        
        plt.title(f'{category_display} Theme Sentiment by Country', fontsize=16, pad=20)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(viz_dir, f'{category}_theme_sentiment_heatmap.png'), dpi=300)
        plt.close()
        
    def _create_mention_heatmap(self, category, theme_data, viz_dir):
        """
        Create a heatmap of theme mention percentage by country.
        
        Parameters:
        - category: Category name
        - theme_data: Theme data dictionary
        - viz_dir: Visualization directory
        """
        # Prepare data for heatmap
        heatmap_data = []
        
        # Collect theme mention data
        for theme, theme_results in theme_data.items():
            if 'by_country' not in theme_results:
                continue
                
            for country, data in theme_results['by_country'].items():
                theme_display = theme.replace('_', ' ').title()
                heatmap_data.append({
                    'Country': country.title(),
                    'Theme': theme_display,
                    'Mention %': data['mention_percentage']
                })
                
        if not heatmap_data:
            return
            
        # Create DataFrame for heatmap
        df = pd.DataFrame(heatmap_data)
        
        # Create pivot table
        pivot = df.pivot(index='Country', columns='Theme', values='Mention %')
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot, annot=True, cmap='YlGnBu', fmt='.1f')
        
        # Format for display
        category_display = category.title()
        
        plt.title(f'{category_display} Theme Mention Percentage by Country', fontsize=16, pad=20)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(viz_dir, f'{category}_theme_mention_heatmap.png'), dpi=300)
        plt.close()
        
    def run_comparisons(self):
        """
        Run the complete theme comparison analysis and create visualizations.
        """
        # Load data
        self.load_data()
        
        # Analyze themes
        results = self.analyze_themes()
        
        # Create visualizations
        self.create_visualizations(results)
        
        # Create heatmaps
        self.create_heatmaps(results)
        
        print(f"\nRegional theme comparison analysis complete. Results saved to {self.output_dir}")
        
def run_regional_theme_comparison():
    """
    Run the regional theme comparison analysis.
    """
    # Initialize analyzer
    analyzer = RegionalThemeComparison()
    
    # Run complete comparison
    analyzer.run_comparisons()
    
if __name__ == "__main__":
    run_regional_theme_comparison()