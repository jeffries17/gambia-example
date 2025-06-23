import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import json
import re

class ExpectationAnalyzer:
    """
    Analyzes the gap between visitor expectations and reality in tourism reviews.
    Focuses on identifying what surprised visitors (positively or negatively),
    what met expectations, and where there are gaps between expectations and experiences.
    """
    
    def __init__(self, output_dir='expectation_insights'):
        """
        Initialize the visitor expectations analyzer.
        
        Parameters:
        - output_dir: Directory to save expectations analysis outputs
        """
        self.output_dir = output_dir
        
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created visitor expectations insights directory: {output_dir}")
        
        # Define expectation-related keywords and patterns
        self.expectation_keywords = {
            'expectation_words': ['expect', 'expected', 'anticipate', 'anticipated', 'hope', 'hoped', 
                                 'thought', 'think', 'imagine', 'imagined', 'assume', 'assumed'],
            'positive_surprise': ['exceed', 'exceeded', 'better than', 'surprised', 'amazed', 'impressed', 
                                'wow', 'wonderful', 'exceptional', 'outstanding', 'blown away', 'incredible'],
            'negative_surprise': ['disappoint', 'disappointed', 'letdown', 'let down', 'not as', 'worse than', 
                                'not what', 'underwhelm', 'underwhelmed', 'not up to', 'below', 'failed to'],
            'reality_confirmation': ['as expected', 'met expectation', 'exactly what', 'just as', 'as advertised', 
                                    'as described', 'as promised', 'true to', 'lived up to']
        }
        
        # Define aspects for expectations analysis
        self.expectation_aspects = {
            'facilities': ['room', 'bathroom', 'pool', 'amenity', 'facility', 'property', 'hotel', 'condition', 
                         'building', 'accommodation', 'clean', 'cleanliness', 'wifi', 'internet'],
            'service': ['service', 'staff', 'friendly', 'attentive', 'helpful', 'professional', 'reception', 
                      'manager', 'waiter', 'waitress', 'hospitality', 'host', 'guide'],
            'food': ['food', 'breakfast', 'dinner', 'lunch', 'meal', 'restaurant', 'dining', 'cuisine', 
                   'dish', 'taste', 'delicious', 'menu', 'drink', 'chef', 'cook'],
            'location': ['location', 'view', 'scenery', 'beach', 'close', 'distance', 'walk', 'far', 
                       'central', 'remote', 'access', 'nearby', 'convenience', 'area'],
            'activities': ['activity', 'tour', 'excursion', 'experience', 'guide', 'adventure', 'swim', 
                        'snorkel', 'dive', 'boat', 'hike', 'culture', 'entertainment'],
            'value': ['price', 'value', 'money', 'worth', 'expensive', 'cheap', 'cost', 'affordable', 
                    'overpriced', 'reasonable', 'budget', 'payment', 'fee']
        }
    
    def detect_expectation_mentions(self, df):
        """
        Detect mentions of expectations and surprises in reviews.
        """
        print("Detecting expectation and surprise mentions in reviews...")
        
        # Make a copy of the DataFrame to avoid warnings
        df = df.copy()
        
        # For each expectation keyword category, create a flag
        for category, keywords in self.expectation_keywords.items():
            # Create a regex-safe pattern by escaping special characters
            safe_keywords = [re.escape(keyword) for keyword in keywords]
            pattern = '|'.join(safe_keywords)
            
            # Use word boundaries to match whole words only
            pattern = r'\b(' + pattern + r')\b'
            
            # Create the flag column
            df[f'mentions_{category}'] = df['text'].str.lower().str.contains(
                pattern, 
                regex=True, 
                na=False
            )
        
        # Create a general expectation mention flag
        mention_columns = [f'mentions_{cat}' for cat in self.expectation_keywords.keys()]
        df['mentions_expectations'] = df[mention_columns].any(axis=1).astype(int)
        
        # Calculate statistics
        expectation_mention_rate = df['mentions_expectations'].mean() * 100
        
        print(f"Expectation/surprise mentions found in {expectation_mention_rate:.1f}% of reviews")
        for category in self.expectation_keywords.keys():
            count = df[f'mentions_{category}'].sum()
            print(f"  {category.replace('_', ' ').title()}: {count}")
        
        return df
    
    def categorize_expectation_aspects(self, df):
        """
        Categorize which aspects are mentioned in expectation-related reviews.
        """
        print("\nCategorizing expectation-related aspects...")
        
        # Filter to reviews that mention expectations
        expectation_reviews = df[df['mentions_expectations'] == 1].copy()
        
        if len(expectation_reviews) == 0:
            print("No expectation-related reviews found for categorization")
            return df
        
        # For each aspect, create a flag using safe regex patterns
        for aspect, keywords in self.expectation_aspects.items():
            safe_keywords = [re.escape(keyword) for keyword in keywords]
            pattern = r'\b(' + '|'.join(safe_keywords) + r')\b'
            
            # Add the flag column
            expectation_reviews[f'expect_{aspect}'] = expectation_reviews['text'].str.lower().str.contains(
                pattern,
                regex=True,
                na=False
            )
        
        # Calculate aspect mention statistics
        aspect_counts = {}
        for aspect in self.expectation_aspects.keys():
            col = f'expect_{aspect}'
            count = expectation_reviews[col].sum()
            percentage = (count / len(expectation_reviews)) * 100
            aspect_counts[aspect] = (count, percentage)
        
        # Add the calculated columns back to the main dataframe
        aspect_columns = [f'expect_{aspect}' for aspect in self.expectation_aspects.keys()]
        df[aspect_columns] = 0  # Initialize columns
        df.loc[expectation_reviews.index, aspect_columns] = expectation_reviews[aspect_columns]
        
        # Print statistics
        for aspect, (count, percentage) in sorted(aspect_counts.items(), key=lambda x: x[1][0], reverse=True):
            print(f"  {aspect}: {count} mentions ({percentage:.1f}% of expectation reviews)")
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        aspects = [k for k, v in sorted(aspect_counts.items(), key=lambda x: x[1][0], reverse=True)]
        counts = [v[0] for k, v in sorted(aspect_counts.items(), key=lambda x: x[1][0], reverse=True)]
        
        plt.bar(aspects, counts, color='skyblue')
        plt.title('Expectation Mentions by Aspect')
        plt.xlabel('Aspect')
        plt.ylabel('Number of Mentions')
        plt.xticks(rotation=45, ha='right')
        
        # Adjust layout with specific margins
        plt.subplots_adjust(bottom=0.2)
        
        save_path = f'{self.output_dir}/expectation_mentions_by_aspect.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        print(f"Saved expectation mentions chart to {save_path}")
        
        return df
    
    def analyze_expectation_sentiment(self, df):
        """
        Analyze sentiment in expectation-related mentions.
        
        Parameters:
        - df: DataFrame with review data including expectation mentions
        
        Returns:
        - DataFrame with expectation sentiment analysis
        """
        print("\nAnalyzing sentiment in expectation-related mentions...")
        
        # Check for expectation mentions
        if 'mentions_expectations' not in df.columns or df['mentions_expectations'].sum() == 0:
            print("No expectation mentions found for sentiment analysis")
            return df
        
        # Filter to reviews that mention expectations
        expectation_reviews = df[df['mentions_expectations'] == 1]
        
        # Calculate average sentiment for expectation-mentioning reviews
        avg_expectation_sentiment = expectation_reviews['sentiment_score'].mean()
        print(f"Average sentiment in expectation-mentioning reviews: {avg_expectation_sentiment:.3f}")
        
        # Compare to overall sentiment
        overall_sentiment = df['sentiment_score'].mean()
        sentiment_diff = avg_expectation_sentiment - overall_sentiment
        
        if sentiment_diff > 0.05:
            print("Expectation-mentioning reviews are more positive than average")
        elif sentiment_diff < -0.05:
            print("Expectation-mentioning reviews are more negative than average")
        else:
            print("Expectation-mentioning reviews have similar sentiment to average")
        
        # Calculate sentiment for different expectation types
        sentiment_by_type = {}
        for category in ['positive_surprise', 'negative_surprise', 'reality_confirmation']:
            col = f'mentions_{category}'
            if expectation_reviews[col].sum() > 0:
                sentiment = expectation_reviews[expectation_reviews[col] == 1]['sentiment_score'].mean()
                sentiment_by_type[category] = sentiment
        
        if sentiment_by_type:
            print("\nSentiment by expectation type:")
            for category, sentiment in sentiment_by_type.items():
                print(f"  {category.replace('_', ' ')}: {sentiment:.3f}")
            
            # Create a bar chart of sentiment by expectation type
            categories = [cat.replace('_', ' ') for cat in sentiment_by_type.keys()]
            sentiments = list(sentiment_by_type.values())
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(categories, sentiments, color=['green' if s > 0 else 'red' for s in sentiments])
            plt.title('Sentiment by Expectation Type')
            plt.xlabel('Expectation Type')
            plt.ylabel('Average Sentiment Score (-1 to 1)')
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Add sentiment labels
            for bar, sentiment in zip(bars, sentiments):
                plt.text(bar.get_x() + bar.get_width()/2, 
                       sentiment + (0.02 if sentiment >= 0 else -0.08),
                       f'{sentiment:.2f}', ha='center', va='bottom' if sentiment >= 0 else 'top',
                       color='black')
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/sentiment_by_expectation_type.png')
            print(f"Saved expectation sentiment chart to {self.output_dir}/sentiment_by_expectation_type.png")
        
        # Calculate average sentiment for each aspect in expectation reviews
        sentiment_by_aspect = {}
        for aspect in self.expectation_aspects.keys():
            col = f'expect_{aspect}'
            if col in expectation_reviews.columns and expectation_reviews[col].sum() > 0:
                aspect_sentiment = expectation_reviews[expectation_reviews[col] == 1]['sentiment_score'].mean()
                sentiment_by_aspect[aspect] = aspect_sentiment
        
        if sentiment_by_aspect:
            # Sort by sentiment
            sorted_sentiments = sorted(sentiment_by_aspect.items(), key=lambda x: x[1], reverse=True)
            
            print("\nExpectation sentiment by aspect:")
            for aspect, sentiment in sorted_sentiments:
                print(f"  {aspect}: {sentiment:.3f}")
            
            # Create a bar chart of sentiment by aspect
            aspects = [asp for asp, _ in sorted_sentiments]
            sentiments = [sent for _, sent in sorted_sentiments]
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(aspects, sentiments, color=['green' if s > 0 else 'red' for s in sentiments])
            plt.title('Expectation Sentiment by Aspect')
            plt.xlabel('Aspect')
            plt.ylabel('Average Sentiment Score (-1 to 1)')
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Add sentiment labels
            for bar, sentiment in zip(bars, sentiments):
                plt.text(bar.get_x() + bar.get_width()/2, 
                       sentiment + (0.02 if sentiment >= 0 else -0.08),
                       f'{sentiment:.2f}', ha='center', va='bottom' if sentiment >= 0 else 'top',
                       color='black')
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/expectation_sentiment_by_aspect.png')
            print(f"Saved expectation aspect sentiment chart to {self.output_dir}/expectation_sentiment_by_aspect.png")
        
        return df
    
    def extract_expectation_phrases(self, df, min_count=2):
        """
        Extract common phrases from expectation-mentioning reviews.
        
        Parameters:
        - df: DataFrame with review data including expectation mentions
        - min_count: Minimum occurrence count for phrases
        
        Returns:
        - Dictionary of phrases by expectation category
        """
        print("\nExtracting key phrases from expectation-related reviews...")
        
        # Check for expectation mentions
        if 'mentions_expectations' not in df.columns or df['mentions_expectations'].sum() == 0:
            print("No expectation mentions found for phrase extraction")
            return {}
        
        # Filter to reviews that mention expectations
        expectation_reviews = df[df['mentions_expectations'] == 1]
        
        # Separate into different expectation categories
        positive_surprise = expectation_reviews[expectation_reviews['mentions_positive_surprise'] == 1]
        negative_surprise = expectation_reviews[expectation_reviews['mentions_negative_surprise'] == 1]
        reality_confirm = expectation_reviews[expectation_reviews['mentions_reality_confirmation'] == 1]
        
        expectation_phrases = {}
        
        # Extract phrases from positive surprise reviews
        if len(positive_surprise) > 0 and 'processed_text' in positive_surprise.columns:
            pos_text = ' '.join(positive_surprise['processed_text'].fillna(''))
            pos_words = pos_text.split()
            word_counts = Counter(pos_words)
            
            # Filter to words appearing at least min_count times
            common_words = {word: count for word, count in word_counts.items() if count >= min_count}
            expectation_phrases['positive_surprise'] = sorted(common_words.items(), key=lambda x: x[1], reverse=True)
            
            print("\nCommon words in positive surprise reviews:")
            for word, count in sorted(common_words.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {word}: {count}")
                
            # Create word cloud for positive surprise phrases
            if common_words:
                wordcloud = WordCloud(width=800, height=400, background_color='white', 
                                   max_words=100, contour_width=3, contour_color='steelblue')
                wordcloud.generate_from_frequencies(common_words)
                
                plt.figure(figsize=(10, 6))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title('Common Words in Positive Surprise Reviews')
                plt.tight_layout()
                plt.savefig(f'{self.output_dir}/positive_surprise_wordcloud.png')
                print(f"Saved positive surprise wordcloud to {self.output_dir}/positive_surprise_wordcloud.png")
        
        # Extract phrases from negative surprise reviews
        if len(negative_surprise) > 0 and 'processed_text' in negative_surprise.columns:
            neg_text = ' '.join(negative_surprise['processed_text'].fillna(''))
            neg_words = neg_text.split()
            word_counts = Counter(neg_words)
            
            # Filter to words appearing at least min_count times
            common_words = {word: count for word, count in word_counts.items() if count >= min_count}
            expectation_phrases['negative_surprise'] = sorted(common_words.items(), key=lambda x: x[1], reverse=True)
            
            print("\nCommon words in negative surprise reviews:")
            for word, count in sorted(common_words.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {word}: {count}")
                
            # Create word cloud for negative surprise phrases
            if common_words:
                wordcloud = WordCloud(width=800, height=400, background_color='white', 
                                   max_words=100, contour_width=3, contour_color='darkred')
                wordcloud.generate_from_frequencies(common_words)
                
                plt.figure(figsize=(10, 6))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title('Common Words in Negative Surprise Reviews')
                plt.tight_layout()
                plt.savefig(f'{self.output_dir}/negative_surprise_wordcloud.png')
                print(f"Saved negative surprise wordcloud to {self.output_dir}/negative_surprise_wordcloud.png")
        
        return expectation_phrases
    
    def analyze_expectation_gaps(self, df):
        """
        Identify gaps between visitor expectations and reality.
        
        Parameters:
        - df: DataFrame with review data including expectation mentions
        
        Returns:
        - DataFrame with expectation gap analysis
        """
        print("\nAnalyzing gaps between expectations and reality...")
        
        # Check for expectation mentions
        if 'mentions_expectations' not in df.columns or df['mentions_expectations'].sum() == 0:
            print("No expectation mentions found for gap analysis")
            return df
        
        # Calculate sentiment for different expectation aspects
        gap_analysis = {}
        
        for aspect in self.expectation_aspects.keys():
            aspect_col = f'expect_{aspect}'
            
            # Skip if aspect column doesn't exist
            if aspect_col not in df.columns:
                continue
                
            # Filter to reviews that mention this aspect and expectations
            aspect_reviews = df[(df[aspect_col] == 1) & (df['mentions_expectations'] == 1)]
            
            if len(aspect_reviews) < 5:  # Need enough reviews for meaningful analysis
                continue
                
            # Calculate sentiment for different expectation categories
            pos_surprise = aspect_reviews[aspect_reviews['mentions_positive_surprise'] == 1]
            neg_surprise = aspect_reviews[aspect_reviews['mentions_negative_surprise'] == 1]
            
            # Only include in gap analysis if we have both positive and negative surprises
            if len(pos_surprise) > 0 and len(neg_surprise) > 0:
                pos_sentiment = pos_surprise['sentiment_score'].mean()
                neg_sentiment = neg_surprise['sentiment_score'].mean()
                
                # Calculate expectation gap metric
                # Positive values mean more positive than negative surprises
                surprise_ratio = len(pos_surprise) / max(1, len(neg_surprise))
                sentiment_gap = pos_sentiment - neg_sentiment
                
                gap_analysis[aspect] = {
                    'positive_count': len(pos_surprise),
                    'negative_count': len(neg_surprise),
                    'surprise_ratio': surprise_ratio,
                    'sentiment_gap': sentiment_gap,
                    'total_mentions': len(aspect_reviews)
                }
        
        if gap_analysis:
            # Sort by sentiment gap
            sorted_gaps = sorted(gap_analysis.items(), key=lambda x: x[1]['sentiment_gap'])
            
            print("\nExpectation gaps by aspect:")
            for aspect, metrics in sorted_gaps:
                print(f"  {aspect}:")
                print(f"    Positive surprises: {metrics['positive_count']}")
                print(f"    Negative surprises: {metrics['negative_count']}")
                print(f"    Positive/Negative ratio: {metrics['surprise_ratio']:.2f}")
                print(f"    Sentiment gap: {metrics['sentiment_gap']:.2f}")
            
            # Create visualization of expectation gaps
            aspects = [asp for asp, _ in sorted_gaps]
            sentiment_gaps = [m['sentiment_gap'] for _, m in sorted_gaps]
            surprise_ratios = [m['surprise_ratio'] for _, m in sorted_gaps]
            
            # Set up a more complex figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot sentiment gaps
            bars1 = ax1.barh(aspects, sentiment_gaps, color=['green' if s > 0 else 'red' for s in sentiment_gaps])
            ax1.set_title('Sentiment Gap by Aspect (Positive vs Negative Surprises)')
            ax1.set_xlabel('Sentiment Gap (Higher is Better)')
            ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            
            # Add labels
            for bar, gap in zip(bars1, sentiment_gaps):
                ax1.text(gap + (0.02 if gap >= 0 else -0.08), 
                       bar.get_y() + bar.get_height()/2,
                       f'{gap:.2f}', va='center',
                       ha='left' if gap >= 0 else 'right',
                       color='black')
            
            # Plot surprise ratios
            bars2 = ax2.barh(aspects, surprise_ratios, color='skyblue')
            ax2.set_title('Positive-to-Negative Surprise Ratio by Aspect')
            ax2.set_xlabel('Ratio (Higher = More Positive Surprises)')
            ax2.axvline(x=1, color='black', linestyle='-', alpha=0.3)
            
            # Add labels
            for bar, ratio in zip(bars2, surprise_ratios):
                ax2.text(ratio + 0.05, 
                       bar.get_y() + bar.get_height()/2,
                       f'{ratio:.2f}', va='center',
                       color='black')
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/expectation_gaps.png')
            print(f"Saved expectation gap chart to {self.output_dir}/expectation_gaps.png")
            
            # Save gap analysis to file
            gap_df = pd.DataFrame({
                aspect: metrics for aspect, metrics in gap_analysis.items()
            }).T
            
            gap_df.to_csv(f'{self.output_dir}/expectation_gap_analysis.csv')
        
        return df
    
    def generate_expectation_recommendations(self, df):
        """
        Generate recommendations based on expectation-reality gap analysis.
        
        Parameters:
        - df: DataFrame with expectation analysis results
        
        Returns:
        - Dictionary with expectation-related recommendations
        """
        print("\nGenerating expectation-related recommendations...")
        
        recommendations = {
            "expectation_setting": [],
            "pleasant_surprises": [],
            "expectation_gaps": [],
            "marketing_focus": []
        }
        
        # Check for expectation mentions
        if 'mentions_expectations' not in df.columns or df['mentions_expectations'].sum() == 0:
            print("No expectation mentions found for generating recommendations")
            return recommendations
        
        # Calculate basic metrics
        positive_count = df['mentions_positive_surprise'].sum()
        negative_count = df['mentions_negative_surprise'].sum()
        confirmation_count = df['mentions_reality_confirmation'].sum()
        
        total_expectation_mentions = positive_count + negative_count + confirmation_count
        
        if total_expectation_mentions > 0:
            positive_rate = positive_count / total_expectation_mentions
            negative_rate = negative_count / total_expectation_mentions
            confirmation_rate = confirmation_count / total_expectation_mentions
            
            print(f"Positive surprises: {positive_rate*100:.1f}%")
            print(f"Negative surprises: {negative_rate*100:.1f}%")
            print(f"Reality confirmations: {confirmation_rate*100:.1f}%")
            
            # Overall expectation setting recommendations
            if negative_rate > 0.3:
                recommendations["expectation_setting"].append(
                    f"Adjust expectations in marketing materials as {negative_rate*100:.1f}% "
                    "of expectation mentions are negative surprises"
                )
            
            if positive_rate > 0.5:
                recommendations["marketing_focus"].append(
                    f"Leverage positive surprises in marketing as {positive_rate*100:.1f}% "
                    "of expectation mentions indicate visitors are pleasantly surprised"
                )
        
        # Analyze expectation sentiment by aspect
        aspect_sentiments = {}
        for aspect in self.expectation_aspects.keys():
            aspect_col = f'expect_{aspect}'
            
            # Skip if aspect column doesn't exist
            if aspect_col not in df.columns:
                continue
                
            # Filter to reviews that mention this aspect and expectations
            aspect_reviews = df[(df[aspect_col] == 1) & (df['mentions_expectations'] == 1)]
            
            if len(aspect_reviews) < 5:  # Need enough reviews for meaningful analysis
                continue
                
            # Calculate sentiment for different expectation categories
            aspect_expectation_sentiment = aspect_reviews['sentiment_score'].mean()
            aspect_sentiments[aspect] = aspect_expectation_sentiment
            
            # Check for positive surprises
            pos_surprise = aspect_reviews[aspect_reviews['mentions_positive_surprise'] == 1]
            if len(pos_surprise) >= 5:
                pos_sentiment = pos_surprise['sentiment_score'].mean()
                
                if pos_sentiment > 0.3:  # Very positive
                    recommendations["pleasant_surprises"].append(
                        f"Highlight unexpected {aspect} delights in marketing as this aspect "
                        f"generates positive surprises (sentiment: {pos_sentiment:.2f})"
                    )
            
            # Check for negative surprises
            neg_surprise = aspect_reviews[aspect_reviews['mentions_negative_surprise'] == 1]
            if len(neg_surprise) >= 5:
                neg_sentiment = neg_surprise['sentiment_score'].mean()
                
                if neg_sentiment < -0.1:  # Negative
                    recommendations["expectation_gaps"].append(
                        f"Address {aspect} expectation gaps which generate negative surprises "
                        f"(sentiment: {neg_sentiment:.2f})"
                    )
        
        # Create recommendations based on highest and lowest aspect sentiments
        if aspect_sentiments:
            best_aspect = max(aspect_sentiments.items(), key=lambda x: x[1])
            worst_aspect = min(aspect_sentiments.items(), key=lambda x: x[1])
            
            if best_aspect[1] > 0.2:
                recommendations["marketing_focus"].append(
                    f"Emphasize {best_aspect[0]} in marketing materials as expectations for this "
                    f"aspect are being met or exceeded (sentiment: {best_aspect[1]:.2f})"
                )
            
            if worst_aspect[1] < 0:
                recommendations["expectation_setting"].append(
                    f"Adjust expectations for {worst_aspect[0]} in marketing materials "
                    f"to address negative sentiment (sentiment: {worst_aspect[1]:.2f})"
                )
        
        # Save recommendations to file
        with open(f'{self.output_dir}/expectation_recommendations.json', 'w', encoding='utf-8') as f:
            json.dump(recommendations, f, indent=2)
        
        # Print key recommendations
        print("\nKey expectation-related recommendations:")
        for category, recs in recommendations.items():
            if recs:
                print(f"\n{category.replace('_', ' ').title()}:")
                for rec in recs:
                    print(f"  - {rec}")
        
        return recommendations
    
    def run_analysis(self, df):
        """
        Run the complete visitor expectations analysis.
        
        Parameters:
        - df: DataFrame with review data
        
        Returns:
        - DataFrame with visitor expectations analysis results
        """
        print("\n=== Running Visitor Expectations Analysis ===")
        
        if df is None or len(df) == 0:
            print("No data available for expectations analysis")
            return None
        
        # Detect expectation and surprise mentions
        df = self.detect_expectation_mentions(df)
        
        # Categorize expectation aspects
        df = self.categorize_expectation_aspects(df)
        
        # Analyze expectation sentiment
        df = self.analyze_expectation_sentiment(df)
        
        # Extract expectation phrases
        self.extract_expectation_phrases(df)
        
        # Analyze expectation gaps
        df = self.analyze_expectation_gaps(df)
        
        # Generate expectation recommendations
        self.generate_expectation_recommendations(df)
        
        print("\nVisitor expectations analysis complete.")
        return df