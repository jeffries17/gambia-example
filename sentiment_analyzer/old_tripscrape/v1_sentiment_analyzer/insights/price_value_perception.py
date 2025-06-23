import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import json
import re

class PriceValueAnalyzer:
    """
    Analyzes price-value perceptions in tourism reviews.
    Focuses on understanding what experiences visitors consider worth the money,
    overpriced, or offering good value.
    """
    
    def __init__(self, output_dir='price_value_insights'):
        """
        Initialize the price-value perception analyzer.
        
        Parameters:
        - output_dir: Directory to save price-value analysis outputs
        """
        self.output_dir = output_dir
        
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created price-value insights directory: {output_dir}")
        
        # Define price-related keywords and patterns
        self.price_keywords = {
            'positive_value': ['worth', 'reasonable', 'fair', 'good value', 'great value', 'value for money', 
                              'affordable', 'inexpensive', 'bargain', 'cheap', 'economical', 'budget-friendly'],
            'negative_value': ['overpriced', 'expensive', 'costly', 'pricey', 'not worth', 'ripoff', 'rip off', 
                             'high price', 'too much', 'steep price', 'over priced', 'cost too much'],
            'price_mention': ['price', 'cost', 'money', 'paid', 'spend', 'spent', 'pay', 'dollar', 'payment',
                            'fee', 'charge', 'budget', 'pricing', 'rate']
        }
        
        # Define experience categories for price-value analysis
        self.experience_categories = {
            'accommodation': ['hotel', 'resort', 'room', 'accommodation', 'stay', 'night', 'bed', 'sleep'],
            'dining': ['restaurant', 'food', 'meal', 'dinner', 'lunch', 'breakfast', 'eat', 'dining'],
            'activities': ['tour', 'activity', 'excursion', 'guide', 'experience', 'trip', 'adventure', 'visit'],
            'transport': ['transport', 'transfer', 'taxi', 'shuttle', 'car', 'bus', 'flight', 'travel'],
            'shopping': ['shop', 'store', 'buy', 'purchase', 'souvenir', 'market', 'merchandise', 'price']
        }
    
    def detect_price_mentions(self, df):
        """
        Detect mentions of prices and value perceptions in reviews.
        """
        print("Detecting price and value mentions in reviews...")
        
        # Make a copy of the DataFrame to avoid warnings
        df = df.copy()
        
        # For each price keyword category, create a flag with safe regex
        for category, keywords in self.price_keywords.items():
            # Create regex-safe keywords
            safe_keywords = [re.escape(keyword) for keyword in keywords]
            pattern = r'\b(' + '|'.join(safe_keywords) + r')\b'
            
            # Create the flag column
            df[f'mentions_{category}'] = df['text'].str.lower().str.contains(
                pattern,
                regex=True,
                na=False
            ).astype(int)
        
        # Create a general price mention flag
        mention_columns = [
            'mentions_positive_value',
            'mentions_negative_value',
            'mentions_price_mention'
        ]
        df['mentions_price'] = df[mention_columns].any(axis=1).astype(int)
        
        # Calculate statistics
        price_mention_rate = df['mentions_price'].mean() * 100
        
        print(f"Price/value mentions found in {price_mention_rate:.1f}% of reviews")
        print(f"  Positive value mentions: {df['mentions_positive_value'].sum()}")
        print(f"  Negative value mentions: {df['mentions_negative_value'].sum()}")
        print(f"  General price mentions: {df['mentions_price_mention'].sum()}")
        
        return df
    
    def categorize_price_experiences(self, df):
        """
        Categorize which aspects of the experience are mentioned in price-related reviews.
        """
        print("\nCategorizing price-related experiences...")
        
        # Filter to reviews that mention price
        price_reviews = df[df['mentions_price'] == 1].copy()
        
        if len(price_reviews) == 0:
            print("No price-related reviews found for categorization")
            return df
        
        # For each experience category, create a flag using safe regex patterns
        for category, keywords in self.experience_categories.items():
            safe_keywords = [re.escape(keyword) for keyword in keywords]
            pattern = r'\b(' + '|'.join(safe_keywords) + r')\b'
            
            price_reviews[f'price_{category}'] = price_reviews['text'].str.lower().str.contains(
                pattern,
                regex=True,
                na=False
            ).astype(int)
        
        # Calculate experience mention statistics
        experience_counts = {}
        for category in self.experience_categories.keys():
            col = f'price_{category}'
            count = price_reviews[col].sum()
            percentage = (count / len(price_reviews)) * 100
            experience_counts[category] = (count, percentage)
        
        # Sort experiences by count
        sorted_experiences = sorted(experience_counts.items(), key=lambda x: x[1][0], reverse=True)
        
        # Print statistics
        print("Price mentions by experience category:")
        for category, (count, percentage) in sorted_experiences:
            print(f"  {category}: {count} mentions ({percentage:.1f}% of price reviews)")
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        categories = [cat.replace('_', ' ').title() for cat, _ in sorted_experiences]
        counts = [count for _, (count, _) in sorted_experiences]
        
        bars = plt.bar(categories, counts, color='skyblue')
        plt.title('Price Mentions by Experience Category')
        plt.xlabel('Experience Category')
        plt.ylabel('Number of Mentions')
        plt.xticks(rotation=45, ha='right')
        
        # Add count labels
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom')
        
        plt.subplots_adjust(bottom=0.2)
        
        save_path = f'{self.output_dir}/price_mentions_by_category.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        print(f"Saved price mentions chart to {save_path}")
        
        # Add the calculated columns back to the main DataFrame
        category_columns = [f'price_{category}' for category in self.experience_categories.keys()]
        df[category_columns] = 0  # Initialize columns
        df.loc[price_reviews.index, category_columns] = price_reviews[category_columns]
        
        return df
    
    def analyze_value_sentiment(self, df):
        """
        Analyze sentiment in price-value mentions.
        
        Parameters:
        - df: DataFrame with review data including price mentions
        
        Returns:
        - DataFrame with value sentiment analysis
        """
        print("\nAnalyzing sentiment in price-value mentions...")
        
        # Check for price mentions
        if 'mentions_price' not in df.columns or df['mentions_price'].sum() == 0:
            print("No price mentions found for sentiment analysis")
            return df
        
        # Filter to reviews that mention price
        price_reviews = df[df['mentions_price'] == 1]
        
        # Calculate average sentiment for price-mentioning reviews
        avg_price_sentiment = price_reviews['sentiment_score'].mean()
        print(f"Average sentiment in price-mentioning reviews: {avg_price_sentiment:.3f}")
        
        # Compare to overall sentiment
        overall_sentiment = df['sentiment_score'].mean()
        sentiment_diff = avg_price_sentiment - overall_sentiment
        
        if sentiment_diff > 0.05:
            print("Price-mentioning reviews are more positive than average")
        elif sentiment_diff < -0.05:
            print("Price-mentioning reviews are more negative than average")
        else:
            print("Price-mentioning reviews have similar sentiment to average")
        
        # Calculate sentiment for positive and negative value mentions
        if df['mentions_positive_value'].sum() > 0:
            positive_value_sentiment = df[df['mentions_positive_value'] == 1]['sentiment_score'].mean()
            print(f"Sentiment in positive value mentions: {positive_value_sentiment:.3f}")
        
        if df['mentions_negative_value'].sum() > 0:
            negative_value_sentiment = df[df['mentions_negative_value'] == 1]['sentiment_score'].mean()
            print(f"Sentiment in negative value mentions: {negative_value_sentiment:.3f}")
        
        # Calculate average sentiment for each experience category in price reviews
        sentiment_by_category = {}
        for category in self.experience_categories.keys():
            col = f'price_{category}'
            if col in price_reviews.columns and price_reviews[col].sum() > 0:
                category_sentiment = price_reviews[price_reviews[col] == 1]['sentiment_score'].mean()
                sentiment_by_category[category] = category_sentiment
        
        if sentiment_by_category:
            # Sort by sentiment
            sorted_sentiments = sorted(sentiment_by_category.items(), key=lambda x: x[1], reverse=True)
            
            print("\nValue sentiment by experience category:")
            for category, sentiment in sorted_sentiments:
                print(f"  {category}: {sentiment:.3f}")
            
            # Create a bar chart of sentiment by category
            categories = [cat for cat, _ in sorted_sentiments]
            sentiments = [sent for _, sent in sorted_sentiments]
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(categories, sentiments, color=['green' if s > 0 else 'red' for s in sentiments])
            plt.title('Value Sentiment by Experience Category')
            plt.xlabel('Experience Category')
            plt.ylabel('Average Sentiment Score (-1 to 1)')
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Add sentiment labels
            for bar, sentiment in zip(bars, sentiments):
                plt.text(bar.get_x() + bar.get_width()/2, 
                       sentiment + (0.02 if sentiment >= 0 else -0.08),
                       f'{sentiment:.2f}', ha='center', va='bottom' if sentiment >= 0 else 'top',
                       color='black')
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/value_sentiment_by_category.png')
            print(f"Saved value sentiment chart to {self.output_dir}/value_sentiment_by_category.png")
        
        return df
    
    def extract_price_phrases(self, df, min_count=2):
        """
        Extract common phrases from price-mentioning reviews.
        
        Parameters:
        - df: DataFrame with review data including price mentions
        - min_count: Minimum occurrence count for phrases
        
        Returns:
        - Dictionary of phrases by sentiment category
        """
        print("\nExtracting key phrases from price-value reviews...")
        
        # Check for price mentions
        if 'mentions_price' not in df.columns or df['mentions_price'].sum() == 0:
            print("No price mentions found for phrase extraction")
            return {}
        
        # Filter to reviews that mention price
        price_reviews = df[df['mentions_price'] == 1]
        
        # Separate into positive and negative value reviews
        positive_value = price_reviews[price_reviews['mentions_positive_value'] == 1]
        negative_value = price_reviews[price_reviews['mentions_negative_value'] == 1]
        general_price = price_reviews[
            (price_reviews['mentions_price_mention'] == 1) & 
            (price_reviews['mentions_positive_value'] == 0) & 
            (price_reviews['mentions_negative_value'] == 0)
        ]
        
        value_phrases = {}
        
        # Extract phrases from positive value reviews
        if len(positive_value) > 0 and 'processed_text' in positive_value.columns:
            pos_text = ' '.join(positive_value['processed_text'].fillna(''))
            pos_words = pos_text.split()
            word_counts = Counter(pos_words)
            
            # Filter to words appearing at least min_count times
            common_words = {word: count for word, count in word_counts.items() if count >= min_count}
            value_phrases['positive_value'] = sorted(common_words.items(), key=lambda x: x[1], reverse=True)
            
            print("\nCommon words in positive value reviews:")
            for word, count in sorted(common_words.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {word}: {count}")
                
            # Create word cloud for positive value phrases
            if common_words:
                wordcloud = WordCloud(width=800, height=400, background_color='white', 
                                   max_words=100, contour_width=3, contour_color='steelblue')
                wordcloud.generate_from_frequencies(common_words)
                
                plt.figure(figsize=(10, 6))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title('Common Words in Positive Value Reviews')
                plt.tight_layout()
                plt.savefig(f'{self.output_dir}/positive_value_wordcloud.png')
                print(f"Saved positive value wordcloud to {self.output_dir}/positive_value_wordcloud.png")
        
        # Extract phrases from negative value reviews
        if len(negative_value) > 0 and 'processed_text' in negative_value.columns:
            neg_text = ' '.join(negative_value['processed_text'].fillna(''))
            neg_words = neg_text.split()
            word_counts = Counter(neg_words)
            
            # Filter to words appearing at least min_count times
            common_words = {word: count for word, count in word_counts.items() if count >= min_count}
            value_phrases['negative_value'] = sorted(common_words.items(), key=lambda x: x[1], reverse=True)
            
            print("\nCommon words in negative value reviews:")
            for word, count in sorted(common_words.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {word}: {count}")
                
            # Create word cloud for negative value phrases
            if common_words:
                wordcloud = WordCloud(width=800, height=400, background_color='white', 
                                   max_words=100, contour_width=3, contour_color='darkred')
                wordcloud.generate_from_frequencies(common_words)
                
                plt.figure(figsize=(10, 6))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title('Common Words in Negative Value Reviews')
                plt.tight_layout()
                plt.savefig(f'{self.output_dir}/negative_value_wordcloud.png')
                print(f"Saved negative value wordcloud to {self.output_dir}/negative_value_wordcloud.png")
        
        return value_phrases
    
    def analyze_price_patterns(self, df):
        """
        Analyze patterns in price-value perceptions across segments.
        """
        print("\nAnalyzing patterns in price-value perceptions...")
        
        if 'mentions_price' not in df.columns or df['mentions_price'].sum() == 0:
            print("No price mentions found for pattern analysis")
            return df
        
        # Get all price-mentioning reviews
        price_reviews = df[df['mentions_price'] == 1]
        
        # Analyze by trip type if available
        if 'trip_type_standard' in df.columns:
            trip_price_data = []
            
            for trip_type in df['trip_type_standard'].unique():
                if pd.notna(trip_type):
                    trip_df = df[df['trip_type_standard'] == trip_type]
                    if len(trip_df) >= 5:  # Only analyze with enough data
                        price_rate = trip_df['mentions_price'].mean() * 100
                        pos_rate = trip_df['mentions_positive_value'].mean() * 100
                        neg_rate = trip_df['mentions_negative_value'].mean() * 100
                        
                        trip_price_data.append({
                            'trip_type': trip_type,
                            'price_rate': price_rate,
                            'positive_rate': pos_rate,
                            'negative_rate': neg_rate
                        })
            
            if trip_price_data:
                trip_price_df = pd.DataFrame(trip_price_data)
                print("\nPrice mention rate by trip type:")
                for _, row in trip_price_df.iterrows():
                    print(f"  {row['trip_type']}: {row['price_rate']:.1f}%")
                
                # Calculate positive-to-negative ratio
                print("\nPositive-to-negative value mention ratio by trip type:")
                for _, row in trip_price_df.iterrows():
                    if row['negative_rate'] > 0:
                        ratio = row['positive_rate'] / row['negative_rate']
                        print(f"  {row['trip_type']}: {ratio:.2f}")
                    elif row['positive_rate'] > 0:
                        print(f"  {row['trip_type']}: Only positive mentions")
                    else:
                        print(f"  {row['trip_type']}: 1.00")
                
                # Create visualization
                plt.figure(figsize=(12, 6))
                trips = trip_price_df['trip_type']
                pos_rates = trip_price_df['positive_rate']
                neg_rates = trip_price_df['negative_rate']
                
                x = np.arange(len(trips))
                width = 0.35
                
                plt.bar(x - width/2, pos_rates, width, label='Positive', color='green', alpha=0.7)
                plt.bar(x + width/2, neg_rates, width, label='Negative', color='red', alpha=0.7)
                
                plt.title('Value Mentions by Trip Type')
                plt.xlabel('Trip Type')
                plt.ylabel('Mention Rate (%)')
                plt.xticks(x, trips, rotation=45, ha='right')
                plt.legend()
                
                plt.subplots_adjust(bottom=0.2)
                
                save_path = f'{self.output_dir}/value_mentions_by_trip_type.png'
                plt.savefig(save_path, bbox_inches='tight')
                plt.close()
                
                print(f"Saved value mentions chart to {save_path}")
        
        return df
    
    def generate_price_value_recommendations(self, df):
        """
        Generate recommendations based on price-value perception analysis.
        
        Parameters:
        - df: DataFrame with price-value analysis results
        
        Returns:
        - Dictionary with price-value recommendations
        """
        print("\nGenerating price-value perception recommendations...")
        
        recommendations = {
            "pricing_strategy": [],
            "value_communication": [],
            "experience_improvements": [],
            "segment_targeting": []
        }
        
        # Check for price mentions
        if 'mentions_price' not in df.columns or df['mentions_price'].sum() == 0:
            print("No price mentions found for generating recommendations")
            return recommendations
        
        # Filter to reviews that mention price
        price_reviews = df[df['mentions_price'] == 1]
        
        # Overall value sentiment recommendations
        avg_price_sentiment = price_reviews['sentiment_score'].mean()
        
        if avg_price_sentiment < -0.1:
            recommendations["pricing_strategy"].append(
                f"Address overall value perception issues as price-mentioning reviews "
                f"have negative sentiment ({avg_price_sentiment:.2f})"
            )
        elif avg_price_sentiment > 0.2:
            recommendations["value_communication"].append(
                f"Leverage positive value perceptions in marketing as price-mentioning "
                f"reviews have strong positive sentiment ({avg_price_sentiment:.2f})"
            )
        
        # Analyze sentiment for each experience category in price reviews
        sentiment_by_category = {}
        for category in self.experience_categories.keys():
            col = f'price_{category}'
            if col in price_reviews.columns and price_reviews[col].sum() >= 5:
                category_sentiment = price_reviews[price_reviews[col] == 1]['sentiment_score'].mean()
                sentiment_by_category[category] = category_sentiment
        
        if sentiment_by_category:
            # Identify categories with negative and positive sentiment
            negative_categories = {k: v for k, v in sentiment_by_category.items() if v < -0.1}
            positive_categories = {k: v for k, v in sentiment_by_category.items() if v > 0.2}
            
            for category, sentiment in negative_categories.items():
                recommendations["experience_improvements"].append(
                    f"Improve price-value perception for {category} experiences "
                    f"which have negative sentiment ({sentiment:.2f})"
                )
            
            for category, sentiment in positive_categories.items():
                recommendations["value_communication"].append(
                    f"Highlight value proposition of {category} experiences "
                    f"which have positive sentiment ({sentiment:.2f})"
                )
        
        # Analyze by trip type
        if 'trip_type_standard' in df.columns:
            # Calculate positive and negative value mentions by trip type
            trip_types = df['trip_type_standard'].unique()
            
            for trip in trip_types:
                if pd.notna(trip) and trip != 'unknown':
                    trip_df = df[df['trip_type_standard'] == trip]
                    
                    if len(trip_df) >= 10:  # Only analyze with enough data
                        pos_rate = trip_df['mentions_positive_value'].mean()
                        neg_rate = trip_df['mentions_negative_value'].mean()
                        
                        if neg_rate > 0.1 and neg_rate > pos_rate:
                            recommendations["segment_targeting"].append(
                                f"Address value perception concerns for {trip} travelers "
                                f"who mention negative value at a high rate ({neg_rate*100:.1f}%)"
                            )
                        elif pos_rate > 0.1 and pos_rate > neg_rate:
                            recommendations["segment_targeting"].append(
                                f"Target {trip} travelers in marketing campaigns "
                                f"as they mention positive value at a high rate ({pos_rate*100:.1f}%)"
                            )
        
        # Analyze by theme
        theme_cols = [col for col in df.columns if col.startswith('theme_')]
        
        if theme_cols:
            for col in theme_cols:
                theme = col.replace('theme_', '').replace('_', ' ')
                theme_df = df[df[col] == 1]
                
                if len(theme_df) >= 10:  # Only analyze with enough data
                    price_rate = theme_df['mentions_price'].mean()
                    
                    if price_rate > 0.2:  # If price is frequently mentioned
                        theme_sentiment = theme_df[theme_df['mentions_price'] == 1]['sentiment_score'].mean()
                        
                        if theme_sentiment < -0.1:
                            recommendations["pricing_strategy"].append(
                                f"Review pricing for {theme} experiences where price is "
                                f"frequently mentioned with negative sentiment ({theme_sentiment:.2f})"
                            )
                        elif theme_sentiment > 0.2:
                            recommendations["value_communication"].append(
                                f"Emphasize value of {theme} experiences in marketing "
                                f"which have positive price-value sentiment ({theme_sentiment:.2f})"
                            )
        
        # Save recommendations to file
        with open(f'{self.output_dir}/price_value_recommendations.json', 'w', encoding='utf-8') as f:
            json.dump(recommendations, f, indent=2)
        
        # Print key recommendations
        print("\nKey price-value recommendations:")
        for category, recs in recommendations.items():
            if recs:
                print(f"\n{category.replace('_', ' ').title()}:")
                for rec in recs:
                    print(f"  - {rec}")
        
        return recommendations
    
    def run_analysis(self, df):
        """
        Run the complete price-value perception analysis.
        
        Parameters:
        - df: DataFrame with review data
        
        Returns:
        - DataFrame with price-value analysis results
        """
        print("\n=== Running Price-Value Perception Analysis ===")
        
        if df is None or len(df) == 0:
            print("No data available for price-value analysis")
            return None
        
        # Detect price and value mentions
        df = self.detect_price_mentions(df)
        
        # Categorize price experiences
        df = self.categorize_price_experiences(df)
        
        # Analyze value sentiment
        df = self.analyze_value_sentiment(df)
        
        # Extract price phrases
        self.extract_price_phrases(df)
        
        # Analyze price patterns
        df = self.analyze_price_patterns(df)
        
        # Generate price-value recommendations
        self.generate_price_value_recommendations(df)
        
        print("\nPrice-value perception analysis complete.")
        return df