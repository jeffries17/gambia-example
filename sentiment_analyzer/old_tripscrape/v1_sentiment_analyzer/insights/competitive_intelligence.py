import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import json
import re

class CompetitiveAnalyzer:
    """
    Analyzes competitive intelligence in tourism reviews.
    Focuses on identifying how Tonga compares to other destinations,
    what unique advantages it has, and what aspects need improvement to be competitive.
    """
    
    def __init__(self, output_dir='competitive_insights'):
        """
        Initialize the competitive intelligence analyzer.
        
        Parameters:
        - output_dir: Directory to save competitive analysis outputs
        """
        self.output_dir = output_dir
        
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created competitive intelligence insights directory: {output_dir}")
        
        # Define comparison-related keywords and patterns
        self.comparison_keywords = {
            'comparative_words': ['better', 'worse', 'best', 'worst', 'compared', 'comparison', 
                                 'unlike', 'similar', 'same as', 'different', 'prefer', 'than',
                                 'superior', 'inferior', 'surpass', 'exceed', 'match', 'rival'],
            'competitor_mentions': ['fiji', 'samoa', 'vanuatu', 'bali', 'maldives', 'hawaii', 'thailand',
                                   'caribbean', 'pacific', 'bora bora', 'tahiti', 'cook islands',
                                   'australia', 'new zealand', 'new caledonia', 'other island'],
            'return_intention': ['return', 'come back', 'visit again', 'next time', 'next trip',
                               'repeat', 'revisit', 'back again', 'will be back', 'never again'],
            'uniqueness': ['unique', 'authentic', 'genuine', 'original', 'unlike anywhere', 'one of a kind',
                          'special', 'distinctive', 'incomparable', 'nowhere else', 'only in tonga']
        }
        
        # Define competitive aspects for analysis
        self.competitive_aspects = {
            'scenery_nature': ['beach', 'ocean', 'sea', 'lagoon', 'landscape', 'view', 'scenic', 
                             'nature', 'beauty', 'beautiful', 'pristine', 'paradise', 'tropical'],
            'culture_people': ['culture', 'tradition', 'local', 'people', 'friendly', 'welcoming',
                             'authentic', 'heritage', 'history', 'community', 'village', 'ceremony'],
            'activities_attractions': ['activity', 'tour', 'whale', 'snorkel', 'dive', 'swim', 'hike',
                                     'boat', 'fishing', 'kayak', 'adventure', 'attraction', 'experience'],
            'facilities_infrastructure': ['hotel', 'resort', 'room', 'accommodation', 'restaurant', 'road',
                                       'airport', 'transport', 'infrastructure', 'wifi', 'internet',
                                       'electricity', 'water', 'facilities', 'development'],
            'price_value': ['price', 'value', 'cost', 'money', 'expensive', 'cheap', 'affordable',
                          'worth', 'bargain', 'overpriced', 'reasonable', 'budget', 'premium'],
            'service_hospitality': ['service', 'staff', 'hospitality', 'friendly', 'helpful', 'attentive',
                                  'professional', 'welcoming', 'guide', 'host', 'manager', 'reception']
        }
        
        # List of Pacific and popular tropical destinations (competitors)
        self.competitor_destinations = [
            'fiji', 'samoa', 'cook islands', 'tahiti', 'bora bora', 'french polynesia',
            'vanuatu', 'new caledonia', 'solomon islands', 'hawaii', 'bali', 'thailand',
            'philippines', 'maldives', 'caribbean', 'bahamas', 'cancun', 'costa rica',
            'seychelles', 'mauritius', 'zanzibar', 'australia', 'new zealand'
        ]
    
    def detect_comparison_mentions(self, df):
        """
        Detect mentions of comparisons, competitors, and uniqueness in reviews.
        """
        print("Detecting comparison and competitor mentions in reviews...")
        
        # Make a copy of the DataFrame to avoid warnings
        df = df.copy()
        
        # For each comparison keyword category, create a flag with safe regex
        for category, keywords in self.comparison_keywords.items():
            # Create regex-safe keywords
            safe_keywords = [re.escape(keyword) for keyword in keywords]
            pattern = r'\b(' + '|'.join(safe_keywords) + r')\b'
            
            # Create the flag column
            df[f'mentions_{category}'] = df['text'].str.lower().str.contains(
                pattern,
                regex=True,
                na=False
            ).astype(int)
        
        # Create a general comparison mention flag
        mention_columns = ['mentions_comparative_words', 'mentions_competitor_mentions', 'mentions_uniqueness']
        df['mentions_comparison'] = df[mention_columns].any(axis=1).astype(int)
        
        # Calculate statistics
        comparison_mention_rate = df['mentions_comparison'].mean() * 100
        
        print(f"Comparison mentions found in {comparison_mention_rate:.1f}% of reviews")
        print(f"  Comparative words: {df['mentions_comparative_words'].sum()}")
        print(f"  Competitor mentions: {df['mentions_competitor_mentions'].sum()}")
        print(f"  Uniqueness mentions: {df['mentions_uniqueness'].sum()}")
        print(f"  Return intention mentions: {df['mentions_return_intention'].sum()}")
        
        return df
    
    def identify_specific_competitors(self, df):
      """
      Identify specific competitor destinations mentioned in reviews.
      
      Parameters:
      - df: DataFrame with review data
      
      Returns:
      - DataFrame with specific competitor mentions
      """
      print("\nIdentifying specific competitor destinations mentioned in reviews...")
      
      # Filter to reviews that mention competitors
      competitor_reviews = df[df['mentions_competitor_mentions'] == 1]
      
      if len(competitor_reviews) == 0:
          print("No competitor mentions found for analysis")
          return df
      
      # For each competitor destination, create a flag
      competitor_columns = []
      for destination in self.competitor_destinations:
          col_name = f'competitor_{destination.replace(" ", "_")}'
          # Use re.escape to properly escape spaces and special characters
          pattern = r'\b' + re.escape(destination) + r'\b'
          df[col_name] = df['text'].str.lower().str.contains(pattern, na=False).astype(int)
          competitor_columns.append(col_name)
          
          # Calculate mention counts for each competitor
          competitor_counts = {}
          for destination, col_name in zip(self.competitor_destinations, competitor_columns):
              count = df[col_name].sum()
              if count > 0:
                  competitor_counts[destination] = count
          
          # Sort by count
          sorted_competitors = sorted(competitor_counts.items(), key=lambda x: x[1], reverse=True)
          
          print("Specific competitor destinations mentioned:")
          for destination, count in sorted_competitors:
              print(f"  {destination}: {count} mentions")
          
          # Create a bar chart of competitor mentions
          if sorted_competitors:
              destinations = [d.title() for d, _ in sorted_competitors]
              counts = [c for _, c in sorted_competitors]
              
              plt.figure(figsize=(12, 6))
              bars = plt.bar(destinations, counts, color='skyblue')
              plt.title('Mentions of Other Destinations in Reviews')
              plt.xlabel('Destination')
              plt.ylabel('Number of Mentions')
              plt.xticks(rotation=45, ha='right')
              
              # Add count labels
              for bar, count in zip(bars, counts):
                  plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        str(count), ha='center', va='bottom')
              
              plt.tight_layout()
              plt.savefig(f'{self.output_dir}/competitor_mentions.png')
              print(f"Saved competitor mentions chart to {self.output_dir}/competitor_mentions.png")
          
          return df
    
    def categorize_comparative_aspects(self, df):
        """
        Categorize which aspects are mentioned in comparison-related reviews.
        """
        print("\nCategorizing comparison-related aspects...")
        
        # Filter to reviews that mention comparisons
        comparison_reviews = df[df['mentions_comparison'] == 1].copy()
        
        if len(comparison_reviews) == 0:
            print("No comparison-related reviews found for categorization")
            return df
        
        # For each aspect, create a flag using safe regex patterns
        for aspect, keywords in self.competitive_aspects.items():
            safe_keywords = [re.escape(keyword) for keyword in keywords]
            pattern = r'\b(' + '|'.join(safe_keywords) + r')\b'
            
            comparison_reviews[f'compare_{aspect}'] = comparison_reviews['text'].str.lower().str.contains(
                pattern,
                regex=True,
                na=False
            ).astype(int)
        
        # Calculate aspect mention statistics
        aspect_counts = {}
        for aspect in self.competitive_aspects.keys():
            col = f'compare_{aspect}'
            count = comparison_reviews[col].sum()
            percentage = (count / len(comparison_reviews)) * 100
            aspect_counts[aspect] = (count, percentage)
        
        # Sort aspects by count
        sorted_aspects = sorted(aspect_counts.items(), key=lambda x: x[1][0], reverse=True)
        
        # Print statistics
        print("Comparison mentions by aspect:")
        for aspect, (count, percentage) in sorted_aspects:
            print(f"  {aspect}: {count} mentions ({percentage:.1f}% of comparison reviews)")
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        aspects = [asp.replace('_', ' ').title() for asp, _ in sorted_aspects]
        counts = [count for _, (count, _) in sorted_aspects]
        
        plt.bar(aspects, counts, color='skyblue')
        plt.title('Comparison Mentions by Aspect')
        plt.xlabel('Aspect')
        plt.ylabel('Number of Mentions')
        plt.xticks(rotation=45, ha='right')
        
        plt.subplots_adjust(bottom=0.2)
        
        save_path = f'{self.output_dir}/comparison_mentions_by_aspect.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        print(f"Saved comparison aspects chart to {save_path}")
        
        # Add the calculated columns back to the main DataFrame
        aspect_columns = [f'compare_{aspect}' for aspect in self.competitive_aspects.keys()]
        df[aspect_columns] = 0  # Initialize columns
        df.loc[comparison_reviews.index, aspect_columns] = comparison_reviews[aspect_columns]
        
        return df
    
    def analyze_comparison_sentiment(self, df):
        """
        Analyze sentiment in comparison-related mentions.
        
        Parameters:
        - df: DataFrame with review data including comparison mentions
        
        Returns:
        - DataFrame with comparison sentiment analysis
        """
        print("\nAnalyzing sentiment in comparison-related mentions...")
        
        # Check for comparison mentions
        if 'mentions_comparison' not in df.columns or df['mentions_comparison'].sum() == 0:
            print("No comparison mentions found for sentiment analysis")
            return df
        
        # Filter to reviews that mention comparisons
        comparison_reviews = df[df['mentions_comparison'] == 1]
        
        # Calculate average sentiment for comparison-mentioning reviews
        avg_comparison_sentiment = comparison_reviews['sentiment_score'].mean()
        print(f"Average sentiment in comparison-mentioning reviews: {avg_comparison_sentiment:.3f}")
        
        # Compare to overall sentiment
        overall_sentiment = df['sentiment_score'].mean()
        sentiment_diff = avg_comparison_sentiment - overall_sentiment
        
        if sentiment_diff > 0.05:
            print("Comparison-mentioning reviews are more positive than average")
        elif sentiment_diff < -0.05:
            print("Comparison-mentioning reviews are more negative than average")
        else:
            print("Comparison-mentioning reviews have similar sentiment to average")
        
        # Calculate sentiment for different comparison types
        sentiment_by_type = {}
        for category in ['comparative_words', 'competitor_mentions', 'uniqueness', 'return_intention']:
            col = f'mentions_{category}'
            if comparison_reviews[col].sum() > 0:
                sentiment = comparison_reviews[comparison_reviews[col] == 1]['sentiment_score'].mean()
                sentiment_by_type[category] = sentiment
        
        if sentiment_by_type:
            print("\nSentiment by comparison type:")
            for category, sentiment in sentiment_by_type.items():
                print(f"  {category.replace('_', ' ')}: {sentiment:.3f}")
            
            # Create a bar chart of sentiment by comparison type
            categories = [cat.replace('_', ' ').title() for cat in sentiment_by_type.keys()]
            sentiments = list(sentiment_by_type.values())
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(categories, sentiments, color=['green' if s > 0 else 'red' for s in sentiments])
            plt.title('Sentiment by Comparison Type')
            plt.xlabel('Comparison Type')
            plt.ylabel('Average Sentiment Score (-1 to 1)')
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            
            # Add sentiment labels
            for bar, sentiment in zip(bars, sentiments):
                plt.text(bar.get_x() + bar.get_width()/2, 
                       sentiment + (0.02 if sentiment >= 0 else -0.08),
                       f'{sentiment:.2f}', ha='center', va='bottom' if sentiment >= 0 else 'top',
                       color='black')
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/sentiment_by_comparison_type.png')
            print(f"Saved comparison sentiment chart to {self.output_dir}/sentiment_by_comparison_type.png")
        
        # Calculate average sentiment for each aspect in comparison reviews
        sentiment_by_aspect = {}
        for aspect in self.competitive_aspects.keys():
            col = f'compare_{aspect}'
            if col in comparison_reviews.columns and comparison_reviews[col].sum() > 0:
                aspect_sentiment = comparison_reviews[comparison_reviews[col] == 1]['sentiment_score'].mean()
                sentiment_by_aspect[aspect] = aspect_sentiment
        
        if sentiment_by_aspect:
            # Sort by sentiment
            sorted_sentiments = sorted(sentiment_by_aspect.items(), key=lambda x: x[1], reverse=True)
            
            print("\nComparison sentiment by aspect:")
            for aspect, sentiment in sorted_sentiments:
                print(f"  {aspect.replace('_', ' ')}: {sentiment:.3f}")
            
            # Create a bar chart of sentiment by aspect
            aspects = [asp.replace('_', ' ').title() for asp, _ in sorted_sentiments]
            sentiments = [sent for _, sent in sorted_sentiments]
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(aspects, sentiments, color=['green' if s > 0 else 'red' for s in sentiments])
            plt.title('Comparison Sentiment by Aspect')
            plt.xlabel('Aspect')
            plt.ylabel('Average Sentiment Score (-1 to 1)')
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            
            # Add sentiment labels
            for bar, sentiment in zip(bars, sentiments):
                plt.text(bar.get_x() + bar.get_width()/2, 
                       sentiment + (0.02 if sentiment >= 0 else -0.08),
                       f'{sentiment:.2f}', ha='center', va='bottom' if sentiment >= 0 else 'top',
                       color='black')
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/comparison_sentiment_by_aspect.png')
            print(f"Saved aspect comparison sentiment chart to {self.output_dir}/comparison_sentiment_by_aspect.png")
        
        return df
    
    def analyze_return_intentions(self, df):
        """
        Analyze return intentions mentions in reviews.
        """
        print("\nAnalyzing return intentions in reviews...")
        
        # Check for return intention mentions
        if 'mentions_return_intention' not in df.columns or df['mentions_return_intention'].sum() == 0:
            print("No return intention mentions found for analysis")
            return df
        
        # Filter to reviews that mention return intentions
        return_reviews = df[df['mentions_return_intention'] == 1].copy()
        
        # Create regex-safe patterns for positive and negative mentions
        positive_keywords = [
            'return', 'come back', 'visit again', 'next time', 'will be back', 'repeat'
        ]
        negative_keywords = [
            'never again', 'will not return', 'would not return', "won't be back", 'last time'
        ]
        
        pos_pattern = r'\b(' + '|'.join(re.escape(k) for k in positive_keywords) + r')\b'
        neg_pattern = r'\b(' + '|'.join(re.escape(k) for k in negative_keywords) + r')\b'
        
        # Create flag columns
        return_reviews['positive_return'] = return_reviews['text'].str.lower().str.contains(
            pos_pattern, regex=True, na=False).astype(int)
        return_reviews['negative_return'] = return_reviews['text'].str.lower().str.contains(
            neg_pattern, regex=True, na=False).astype(int)
        
        # For reviews that don't match either pattern, use sentiment as a proxy
        unclassified = (return_reviews['positive_return'] == 0) & (return_reviews['negative_return'] == 0)
        return_reviews.loc[unclassified & (return_reviews['sentiment_score'] > 0), 'positive_return'] = 1
        return_reviews.loc[unclassified & (return_reviews['sentiment_score'] < 0), 'negative_return'] = 1
        
        # Calculate statistics
        positive_return_count = return_reviews['positive_return'].sum()
        negative_return_count = return_reviews['negative_return'].sum()
        return_sentiment = return_reviews['sentiment_score'].mean()
        
        print(f"Return intention mentions: {len(return_reviews)}")
        print(f"  Positive return intentions: {positive_return_count} "
            f"({positive_return_count/len(return_reviews)*100:.1f}%)")
        print(f"  Negative return intentions: {negative_return_count} "
            f"({negative_return_count/len(return_reviews)*100:.1f}%)")
        print(f"  Average sentiment: {return_sentiment:.3f}")
        
        # Create visualization
        plt.figure(figsize=(8, 8))
        plt.pie([positive_return_count, negative_return_count],
                labels=['Would Return', 'Would Not Return'],
                autopct='%1.1f%%',
                colors=['green', 'red'],
                startangle=90)
        plt.title('Return Intentions in Reviews')
        plt.axis('equal')
        
        save_path = f'{self.output_dir}/return_intentions.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        print(f"Saved return intentions chart to {save_path}")
        
        # Add results back to main DataFrame
        df['positive_return'] = 0
        df['negative_return'] = 0
        df.loc[return_reviews.index, 'positive_return'] = return_reviews['positive_return']
        df.loc[return_reviews.index, 'negative_return'] = return_reviews['negative_return']
        
        return df
    
    def extract_unique_advantages(self, df):
        """
        Extract mentions of unique advantages and strengths relative to competitors.
        
        Parameters:
        - df: DataFrame with review data including uniqueness mentions
        
        Returns:
        - DataFrame with uniqueness analysis
        """
        print("\nExtracting unique advantages and strengths...")
        
        # Check for uniqueness mentions
        if 'mentions_uniqueness' not in df.columns or df['mentions_uniqueness'].sum() == 0:
            print("No uniqueness mentions found for analysis")
            return df
        
        # Filter to reviews that mention uniqueness
        uniqueness_reviews = df[df['mentions_uniqueness'] == 1]
        
        # Calculate sentiment for uniqueness reviews
        uniqueness_sentiment = uniqueness_reviews['sentiment_score'].mean()
        print(f"Average sentiment in uniqueness-mentioning reviews: {uniqueness_sentiment:.3f}")
        
        # Analyze which aspects are mentioned as unique
        unique_aspects = {}
        for aspect in self.competitive_aspects.keys():
            col = f'compare_{aspect}'
            if col in uniqueness_reviews.columns:
                count = uniqueness_reviews[col].sum()
                if count > 0:
                    percentage = (count / len(uniqueness_reviews)) * 100
                    sentiment = uniqueness_reviews[uniqueness_reviews[col] == 1]['sentiment_score'].mean()
                    unique_aspects[aspect] = {
                        'count': count,
                        'percentage': percentage,
                        'sentiment': sentiment
                    }
        
        if unique_aspects:
            # Sort by count
            sorted_aspects = sorted(unique_aspects.items(), key=lambda x: x[1]['count'], reverse=True)
            
            print("\nAspects mentioned as unique:")
            for aspect, metrics in sorted_aspects:
                print(f"  {aspect.replace('_', ' ')}: {metrics['count']} mentions "
                     f"({metrics['percentage']:.1f}%, sentiment: {metrics['sentiment']:.3f})")
            
            # Create a visualization of unique aspects
            aspects = [asp.replace('_', ' ').title() for asp, _ in sorted_aspects]
            counts = [m['count'] for _, m in sorted_aspects]
            sentiments = [m['sentiment'] for _, m in sorted_aspects]
            
            # Set up a more complex figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot mention counts
            bars1 = ax1.bar(aspects, counts, color='skyblue')
            ax1.set_title('Unique Advantages by Aspect (Mention Count)')
            ax1.set_xlabel('Aspect')
            ax1.set_ylabel('Number of Mentions')
            ax1.set_xticklabels(aspects, rotation=45, ha='right')
            
            # Add count labels
            for bar, count in zip(bars1, counts):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        str(count), ha='center', va='bottom')
            
            # Plot sentiment
            bars2 = ax2.bar(aspects, sentiments, color=['green' if s > 0 else 'red' for s in sentiments])
            ax2.set_title('Sentiment of Unique Advantage Mentions')
            ax2.set_xlabel('Aspect')
            ax2.set_ylabel('Average Sentiment Score (-1 to 1)')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax2.set_xticklabels(aspects, rotation=45, ha='right')
            
            # Add sentiment labels
            for bar, sentiment in zip(bars2, sentiments):
                ax2.text(bar.get_x() + bar.get_width()/2, 
                       sentiment + (0.02 if sentiment >= 0 else -0.08),
                       f'{sentiment:.2f}', ha='center', va='bottom' if sentiment >= 0 else 'top',
                       color='black')
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/unique_advantages.png')
            print(f"Saved unique advantages chart to {self.output_dir}/unique_advantages.png")
            
            # Extract common words in uniqueness reviews
            if 'processed_text' in uniqueness_reviews.columns:
                uniqueness_text = ' '.join(uniqueness_reviews['processed_text'].fillna(''))
                uniqueness_words = uniqueness_text.split()
                word_counts = Counter(uniqueness_words)
                
                # Filter to words appearing at least 2 times
                common_words = {word: count for word, count in word_counts.items() if count >= 2}
                
                if common_words:
                    print("\nCommon words in uniqueness reviews:")
                    for word, count in sorted(common_words.items(), key=lambda x: x[1], reverse=True)[:15]:
                        print(f"  {word}: {count}")
                    
                    # Create word cloud for uniqueness phrases
                    wordcloud = WordCloud(width=800, height=400, background_color='white', 
                                       max_words=100, contour_width=3, contour_color='steelblue')
                    wordcloud.generate_from_frequencies(common_words)
                    
                    plt.figure(figsize=(10, 6))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis('off')
                    plt.title('Common Words in Uniqueness-Mentioning Reviews')
                    plt.tight_layout()
                    plt.savefig(f'{self.output_dir}/uniqueness_wordcloud.png')
                    print(f"Saved uniqueness wordcloud to {self.output_dir}/uniqueness_wordcloud.png")
        
        return df
    
    def generate_competitive_recommendations(self, df):
        """
        Generate recommendations based on competitive intelligence analysis.
        
        Parameters:
        - df: DataFrame with competitive analysis results
        
        Returns:
        - Dictionary with competitive recommendations
        """
        print("\nGenerating competitive intelligence recommendations...")
        
        recommendations = {
            "unique_selling_points": [],
            "competitive_improvements": [],
            "marketing_positioning": [],
            "destination_comparison": []
        }
        
        # Check for comparison mentions
        if 'mentions_comparison' not in df.columns or df['mentions_comparison'].sum() == 0:
            print("No comparison mentions found for generating recommendations")
            return recommendations
        
        # Analyze return intentions
        if 'mentions_return_intention' in df.columns and df['mentions_return_intention'].sum() > 0:
            return_reviews = df[df['mentions_return_intention'] == 1]
            positive_return = return_reviews['positive_return'].sum()
            negative_return = return_reviews['negative_return'].sum()
            
            # Calculate return rate
            if len(return_reviews) > 0:
                positive_rate = positive_return / len(return_reviews) * 100
                
                if positive_rate > 75:
                    recommendations["marketing_positioning"].append(
                        f"Highlight strong visitor loyalty in marketing with {positive_rate:.1f}% "
                        "of return intention mentions being positive"
                    )
                elif positive_rate < 50:
                    recommendations["competitive_improvements"].append(
                        f"Address loyalty concerns as only {positive_rate:.1f}% "
                        "of return intention mentions are positive"
                    )
        
        # Analyze uniqueness mentions
        if 'mentions_uniqueness' in df.columns and df['mentions_uniqueness'].sum() > 0:
            uniqueness_reviews = df[df['mentions_uniqueness'] == 1]
            uniqueness_sentiment = uniqueness_reviews['sentiment_score'].mean()
            
            # Find aspects that are mentioned as unique with positive sentiment
            unique_positive_aspects = []
            for aspect in self.competitive_aspects.keys():
                col = f'compare_{aspect}'
                if col in uniqueness_reviews.columns:
                    aspect_reviews = uniqueness_reviews[uniqueness_reviews[col] == 1]
                    if len(aspect_reviews) >= 3:  # Enough mentions to be meaningful
                        aspect_sentiment = aspect_reviews['sentiment_score'].mean()
                        if aspect_sentiment > 0.2:  # Strong positive sentiment
                            unique_positive_aspects.append((aspect, aspect_sentiment, len(aspect_reviews)))
            
            # Sort by sentiment and count
            unique_positive_aspects.sort(key=lambda x: (x[1], x[2]), reverse=True)
            
            # Generate recommendations for top unique aspects
            for aspect, sentiment, count in unique_positive_aspects[:3]:  # Top 3
                recommendations["unique_selling_points"].append(
                    f"Leverage {aspect.replace('_', ' ')} as a key differentiator which is seen as "
                    f"uniquely positive (sentiment: {sentiment:.2f}, mentions: {count})"
                )
        
        # Analyze competitor mentions
        competitor_columns = [col for col in df.columns if col.startswith('competitor_')]
        if competitor_columns:
            # Find most mentioned competitors
            competitor_counts = {}
            for col in competitor_columns:
                destination = col.replace('competitor_', '').replace('_', ' ').title()
                count = df[col].sum()
                if count >= 3:  # Enough mentions to be meaningful
                    # Calculate sentiment for mentions of this competitor
                    sentiment = df[df[col] == 1]['sentiment_score'].mean()
                    competitor_counts[destination] = (count, sentiment)
            
            # Sort by count
            sorted_competitors = sorted(competitor_counts.items(), key=lambda x: x[1][0], reverse=True)
            
            if sorted_competitors:
                top_competitor, (count, sentiment) = sorted_competitors[0]
                
                recommendations["destination_comparison"].append(
                    f"Develop targeted positioning against {top_competitor}, the most frequently "
                    f"compared destination (mentions: {count}, sentiment: {sentiment:.2f})"
                )
        
        # Find competitive weaknesses (aspects with negative sentiment in comparison reviews)
        comparison_reviews = df[df['mentions_comparison'] == 1]
        competitive_weaknesses = []
        
        for aspect in self.competitive_aspects.keys():
            col = f'compare_{aspect}'
            if col in comparison_reviews.columns:
                aspect_reviews = comparison_reviews[comparison_reviews[col] == 1]
                if len(aspect_reviews) >= 3:  # Enough mentions to be meaningful
                    aspect_sentiment = aspect_reviews['sentiment_score'].mean()
                    if aspect_sentiment < -0.1:  # Negative sentiment
                        competitive_weaknesses.append((aspect, aspect_sentiment, len(aspect_reviews)))
        
        # Sort by sentiment (most negative first)
        competitive_weaknesses.sort(key=lambda x: x[1])
        
        # Generate recommendations for competitive weaknesses
        for aspect, sentiment, count in competitive_weaknesses[:3]:  # Top 3 weaknesses
          recommendations["competitive_improvements"].append(
              f"Improve {aspect.replace('_', ' ')} to match competitor standards "
              f"as it receives negative comparisons (sentiment: {sentiment:.2f}, mentions: {count})"
          )
        
        return recommendations
    
    def run_analysis(self, df):
        """
        Run the complete competitive intelligence analysis.
        
        Parameters:
        - df: DataFrame with review data
        
        Returns:
        - DataFrame with competitive intelligence analysis results
        """
        print("\n=== Running Competitive Intelligence Analysis ===")
        
        if df is None or len(df) == 0:
            print("No data available for competitive analysis")
            return None
        
        # Detect comparison and competitor mentions
        df = self.detect_comparison_mentions(df)
        
        # Identify specific competitors
        df = self.identify_specific_competitors(df)
        
        # Categorize comparison aspects
        df = self.categorize_comparative_aspects(df)
        
        # Analyze comparison sentiment
        df = self.analyze_comparison_sentiment(df)
        
        # Analyze return intentions
        df = self.analyze_return_intentions(df)
        
        # Extract unique advantages
        df = self.extract_unique_advantages(df)
        
        # Generate competitive recommendations
        self.generate_competitive_recommendations(df)
        
        print("\nCompetitive intelligence analysis complete.")
        return df