import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import re
import json

class RestaurantAnalysis:
    """
    Extension class for analyzing restaurant-specific reviews in Tonga tourism data.
    """
    
    def __init__(self, base_analyzer, output_dir='restaurant_insights'):
        """
        Initialize with reference to the base analyzer.
        
        Parameters:
        - base_analyzer: Base TongaTourismAnalysis instance
        - output_dir: Directory to save restaurant-specific outputs
        """
        self.analyzer = base_analyzer
        
        # Create full path if output_dir is relative
        if not os.path.isabs(output_dir):
            self.output_dir = os.path.join(base_analyzer.output_dir, output_dir)
        else:
            self.output_dir = output_dir
            
        # Create output directory
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created restaurant analysis directory: {self.output_dir}")
            
        # Restaurant-specific categories and analyses setup
        self.setup_food_categories()
        
    def setup_food_categories(self):
        """
        Set up food and restaurant-specific categories for analysis.
        """
        # Define restaurant-specific categories
        self.food_categories = {
            'seafood': ['fish', 'seafood', 'lobster', 'crab', 'prawn', 'shrimp', 'mussel', 
                       'clam', 'oyster', 'sushi', 'octopus', 'squid', 'calamari'],
            'meat_dishes': ['meat', 'beef', 'pork', 'lamb', 'steak', 'burger', 'chicken', 
                           'bacon', 'sausage', 'ham', 'bbq', 'barbecue', 'grill'],
            'vegetarian': ['vegetarian', 'vegan', 'plant', 'salad', 'vegetable', 'veggie',
                          'tofu', 'meatless', 'veg'],
            'local_cuisine': ['tonga', 'tongan', 'traditional', 'local', 'authentic', 
                            'island', 'pacific', 'native', 'indigenous', 'ota', 'umu', 'lu'],
            'international': ['international', 'western', 'italian', 'chinese', 'japanese', 
                            'indian', 'thai', 'french', 'fusion', 'cuisine', 'foreign'],
            'breakfast': ['breakfast', 'brunch', 'morning', 'coffee', 'pastry', 'egg', 
                         'toast', 'cereal', 'pancake', 'waffle'],
            'desserts': ['dessert', 'sweet', 'cake', 'ice cream', 'chocolate', 'pudding', 
                        'fruit', 'pastry', 'pie', 'cookie', 'candy'],
            'beverages': ['drink', 'beverage', 'coffee', 'tea', 'juice', 'water', 'cocktail', 
                         'wine', 'beer', 'alcohol', 'soda', 'smoothie']
        }
        
        # Dining experience categories
        self.dining_experience_categories = {
            'ambiance': ['ambiance', 'atmosphere', 'decor', 'design', 'interior', 'music', 
                        'noise', 'quiet', 'romantic', 'cozy', 'lighting', 'theme', 'vibe'],
            'service_quality': ['service', 'staff', 'waiter', 'waitress', 'server', 'attentive', 
                              'prompt', 'friendly', 'rude', 'slow', 'quick', 'knowledgeable'],
            'value_for_money': ['price', 'value', 'worth', 'expensive', 'cheap', 'affordable', 
                              'overpriced', 'reasonable', 'cost', 'bill', 'pricey'],
            'hygiene': ['clean', 'hygiene', 'sanitary', 'dirty', 'wash', 'toilet', 'bathroom', 
                       'restroom', 'kitchen', 'cockroach', 'bug', 'pest'],
            'waiting_time': ['wait', 'time', 'quick', 'slow', 'fast', 'delay', 'minute', 
                           'reservation', 'queue', 'long', 'immediate', 'prompt'],
            'portion_size': ['portion', 'size', 'large', 'small', 'generous', 'tiny', 'huge', 
                           'filling', 'enough', 'big', 'little', 'amount', 'quantity']
        }
    
    def verify_restaurant_data(self, df):
        """
        Verify that we're dealing with restaurant reviews through multiple checks:
        1. Check if there's a Food subrating
        2. Check webUrl contains Restaurant_Review
        3. Fallback to text analysis if needed
        
        Parameters:
        - df: DataFrame with review data
        
        Returns:
        - Boolean indicating if reviews are for restaurants
        """
        # Check if subratings contain Food rating
        if 'subratings' in df.columns:
            has_food_rating = df['subratings'].apply(
                lambda x: any(rating.get('name') == 'Food' for rating in x) if isinstance(x, list) else False
            )
            if has_food_rating.any():
                return True
                
        # Check webUrl path
        if 'placeInfo.webUrl' in df.columns:
            has_restaurant_url = df['placeInfo.webUrl'].str.contains('Restaurant_Review', case=False, na=False)
            if has_restaurant_url.any():
                return True
                
        # If neither check worked, assume it's restaurant data (since we're in the restaurant analyzer)
        print("Warning: Could not definitively verify restaurant reviews through metadata")
        return True
        
        # Create safe pattern
        pattern = r'\b(' + '|'.join(re.escape(term) for term in restaurant_terms) + r')\b'
        
        # Filter reviews that mention restaurant terms
        restaurant_mask = df['text'].str.lower().str.contains(pattern, na=False, regex=True)
        restaurant_df = df[restaurant_mask].copy()
        
        # Check place name if available
        if 'placeInfo.name' in df.columns:
            place_terms = ['restaurant', 'cafe', 'bar', 'grill', 'bistro', 'eatery', 'dining']
            place_pattern = r'\b(' + '|'.join(re.escape(term) for term in place_terms) + r')\b'
            place_mask = df['placeInfo.name'].str.lower().str.contains(place_pattern, na=False, regex=True)
            
            # Combine results
            combined_indices = restaurant_df.index.union(df[place_mask].index)
            restaurant_df = df.loc[combined_indices].copy()
        
        # Create flag for restaurant reviews in the original dataframe
        df['is_restaurant_review'] = df.index.isin(restaurant_df.index)
        
        print(f"Identified {len(restaurant_df)} restaurant reviews out of {len(df)} total reviews")
        return restaurant_df
    
    def analyze_food_categories(self, df):
        """
        Analyze food categories mentioned in reviews.
        """
        print("Analyzing food categories mentioned in restaurant reviews...")
        
        # Make a copy of the DataFrame
        df = df.copy()
        
        # For each food category, check if any keywords are present
        for category, keywords in self.food_categories.items():
            # Create safe pattern
            safe_keywords = [re.escape(keyword) for keyword in keywords]
            pattern = r'\b(' + '|'.join(safe_keywords) + r')\b'
            
            # Create the flag column
            df[f'food_{category}'] = df['processed_text'].str.contains(
                pattern, 
                case=False, 
                regex=True, 
                na=False
            ).astype(int)
        
        # Calculate food category mention counts
        food_cols = [f'food_{category}' for category in self.food_categories.keys()]
        category_counts = df[food_cols].sum().sort_values(ascending=False)
        
        print("Food category mentions:")
        for category, count in category_counts.items():
            category_name = category.replace('food_', '')
            print(f"  {category_name}: {count} mentions")
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        category_labels = [cat.replace('food_', '').replace('_', ' ').title() 
                        for cat in category_counts.index]
        
        bars = plt.bar(category_labels, category_counts.values, color='skyblue')
        plt.title('Food Categories Mentioned in Restaurant Reviews')
        plt.xlabel('Food Category')
        plt.ylabel('Number of Mentions')
        plt.xticks(rotation=45, ha='right')
        
        # Add count labels
        for bar, count in zip(bars, category_counts.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(int(count)), ha='center', va='bottom')
        
        plt.subplots_adjust(bottom=0.2)
        
        save_path = f'{self.output_dir}/food_category_mentions.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        print(f"Saved food category chart to {save_path}")
        return df
    
    def analyze_dining_experience(self, df):
        """
        Analyze dining experience aspects mentioned in reviews.
        """
        print("Analyzing dining experience aspects in restaurant reviews...")
        
        # Make a copy of the DataFrame
        df = df.copy()
        
        # For each dining experience category, check if any keywords are present
        for category, keywords in self.dining_experience_categories.items():
            # Create safe pattern
            safe_keywords = [re.escape(keyword) for keyword in keywords]
            pattern = r'\b(' + '|'.join(safe_keywords) + r')\b'
            
            # Create the flag column
            df[f'dining_{category}'] = df['processed_text'].str.contains(
                pattern,
                case=False,
                regex=True,
                na=False
            ).astype(int)
        
        # Calculate dining experience mention counts
        dining_cols = [f'dining_{category}' for category in self.dining_experience_categories.keys()]
        category_counts = df[dining_cols].sum().sort_values(ascending=False)
        
        print("Dining experience aspect mentions:")
        for category, count in category_counts.items():
            category_name = category.replace('dining_', '')
            print(f"  {category_name}: {count} mentions")
        
        # Create visualizations
        plt.figure(figsize=(12, 6))
        category_labels = [cat.replace('dining_', '').replace('_', ' ').title() 
                        for cat in category_counts.index]
        
        bars = plt.bar(category_labels, category_counts.values, color='lightgreen')
        plt.title('Dining Experience Aspects Mentioned in Reviews')
        plt.xlabel('Dining Aspect')
        plt.ylabel('Number of Mentions')
        plt.xticks(rotation=45, ha='right')
        
        # Add count labels
        for bar, count in zip(bars, category_counts.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(int(count)), ha='center', va='bottom')
        
        plt.subplots_adjust(bottom=0.2)
        
        save_path = f'{self.output_dir}/dining_aspect_mentions.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        # Calculate and visualize sentiment by aspect
        if 'sentiment_score' in df.columns:
            plt.figure(figsize=(12, 6))
            aspect_sentiment = {}
            
            for category in self.dining_experience_categories.keys():
                col = f'dining_{category}'
                # Calculate sentiment for reviews mentioning this aspect
                aspect_reviews = df[df[col] == 1]
                if len(aspect_reviews) > 0:
                    sentiment = aspect_reviews['sentiment_score'].mean()
                    aspect_sentiment[category] = sentiment
            
            if aspect_sentiment:
                aspects = list(aspect_sentiment.keys())
                sentiments = list(aspect_sentiment.values())
                colors = ['green' if s > 0 else 'red' for s in sentiments]
                
                bars = plt.bar(aspects, sentiments, color=colors)
                plt.title('Average Sentiment by Dining Experience Aspect')
                plt.xlabel('Dining Aspect')
                plt.ylabel('Average Sentiment Score (-1 to 1)')
                plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                plt.xticks(rotation=45, ha='right')
                
                # Add sentiment labels
                for bar, sentiment in zip(bars, sentiments):
                    label_pos = sentiment + 0.02 if sentiment >= 0 else sentiment - 0.08
                    plt.text(bar.get_x() + bar.get_width()/2, label_pos,
                            f'{sentiment:.2f}', ha='center', va='bottom' if sentiment >= 0 else 'top')
                
                plt.subplots_adjust(bottom=0.2)
                
                save_path = f'{self.output_dir}/dining_aspect_sentiment.png'
                plt.savefig(save_path, bbox_inches='tight')
                plt.close()
                
                print(f"Saved dining aspect sentiment chart to {save_path}")
        
        return df
    
    def analyze_price_value(self, df):
        """
        Analyze price-value perceptions in restaurant reviews.
        
        Parameters:
        - df: DataFrame with restaurant reviews
        
        Returns:
        - DataFrame with price-value analysis
        """
        print("Analyzing price-value perceptions in restaurant reviews...")
        
        # Price-related terms
        price_pattern = r'\b(price|cost|expensive|cheap|affordable|value|worth|money|overpriced|reasonable|budget|pricey)\b'
        
        # Identify reviews that mention price
        df['mentions_price'] = df['text'].str.lower().str.contains(price_pattern, na=False).astype(int)
        price_reviews = df[df['mentions_price'] == 1]
        
        print(f"Found {len(price_reviews)} reviews that mention price or value")
        
        if len(price_reviews) > 0:
            # Calculate average sentiment for price-mentioning reviews
            avg_price_sentiment = price_reviews['sentiment_score'].mean()
            print(f"Average sentiment for price-mentioning reviews: {avg_price_sentiment:.3f}")
            
            # Categorize price sentiment
            price_reviews['price_sentiment'] = price_reviews['sentiment_score'].apply(
                lambda x: 'positive' if x > 0.1 else ('negative' if x < -0.1 else 'neutral'))
            
            # Count by price sentiment category
            price_sentiment_counts = price_reviews['price_sentiment'].value_counts()
            
            # Create pie chart of price sentiment
            plt.figure(figsize=(10, 6))
            plt.pie(price_sentiment_counts, labels=price_sentiment_counts.index, 
                    autopct='%1.1f%%', startangle=90, colors=['#5CB85C', '#F0AD4E', '#D9534F'])
            plt.axis('equal')
            plt.title('Sentiment in Price-Mentioning Restaurant Reviews')
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/price_sentiment_distribution.png')
            print(f"Saved price sentiment chart to {self.output_dir}/price_sentiment_distribution.png")
            
            # Extract common phrases in positive and negative price reviews
            pos_price_reviews = price_reviews[price_reviews['price_sentiment'] == 'positive']
            neg_price_reviews = price_reviews[price_reviews['price_sentiment'] == 'negative']
            
            # Get positive price value phrases
            if len(pos_price_reviews) > 0:
                pos_price_text = ' '.join(pos_price_reviews['processed_text'].fillna(''))
                pos_price_words = pos_price_text.split()
                pos_word_counts = Counter(pos_price_words)
                pos_common_words = dict(pos_word_counts.most_common(20))
                
                # Create word cloud for positive price mentions
                if pos_common_words:
                    wc = WordCloud(width=800, height=400, background_color='white',
                                  max_words=50, contour_width=3, contour_color='steelblue')
                    wc.generate_from_frequencies(pos_common_words)
                    
                    plt.figure(figsize=(10, 6))
                    plt.imshow(wc, interpolation='bilinear')
                    plt.axis('off')
                    plt.title('Common Words in Positive Price-Value Reviews')
                    plt.tight_layout()
                    plt.savefig(f'{self.output_dir}/positive_price_wordcloud.png')
            
            # Get negative price value phrases
            if len(neg_price_reviews) > 0:
                neg_price_text = ' '.join(neg_price_reviews['processed_text'].fillna(''))
                neg_price_words = neg_price_text.split()
                neg_word_counts = Counter(neg_price_words)
                neg_common_words = dict(neg_word_counts.most_common(20))
                
                # Create word cloud for negative price mentions
                if neg_common_words:
                    wc = WordCloud(width=800, height=400, background_color='white',
                                  max_words=50, contour_width=3, contour_color='darkred')
                    wc.generate_from_frequencies(neg_common_words)
                    
                    plt.figure(figsize=(10, 6))
                    plt.imshow(wc, interpolation='bilinear')
                    plt.axis('off')
                    plt.title('Common Words in Negative Price-Value Reviews')
                    plt.tight_layout()
                    plt.savefig(f'{self.output_dir}/negative_price_wordcloud.png')
        
        return df
    
    def analyze_local_vs_international_cuisine(self, df):
        """
        Compare mentions and sentiment for local vs. international cuisine.
        
        Parameters:
        - df: DataFrame with restaurant reviews
        
        Returns:
        - DataFrame with cuisine comparison analysis
        """
        print("Analyzing local vs. international cuisine mentions...")
        
        # Check if the necessary food category columns exist
        if 'food_local_cuisine' not in df.columns or 'food_international' not in df.columns:
            print("Food category columns not found, can't compare cuisines")
            return df
        
        # Count reviews mentioning each cuisine type
        local_reviews = df[df['food_local_cuisine'] == 1]
        intl_reviews = df[df['food_international'] == 1]
        
        print(f"Local cuisine mentioned in {len(local_reviews)} reviews")
        print(f"International cuisine mentioned in {len(intl_reviews)} reviews")
        
        # Compare ratings and sentiment
        comparison_data = []
        
        if len(local_reviews) > 0:
            local_rating = local_reviews['rating'].mean() if 'rating' in local_reviews.columns else None
            local_sentiment = local_reviews['sentiment_score'].mean()
            comparison_data.append({
                'Cuisine Type': 'Local Tongan Cuisine',
                'Reviews': len(local_reviews),
                'Avg Rating': local_rating,
                'Avg Sentiment': local_sentiment
            })
        
        if len(intl_reviews) > 0:
            intl_rating = intl_reviews['rating'].mean() if 'rating' in intl_reviews.columns else None
            intl_sentiment = intl_reviews['sentiment_score'].mean()
            comparison_data.append({
                'Cuisine Type': 'International Cuisine',
                'Reviews': len(intl_reviews),
                'Avg Rating': intl_rating,
                'Avg Sentiment': intl_sentiment
            })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            print(comparison_df)
            
            # Create comparison chart
            plt.figure(figsize=(12, 6))
            x = comparison_df['Cuisine Type']
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Rating comparison
            if 'Avg Rating' in comparison_df.columns and not comparison_df['Avg Rating'].isna().all():
                ax1.bar(x, comparison_df['Avg Rating'], color=['#1B9E77', '#D95F02'])
                ax1.set_title('Average Rating by Cuisine Type')
                ax1.set_ylim(1, 5)
                ax1.set_ylabel('Average Rating')
                
                # Add text labels
                for i, v in enumerate(comparison_df['Avg Rating']):
                    if not pd.isna(v):
                        ax1.text(i, v - 0.3, f"{v:.2f}", ha='center', color='white', fontweight='bold')
            
            # Sentiment comparison
            ax2.bar(x, comparison_df['Avg Sentiment'], color=['#1B9E77', '#D95F02'])
            ax2.set_title('Average Sentiment by Cuisine Type')
            ax2.set_ylabel('Average Sentiment Score (-1 to 1)')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Add text labels
            for i, v in enumerate(comparison_df['Avg Sentiment']):
                if not pd.isna(v):
                    ax2.text(i, v - 0.05 if v > 0 else v + 0.05, 
                           f"{v:.2f}", ha='center', 
                           color='white' if v > 0 else 'black', 
                           fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/cuisine_comparison.png')
            print(f"Saved cuisine comparison chart to {self.output_dir}/cuisine_comparison.png")
            
            # Save comparison data
            comparison_df.to_csv(f'{self.output_dir}/cuisine_comparison.csv', index=False)
        
        return df
    
    def generate_restaurant_recommendations(self, df):
        """
        Generate restaurant-specific recommendations based on the analysis.
        
        Parameters:
        - df: DataFrame with restaurant reviews and analysis
        
        Returns:
        - Dictionary with restaurant recommendations
        """
        print("Generating restaurant-specific recommendations...")
        
        recommendations = {
            "dining_experience": [],
            "cuisine": [],
            "price_value": [],
            "marketing": []
        }
        
        # Dining experience recommendations
        dining_cols = [f'dining_{category}' for category in self.dining_experience_categories.keys()]
        if any(col in df.columns for col in dining_cols):
            # Get counts and sentiment for each dining aspect
            dining_metrics = {}
            for category in self.dining_experience_categories.keys():
                col = f'dining_{category}'
                if col in df.columns:
                    aspect_df = df[df[col] == 1]
                    if len(aspect_df) > 0:
                        count = len(aspect_df)
                        sentiment = aspect_df['sentiment_score'].mean()
                        dining_metrics[category] = {'count': count, 'sentiment': sentiment}
            
            # Identify aspects with negative sentiment
            negative_aspects = {k: v for k, v in dining_metrics.items() if v['sentiment'] < 0}
            for aspect, metrics in negative_aspects.items():
                recommendations["dining_experience"].append(
                    f"Improve {aspect.replace('_', ' ')} which has negative sentiment ({metrics['sentiment']:.2f})"
                )
            
            # Identify most mentioned aspects (focus areas)
            sorted_aspects = sorted(dining_metrics.items(), key=lambda x: x[1]['count'], reverse=True)
            top_aspects = sorted_aspects[:3]
            for aspect, metrics in top_aspects:
                recommendations["dining_experience"].append(
                    f"Focus on {aspect.replace('_', ' ')} as it's frequently mentioned ({metrics['count']} times)"
                )
        
        # Cuisine recommendations
        food_cols = [f'food_{category}' for category in self.food_categories.keys()]
        if any(col in df.columns for col in food_cols):
            # Check local vs international cuisine balance
            local_count = df['food_local_cuisine'].sum() if 'food_local_cuisine' in df.columns else 0
            intl_count = df['food_international'].sum() if 'food_international' in df.columns else 0
            
            if local_count < intl_count:
                recommendations["cuisine"].append(
                    "Increase focus on authentic local Tongan cuisine as it's mentioned less than international options"
                )
            
            # Identify most and least mentioned food categories
            food_mentions = {category: df[f'food_{category}'].sum() for category in self.food_categories.keys() 
                            if f'food_{category}' in df.columns}
            
            if food_mentions:
                least_mentioned = min(food_mentions.items(), key=lambda x: x[1])
                if least_mentioned[1] < 5:  # If very few mentions
                    recommendations["cuisine"].append(
                        f"Consider expanding {least_mentioned[0].replace('_', ' ')} options which are rarely mentioned"
                    )
        
        # Price-value recommendations
        if 'mentions_price' in df.columns and df['mentions_price'].sum() > 0:
            price_reviews = df[df['mentions_price'] == 1]
            avg_price_sentiment = price_reviews['sentiment_score'].mean()
            
            if avg_price_sentiment < 0:
                recommendations["price_value"].append(
                    "Address price-value perception issues as sentiment is negative for price-mentioning reviews"
                )
            elif avg_price_sentiment > 0.3:  # Strong positive sentiment
                recommendations["price_value"].append(
                    "Leverage positive price-value perception in marketing materials"
                )
        
        # Marketing recommendations
        # Find what visitors love about restaurants
        positive_reviews = df[df['sentiment_score'] > 0.3]  # Strongly positive reviews
        if len(positive_reviews) > 0:
            # Get the most common food categories in positive reviews
            pos_food_mentions = {category: positive_reviews[f'food_{category}'].sum() 
                               for category in self.food_categories.keys() 
                               if f'food_{category}' in positive_reviews.columns}
            
            if pos_food_mentions:
                top_category = max(pos_food_mentions.items(), key=lambda x: x[1])
                if top_category[1] > 0:
                    recommendations["marketing"].append(
                        f"Highlight {top_category[0].replace('_', ' ')} in marketing as it receives the most positive mentions"
                    )
        
        # Save recommendations
        with open(f'{self.output_dir}/restaurant_recommendations.json', 'w', encoding='utf-8') as f:
            json.dump(recommendations, f, indent=2)
        
        # Print key recommendations
        print("\nKey restaurant recommendations:")
        for category, recs in recommendations.items():
            if recs:
                print(f"\n{category.replace('_', ' ').title()}:")
                for rec in recs:
                    print(f"  - {rec}")
        
        return recommendations
    
    def run_analysis(self, df=None):
        """
        Run the complete restaurant analysis.
        
        Parameters:
        - df: DataFrame with review data (optional, will get from base analyzer if None)
        
        Returns:
        - DataFrame with restaurant analysis results
        """
        if df is None:
            # Get the processed data from the base analyzer
            df = self.analyzer.get_processed_data()
            
        if df is None or len(df) == 0:
            print("No data available for restaurant analysis")
            return None
        
        print("\n=== Running Restaurant Analysis ===")
        
        # Assuming all data in df is already restaurant-related
        restaurant_df = df
        
        if len(restaurant_df) == 0:
            print("No restaurant reviews found in the data")
            return df
        
        # Run restaurant-specific analyses
        restaurant_df = self.analyze_food_categories(restaurant_df)
        restaurant_df = self.analyze_dining_experience(restaurant_df)
        restaurant_df = self.analyze_price_value(restaurant_df)
        restaurant_df = self.analyze_local_vs_international_cuisine(restaurant_df)
        
        # Generate restaurant-specific recommendations
        self.generate_restaurant_recommendations(restaurant_df)
        
        print("\nRestaurant analysis complete.")
        return restaurant_df
