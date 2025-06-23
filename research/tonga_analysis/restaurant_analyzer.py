import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import json
from wordcloud import WordCloud

class RestaurantAnalyzer:
    """
    Comprehensive analyzer for restaurant reviews in Tonga tourism data.
    Combines features from both v1 and v2 implementations with enhanced
    visualizations and extended analysis capabilities.
    Includes island-specific analysis capabilities.
    """
    
    def __init__(self, sentiment_analyzer, output_dir='outputs/restaurant_analysis'):
        """
        Initialize the restaurant analyzer.
        
        Parameters:
        - sentiment_analyzer: Instance of SentimentAnalyzer for text processing
        - output_dir: Directory for restaurant-specific outputs
        """
        self.sentiment_analyzer = sentiment_analyzer
        self.output_dir = output_dir
        
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created restaurant analysis directory: {output_dir}")
            
        # Create visualization directory
        self.viz_dir = os.path.join(self.output_dir, 'visualizations')
        if not os.path.exists(self.viz_dir):
            os.makedirs(self.viz_dir)
            
        # Create island analysis directory
        self.island_dir = os.path.join(self.output_dir, 'island_analysis')
        if not os.path.exists(self.island_dir):
            os.makedirs(self.island_dir)
            print(f"Created island analysis directory: {self.island_dir}")
            
        # Define restaurant-specific categories
        self.food_categories = {
            'local_cuisine': [
                'tongan', 'traditional', 'local', 'authentic', 'island', 'pacific',
                'ota', 'ika', 'lu', 'taro', 'yam', 'coconut', 'tropical'
            ],
            'seafood': [
                'fish', 'seafood', 'tuna', 'mahi', 'lobster', 'crab', 'prawn',
                'octopus', 'sushi', 'raw fish', 'fresh catch', 'shrimp', 'mussel', 
                'clam', 'oyster', 'squid', 'calamari'
            ],
            'meat_dishes': [
                'meat', 'beef', 'pork', 'lamb', 'steak', 'burger', 'chicken', 
                'bacon', 'sausage', 'ham', 'bbq', 'barbecue', 'grill'
            ],
            'vegetarian': [
                'vegetarian', 'vegan', 'plant', 'salad', 'vegetable', 'veggie',
                'tofu', 'meatless', 'veg'
            ],
            'international': [
                'chinese', 'asian', 'italian', 'western', 'fusion', 'indian',
                'japanese', 'pizza', 'pasta', 'curry', 'thai', 'french', 'cuisine', 'foreign'
            ],
            'breakfast': [
                'breakfast', 'brunch', 'morning', 'coffee', 'pastry', 'egg', 
                'toast', 'cereal', 'pancake', 'waffle'
            ],
            'desserts': [
                'dessert', 'sweet', 'cake', 'ice cream', 'chocolate', 'pudding', 
                'fruit', 'pastry', 'pie', 'cookie', 'candy'
            ],
            'beverages': [
                'drink', 'beverage', 'coffee', 'tea', 'juice', 'water', 'cocktail', 
                'wine', 'beer', 'alcohol', 'soda', 'smoothie'
            ]
        }
        
        self.experience_aspects = {
            'service': [
                'service', 'staff', 'friendly', 'attentive', 'helpful', 'wait',
                'server', 'waiter', 'waitress', 'hospitality', 'prompt', 'rude', 
                'slow', 'quick', 'knowledgeable'
            ],
            'atmosphere': [
                'atmosphere', 'ambiance', 'decor', 'view', 'setting', 'romantic',
                'casual', 'quiet', 'busy', 'crowd', 'design', 'interior', 'music', 
                'noise', 'cozy', 'lighting', 'theme', 'vibe'
            ],
            'value': [
                'price', 'value', 'expensive', 'cheap', 'reasonable', 'worth',
                'overpriced', 'bargain', 'cost', 'affordable', 'bill', 'pricey'
            ],
            'hygiene': [
                'clean', 'hygiene', 'sanitary', 'dirty', 'wash', 'toilet', 'bathroom', 
                'restroom', 'kitchen', 'cockroach', 'bug', 'pest'
            ],
            'waiting_time': [
                'wait', 'time', 'quick', 'slow', 'fast', 'delay', 'minute', 
                'reservation', 'queue', 'long', 'immediate', 'prompt'
            ],
            'portion_size': [
                'portion', 'size', 'large', 'small', 'generous', 'tiny', 'huge', 
                'filling', 'enough', 'big', 'little', 'amount', 'quantity'
            ]
        }
        
        self.cultural_elements = [
            'traditional', 'authentic', 'local', 'tongan', 'culture', 'custom',
            'ceremony', 'feast', 'kava', 'dance', 'music', 'heritage'
        ]
        
        # Meal types for analysis
        self.meal_keywords = {
            'breakfast': ['breakfast', 'brunch', 'morning', 'coffee'],
            'lunch': ['lunch', 'afternoon'],
            'dinner': ['dinner', 'evening', 'night']
        }

    def filter_restaurant_reviews(self, df):
        """
        Filter to only restaurant reviews and add restaurant-specific features.
        Combines approaches from both v1 and v2.
        """
        # Method 1: Direct category filtering (from v1)
        category_filtered = df[df['category'] == 'restaurant'].copy() if 'category' in df.columns else pd.DataFrame()
        
        # Method 2: Keyword-based filtering (from v2)
        restaurant_terms = [
            'restaurant', 'food', 'eat', 'dining', 'dinner', 'lunch', 
            'breakfast', 'meal', 'cafe', 'cuisine', 'dish', 'menu',
            'chef', 'waiter', 'waitress', 'appetizer', 'entree'
        ]
        pattern = '|'.join(restaurant_terms)
        term_filtered = df[df['text'].str.lower().str.contains(pattern, na=False)].copy()
        
        # Method 3: Place name filtering (from v2)
        place_filtered = pd.DataFrame()
        if 'place_name' in df.columns:
            place_pattern = 'restaurant|cafe|bar|grill|bistro|eatery|dining'
            place_filtered = df[df['place_name'].str.lower().str.contains(place_pattern, na=False)].copy()
        elif 'placeInfo.name' in df.columns:
            place_pattern = 'restaurant|cafe|bar|grill|bistro|eatery|dining'
            place_filtered = df[df['placeInfo.name'].str.lower().str.contains(place_pattern, na=False)].copy()
        
        # Combine all filtering methods
        combined_indices = set()
        if not category_filtered.empty:
            combined_indices.update(category_filtered.index)
        if not term_filtered.empty:
            combined_indices.update(term_filtered.index)
        if not place_filtered.empty:
            combined_indices.update(place_filtered.index)
        
        # Create final filtered dataframe
        restaurant_df = df.loc[list(combined_indices)].copy() if combined_indices else pd.DataFrame()
        
        if restaurant_df.empty:
            print("No restaurant reviews found!")
            return restaurant_df
        
        # Flag restaurant reviews in the original dataframe
        df['is_restaurant_review'] = df.index.isin(restaurant_df.index)
        
        # Add cuisine type flags
        for cuisine, keywords in self.food_categories.items():
            pattern = '|'.join(keywords)
            restaurant_df[f'cuisine_{cuisine}'] = restaurant_df['text'].str.lower().str.contains(
                pattern, na=False).astype(int)
        
        # Add experience aspect flags
        for aspect, keywords in self.experience_aspects.items():
            pattern = '|'.join(keywords)
            restaurant_df[f'aspect_{aspect}'] = restaurant_df['text'].str.lower().str.contains(
                pattern, na=False).astype(int)
        
        # Add cultural mention flag
        pattern = '|'.join(self.cultural_elements)
        restaurant_df['mentions_cultural'] = restaurant_df['text'].str.lower().str.contains(
            pattern, na=False).astype(int)
        
        # Add price mention flag
        price_pattern = r'\b(price|cost|expensive|cheap|affordable|value|worth|money|overpriced|reasonable|budget|pricey)\b'
        restaurant_df['mentions_price'] = restaurant_df['text'].str.lower().str.contains(price_pattern, na=False).astype(int)
        
        print(f"Analyzing {len(restaurant_df)} restaurant reviews")
        return restaurant_df

    def analyze_cuisine_preferences(self, df):
        """
        Analyze preferences and sentiment around different cuisine types.
        """
        cuisine_stats = {}
        
        for cuisine in self.food_categories.keys():
            cuisine_reviews = df[df[f'cuisine_{cuisine}'] == 1]
            
            if len(cuisine_reviews) > 0:
                stats = {
                    'review_count': len(cuisine_reviews)
                }
                
                # Add sentiment data if available
                if 'sentiment_score' in cuisine_reviews.columns:
                    stats['avg_sentiment'] = cuisine_reviews['sentiment_score'].mean()
                    
                if 'sentiment_category' in cuisine_reviews.columns:
                    stats['positive_reviews'] = len(cuisine_reviews[cuisine_reviews['sentiment_category'] == 'positive'])
                    
                if 'rating' in cuisine_reviews.columns:
                    stats['avg_rating'] = cuisine_reviews['rating'].mean()
                cuisine_stats[cuisine] = stats
        
        return cuisine_stats

    def analyze_cultural_elements(self, df):
        """
        Analyze mentions and sentiment around cultural elements.
        """
        cultural_reviews = df[df['mentions_cultural'] == 1]
        
        cultural_analysis = {
            'review_count': len(cultural_reviews)
        }
        
        # Add sentiment data if available
        if 'sentiment_score' in cultural_reviews.columns:
            cultural_analysis['avg_sentiment'] = cultural_reviews['sentiment_score'].mean()
            
        # Add common phrases if we have text content
        if not cultural_reviews.empty and 'text' in cultural_reviews.columns:
            cultural_analysis['common_phrases'] = self.extract_common_phrases(cultural_reviews)
            
        # Add sentiment category counts if available
        if 'sentiment_category' in cultural_reviews.columns:
            cultural_analysis['positive_sentiment_count'] = len(cultural_reviews[cultural_reviews['sentiment_category'] == 'positive'])
            cultural_analysis['negative_sentiment_count'] = len(cultural_reviews[cultural_reviews['sentiment_category'] == 'negative'])
        
        return cultural_analysis

    def analyze_by_meal_type(self, df):
        """
        Analyze experiences by meal type (breakfast, lunch, dinner).
        """
        meal_analysis = {}
        
        for meal, keywords in self.meal_keywords.items():
            pattern = '|'.join(keywords)
            meal_reviews = df[df['text'].str.lower().str.contains(pattern, na=False)]
            
            if len(meal_reviews) > 0:
                meal_analysis[meal] = {
                    'review_count': len(meal_reviews)
                }
                
                # Add sentiment data if available
                if 'sentiment_score' in meal_reviews.columns:
                    meal_analysis[meal]['avg_sentiment'] = meal_reviews['sentiment_score'].mean()
                
                # Add top aspects if we can analyze them
                try:
                    meal_analysis[meal]['top_aspects'] = self.get_top_aspects(meal_reviews)
                except Exception as e:
                    print(f"Warning: Could not analyze top aspects for {meal}: {e}")
                
                # Add rating if available
                if 'rating' in meal_reviews.columns:
                    meal_analysis[meal]['avg_rating'] = meal_reviews['rating'].mean()
        
        return meal_analysis

    def analyze_price_value(self, df):
        """
        Analyze price-value perceptions in restaurant reviews.
        """
        price_reviews = df[df['mentions_price'] == 1]
        
        if len(price_reviews) == 0:
            return {'review_count': 0}
        
        # Calculate average sentiment for price-mentioning reviews if available
        avg_price_sentiment = None
        if 'sentiment_score' in price_reviews.columns:
            avg_price_sentiment = price_reviews['sentiment_score'].mean()
            
            # Categorize price sentiment
            price_reviews['price_sentiment'] = price_reviews['sentiment_score'].apply(
                lambda x: 'positive' if x > 0.1 else ('negative' if x < -0.1 else 'neutral')
            )
        else:
            # If no sentiment scores available, set all to neutral
            price_reviews['price_sentiment'] = 'neutral'
        
        # Count by price sentiment category
        price_sentiment_counts = price_reviews['price_sentiment'].value_counts().to_dict()
        
        # Extract common phrases in positive and negative price reviews
        pos_price_reviews = price_reviews[price_reviews['price_sentiment'] == 'positive']
        neg_price_reviews = price_reviews[price_reviews['price_sentiment'] == 'negative']
        
        # Extract common phrases if we have text content
        pos_phrases = {}
        neg_phrases = {}
        
        if 'text' in price_reviews.columns:
            if len(pos_price_reviews) > 0:
                try:
                    pos_phrases = self.extract_common_phrases(pos_price_reviews)
                except Exception as e:
                    print(f"Warning: Could not extract positive price phrases: {e}")
                    
            if len(neg_price_reviews) > 0:
                try:
                    neg_phrases = self.extract_common_phrases(neg_price_reviews)
                except Exception as e:
                    print(f"Warning: Could not extract negative price phrases: {e}")
        
        price_analysis = {
            'review_count': len(price_reviews),
            'avg_sentiment': avg_price_sentiment,
            'sentiment_distribution': price_sentiment_counts,
            'positive_phrases': pos_phrases,
            'negative_phrases': neg_phrases
        }
        
        return price_analysis

    def analyze_local_vs_international(self, df):
        """
        Compare mentions and sentiment for local vs. international cuisine.
        """
        local_reviews = df[df['cuisine_local_cuisine'] == 1]
        intl_reviews = df[df['cuisine_international'] == 1]
        
        comparison = {}
        
        if len(local_reviews) > 0:
            comparison['local'] = {
                'review_count': len(local_reviews)
            }
            
            # Add sentiment data if available
            if 'sentiment_score' in local_reviews.columns:
                comparison['local']['avg_sentiment'] = local_reviews['sentiment_score'].mean()
                
            # Add rating if available
            if 'rating' in local_reviews.columns:
                comparison['local']['avg_rating'] = local_reviews['rating'].mean()
                
            # Add sentiment category counts if available
            if 'sentiment_category' in local_reviews.columns:
                comparison['local']['positive_count'] = len(local_reviews[local_reviews['sentiment_category'] == 'positive'])
        
        if len(intl_reviews) > 0:
            comparison['international'] = {
                'review_count': len(intl_reviews)
            }
            
            # Add sentiment data if available
            if 'sentiment_score' in intl_reviews.columns:
                comparison['international']['avg_sentiment'] = intl_reviews['sentiment_score'].mean()
                
            # Add rating if available
            if 'rating' in intl_reviews.columns:
                comparison['international']['avg_rating'] = intl_reviews['rating'].mean()
                
            # Add sentiment category counts if available
            if 'sentiment_category' in intl_reviews.columns:
                comparison['international']['positive_count'] = len(intl_reviews[intl_reviews['sentiment_category'] == 'positive'])
        
        return comparison

    def extract_common_phrases(self, df, min_count=2):
        """
        Extract common phrases from a set of reviews.
        """
        if len(df) == 0 or 'processed_text' not in df.columns:
            return {}
            
        # Combine all processed text
        text = ' '.join(df['processed_text'].fillna(''))
        
        # Extract words and their frequencies
        words = text.split()
        word_freq = Counter(words)
        
        # Filter to words appearing at least min_count times
        common_phrases = {word: count for word, count in word_freq.items() 
                         if count >= min_count}
        
        return common_phrases

    def get_top_aspects(self, df):
        """
        Helper function to get most mentioned aspects in a set of reviews.
        """
        aspect_counts = {}
        for aspect in self.experience_aspects.keys():
            col = f'aspect_{aspect}'
            if col in df.columns:
                count = df[col].sum()
                if count > 0:
                    aspect_counts[aspect] = count
        
        return dict(sorted(aspect_counts.items(), key=lambda x: x[1], reverse=True))
        
    def run_analysis(self, df):
        """
        Run the complete restaurant analysis pipeline.
        
        Parameters:
        - df: DataFrame with all reviews
        
        Returns:
        - Dictionary with analysis results
        """
        print("\nRunning enhanced restaurant analysis...")
        
        # Create visualizations directory
        viz_dir = os.path.join(self.output_dir, 'visualizations')
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        
        # Filter to restaurant reviews
        restaurant_df = self.filter_restaurant_reviews(df)
        
        if restaurant_df.empty:
            print("No restaurant reviews to analyze")
            return {}
        
        # Run analyses
        cuisine_stats = self.analyze_cuisine_preferences(restaurant_df)
        cultural_elements = self.analyze_cultural_elements(restaurant_df)
        meal_analysis = self.analyze_by_meal_type(restaurant_df)
        price_value = self.analyze_price_value(restaurant_df)
        
        # Experience aspects analysis
        experience_aspects = {}
        for aspect in self.experience_aspects.keys():
            col = f'aspect_{aspect}'
            aspect_reviews = restaurant_df[restaurant_df[col] == 1]
            if len(aspect_reviews) > 0:
                experience_aspects[aspect] = {
                    'review_count': len(aspect_reviews)
                }
                
                # Add sentiment data if available
                if 'sentiment_score' in aspect_reviews.columns:
                    experience_aspects[aspect]['avg_sentiment'] = aspect_reviews['sentiment_score'].mean()
                    
                # Add sentiment category counts if available
                if 'sentiment_category' in aspect_reviews.columns:
                    experience_aspects[aspect]['positive_count'] = len(aspect_reviews[aspect_reviews['sentiment_category'] == 'positive'])
                    experience_aspects[aspect]['negative_count'] = len(aspect_reviews[aspect_reviews['sentiment_category'] == 'negative'])
        
        # Local vs international cuisine comparison
        local_vs_international = self.analyze_local_vs_international(restaurant_df)
        
        # Compile results
        results = {
            'cuisine_preferences': cuisine_stats,
            'cultural_elements': cultural_elements,
            'meal_analysis': meal_analysis,
            'price_value': price_value,
            'experience_aspects': experience_aspects,
            'local_vs_international': local_vs_international
        }
        
        # Generate recommendations
        recommendations = self.generate_restaurant_recommendations(restaurant_df, results)
        results['recommendations'] = recommendations
        
        # Generate visualizations
        self.visualize_cuisine_preferences(restaurant_df, cuisine_stats, viz_dir)
        self.visualize_experience_aspects(restaurant_df, viz_dir)
        self.visualize_price_sentiment(price_value, viz_dir)
        self.visualize_cultural_elements(cultural_elements, viz_dir)
        self.visualize_local_vs_international(local_vs_international, viz_dir)
        
        # Save detailed results
        self.save_results(results)
        
        print(f"Restaurant analysis complete. Visualizations saved to {viz_dir}")
        
        return results
        
    def save_results(self, results):
        """
        Save analysis results to file.
        """
        output_file = os.path.join(self.output_dir, 'restaurant_analysis.json')
        
        # Convert numpy values to Python natives for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Clean results for JSON serialization
        clean_results = json.loads(
            json.dumps(results, default=convert_to_serializable)
        )
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, indent=2)
        
        print(f"Detailed results saved to {output_file}")

    def generate_restaurant_recommendations(self, df, results):
        """
        Generate restaurant-specific recommendations based on analysis results.
        """
        recommendations = {
            "dining_experience": [],
            "cuisine": [],
            "price_value": [],
            "marketing": [],
            "cultural_elements": []
        }
        
        # Experience aspect recommendations
        if 'experience_aspects' in results:
            experience_data = results['experience_aspects']
            negative_aspects = {k: v for k, v in experience_data.items() 
                              if v.get('avg_sentiment', 0) < 0}
            
            for aspect, data in negative_aspects.items():
                # Only add recommendation if we have sentiment data
                if 'avg_sentiment' in data:
                    recommendations["dining_experience"].append(
                        f"Improve {aspect.replace('_', ' ')} which has negative sentiment ({data['avg_sentiment']:.2f})")
            
            # Top mentioned aspects
            top_aspects = sorted(experience_data.items(), key=lambda x: x[1].get('review_count', 0), reverse=True)[:3]
            for aspect, data in top_aspects:
                if data.get('review_count', 0) > 5:
                    recommendations["dining_experience"].append(
                        f"Focus on {aspect.replace('_', ' ')} as it's frequently mentioned ({data['review_count']} times)"
                    )
        
        # Cuisine recommendations
        if 'cuisine_preferences' in results and 'local_vs_international' in results:
            cuisine_data = results['cuisine_preferences']
            comparison = results['local_vs_international']
            
            # Local vs international balance
            if 'local' in comparison and 'international' in comparison:
                local_count = comparison['local']['review_count']
                intl_count = comparison['international']['review_count']
                
                if local_count < intl_count:
                    recommendations["cuisine"].append(
                        "Increase focus on authentic local Tongan cuisine as it's mentioned less than international options"
                    )
            
            # Identify least mentioned food categories
            least_mentioned = min(cuisine_data.items(), key=lambda x: x[1]['review_count']) if cuisine_data else None
            if least_mentioned and least_mentioned[1]['review_count'] < 5:
                recommendations["cuisine"].append(
                    f"Consider expanding {least_mentioned[0].replace('_', ' ')} options which are rarely mentioned"
                )
        
        # Price-value recommendations
        if 'price_value' in results:
            price_data = results['price_value']
            if price_data.get('review_count', 0) > 0:
                avg_sentiment = price_data.get('avg_sentiment', 0)
                
                if avg_sentiment is not None:
                    if avg_sentiment < 0:
                        recommendations["price_value"].append(
                            "Address price-value perception issues as sentiment is negative for price-mentioning reviews"
                        )
                    elif avg_sentiment > 0.3:
                        recommendations["price_value"].append(
                            "Leverage positive price-value perception in marketing materials"
                        )
        
        # Cultural elements recommendations
        if 'cultural_elements' in results:
            cultural_data = results['cultural_elements']
            if cultural_data.get('review_count', 0) > 0:
                cultural_sentiment = cultural_data.get('avg_sentiment')
                
                if cultural_sentiment is not None:
                    if cultural_sentiment > 0.3:
                        recommendations["cultural_elements"].append(
                            "Highlight traditional Tongan elements in dining experiences as they receive positive sentiment"
                        )
                    elif cultural_sentiment < 0:
                        recommendations["cultural_elements"].append(
                            "Review how cultural elements are incorporated into dining experiences to address negative sentiment"
                        )
            else:
                recommendations["cultural_elements"].append(
                    "Consider incorporating more Tongan cultural elements into dining experiences as they are rarely mentioned"
                )
        
        # Marketing recommendations
        # Find what visitors love about restaurants
        cuisine_data = results.get('cuisine_preferences', {})
        if cuisine_data:
            # Check if we have sentiment data
            cuisines_with_sentiment = {k: v for k, v in cuisine_data.items() if v.get('avg_sentiment') is not None}
            if cuisines_with_sentiment:
                # Get the cuisine type with highest sentiment
                top_sentiment_cuisine = max(cuisines_with_sentiment.items(), key=lambda x: x[1]['avg_sentiment'])
                if top_sentiment_cuisine[1]['avg_sentiment'] > 0.3:
                    recommendations["marketing"].append(
                        f"Highlight {top_sentiment_cuisine[0].replace('_', ' ')} in marketing as it receives the most positive sentiment"
                    )
        
        return recommendations

    def visualize_cuisine_preferences(self, df, cuisine_stats, viz_dir):
        """
        Generate visualizations for cuisine preferences.
        """
        # Extract data for plotting
        cuisine_counts = {k: v['review_count'] for k, v in cuisine_stats.items()}
        cuisine_sentiments = {k: v.get('avg_sentiment', 0) for k, v in cuisine_stats.items()}
        
        # Create side-by-side plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Use custom color palette
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f1c40f', '#9b59b6', '#1abc9c', '#f39c12', '#d35400']
        
        # Counts
        bars1 = ax1.bar(range(len(cuisine_counts)), 
                       list(cuisine_counts.values()),
                       color=colors[:len(cuisine_counts)])
        ax1.set_title('Reviews by Cuisine Type')
        ax1.set_xticks(range(len(cuisine_counts)))
        ax1.set_xticklabels([k.replace('_', ' ').title() for k in cuisine_counts.keys()], 
                           rotation=45, ha='right')
        
        # Add count labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        # Sentiments
        bars2 = ax2.bar(range(len(cuisine_sentiments)), 
                       list(cuisine_sentiments.values()),
                       color=colors[:len(cuisine_sentiments)])
        ax2.set_title('Average Sentiment by Cuisine Type')
        ax2.set_xticks(range(len(cuisine_sentiments)))
        ax2.set_xticklabels([k.replace('_', ' ').title() for k in cuisine_sentiments.keys()], 
                           rotation=45, ha='right')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.2)
        
        # Add sentiment labels
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'cuisine_analysis.png'), dpi=300)
        plt.close()

    def visualize_experience_aspects(self, df, viz_dir):
        """
        Generate visualizations for dining experience aspects.
        """
        # Calculate aspect mentions and sentiment
        aspect_data = {}
        for aspect in self.experience_aspects.keys():
            col = f'aspect_{aspect}'
            aspect_reviews = df[df[col] == 1]
            if len(aspect_reviews) > 0:
                # Check if sentiment_score column exists before calculating mean
                sentiment_value = 0
                if 'sentiment_score' in aspect_reviews.columns:
                    # Filter out null values and calculate mean only if there are values
                    sentiment_reviews = aspect_reviews.dropna(subset=['sentiment_score'])
                    if not sentiment_reviews.empty:
                        sentiment_value = sentiment_reviews['sentiment_score'].mean()
                
                aspect_data[aspect] = {
                    'count': len(aspect_reviews),
                    'sentiment': sentiment_value
                }
        
        if not aspect_data:
            return
        
        # Sort by count
        aspect_data = dict(sorted(aspect_data.items(), key=lambda x: x[1]['count'], reverse=True))
        
        # Prepare data for plotting
        aspects = list(aspect_data.keys())
        counts = [aspect_data[a]['count'] for a in aspects]
        sentiments = [aspect_data[a]['sentiment'] for a in aspects]
        
        # Create side-by-side plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Use custom color palette
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f1c40f', '#9b59b6', '#1abc9c']
        
        # Counts
        bars1 = ax1.bar(range(len(aspects)), counts, color=colors[:len(aspects)])
        ax1.set_title('Mentions of Dining Experience Aspects')
        ax1.set_xticks(range(len(aspects)))
        ax1.set_xticklabels([a.replace('_', ' ').title() for a in aspects], rotation=45, ha='right')
        
        # Add count labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        # Sentiments
        bars2 = ax2.bar(range(len(aspects)), sentiments, 
                       color=colors[:len(aspects)])
        ax2.set_title('Sentiment for Dining Experience Aspects')
        ax2.set_xticks(range(len(aspects)))
        ax2.set_xticklabels([a.replace('_', ' ').title() for a in aspects], rotation=45, ha='right')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.2)
        
        # Add sentiment labels
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'dining_aspects.png'), dpi=300)
        plt.close()

    def visualize_price_sentiment(self, price_data, viz_dir):
        """
        Generate visualizations for price-value sentiment.
        """
        if price_data.get('review_count', 0) <= 0:
            return
            
        sentiment_distribution = price_data.get('sentiment_distribution', {})
        if not sentiment_distribution:
            return
            
        # Create pie chart of price sentiment
        plt.figure(figsize=(10, 6))
        labels = list(sentiment_distribution.keys())
        sizes = list(sentiment_distribution.values())
        colors = ['#5CB85C', '#F0AD4E', '#D9534F']
        plt.pie(sizes, labels=labels, 
                autopct='%1.1f%%', startangle=90, colors=colors)
        plt.axis('equal')
        plt.title('Sentiment in Price-Mentioning Restaurant Reviews')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'price_sentiment_distribution.png'))
        plt.close()
        
        # Create word clouds for positive and negative price mentions
        positive_phrases = price_data.get('positive_phrases', {})
        if positive_phrases:
            wc = WordCloud(width=800, height=400, background_color='white',
                          max_words=50, contour_width=3, contour_color='steelblue')
            wc.generate_from_frequencies(positive_phrases)
            
            plt.figure(figsize=(10, 6))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            plt.title('Common Words in Positive Price-Value Reviews')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'positive_price_wordcloud.png'))
            plt.close()
        
        negative_phrases = price_data.get('negative_phrases', {})
        if negative_phrases:
            wc = WordCloud(width=800, height=400, background_color='white',
                          max_words=50, contour_width=3, contour_color='darkred')
            wc.generate_from_frequencies(negative_phrases)
            
            plt.figure(figsize=(10, 6))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            plt.title('Common Words in Negative Price-Value Reviews')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'negative_price_wordcloud.png'))
            plt.close()

    def visualize_cultural_elements(self, cultural_data, viz_dir):
        """
        Generate visualization for cultural elements in reviews.
        """
        if cultural_data.get('review_count', 0) <= 0:
            return
            
        common_phrases = cultural_data.get('common_phrases', {})
        if not common_phrases:
            return
            
        # Create word cloud for cultural elements
        wc = WordCloud(width=800, height=400, background_color='white',
                      colormap='viridis', max_words=100)
        wc.generate_from_frequencies(common_phrases)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title('Common Words in Cultural Dining Reviews')
        plt.tight_layout(pad=0)
        plt.savefig(os.path.join(viz_dir, 'cultural_wordcloud.png'), dpi=300)
        plt.close()

    def visualize_local_vs_international(self, comparison_data, viz_dir):
        """
        Generate visualizations comparing local and international cuisine.
        """
        if not comparison_data or not all(k in comparison_data for k in ['local', 'international']):
            return
            
        local = comparison_data['local']
        intl = comparison_data['international']
        
        # Check if we have the required data for visualization
        if 'review_count' not in local or 'review_count' not in intl:
            return
            
        # Create comparison charts for counts only if we don't have sentiment data
        if 'avg_sentiment' not in local or 'avg_sentiment' not in intl:
            # Only create a single chart for counts
            plt.figure(figsize=(10, 6))
            
            labels = ['Local Tongan Cuisine', 'International Cuisine']
            counts = [local['review_count'], intl['review_count']]
            
            # Review counts comparison
            bars = plt.bar([0, 1], counts, color=['#1B9E77', '#D95F02'])
            plt.title('Number of Reviews by Cuisine Type')
            plt.xticks([0, 1], labels)
            plt.ylabel('Number of Reviews')
            
            # Add text labels
            for i, v in enumerate(counts):
                plt.text(i, v - 0.1 * max(counts), f"{v}", ha='center', color='white', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'cuisine_comparison.png'), dpi=300)
            plt.close()
            return
            
        # If we get here, we have both count and sentiment data
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        labels = ['Local Tongan Cuisine', 'International Cuisine']
        counts = [local['review_count'], intl['review_count']]
        sentiments = [local['avg_sentiment'], intl['avg_sentiment']]
        
        # Review counts comparison
        bars1 = ax1.bar([0, 1], counts, color=['#1B9E77', '#D95F02'])
        ax1.set_title('Number of Reviews by Cuisine Type')
        ax1.set_xticks([0, 1])
        ax1.set_xticklabels(labels)
        ax1.set_ylabel('Number of Reviews')
        
        # Add text labels
        for i, v in enumerate(counts):
            ax1.text(i, v - 0.1 * max(counts), f"{v}", ha='center', color='white', fontweight='bold')
        
        # Sentiment comparison
        bars2 = ax2.bar([0, 1], sentiments, color=['#1B9E77', '#D95F02'])
        ax2.set_title('Average Sentiment by Cuisine Type')
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels(labels)
        ax2.set_ylabel('Average Sentiment Score (-1 to 1)')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add text labels
        for i, v in enumerate(sentiments):
            ax2.text(i, v - 0.05 if v > 0 else v + 0.05, 
                   f"{v:.2f}", ha='center', 
                   color='white' if v > 0 else 'black', 
                   fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'cuisine_comparison.png'), dpi=300)
        plt.close()
    
    def run_island_analysis(self, df):
        """
        Run restaurant analysis segmented by island.
        
        Parameters:
        - df: DataFrame with review data that includes island classification
        
        Returns:
        - Dictionary with island-specific analysis results
        """
        print("\nRunning restaurant analysis by island...")
        
        if 'island_category' not in df.columns:
            print("Error: DataFrame does not contain island classification.")
            return {}
        
        # Filter to only restaurant reviews
        restaurant_df = self.filter_restaurant_reviews(df)
        
        if restaurant_df.empty:
            print("No restaurant reviews to analyze by island")
            return {}
        
        # Get unique islands with more than 5 restaurant reviews
        island_counts = restaurant_df['island_category'].value_counts()
        islands_to_analyze = island_counts[island_counts >= 5].index.tolist()
        
        if not islands_to_analyze:
            print("No islands have sufficient restaurant reviews for analysis")
            return {}
        
        print(f"Analyzing restaurant reviews for {len(islands_to_analyze)} islands: {', '.join(islands_to_analyze)}")
        
        # Analyze each island
        island_results = {}
        
        for island in islands_to_analyze:
            print(f"\nAnalyzing restaurant reviews for {island}...")
            island_df = restaurant_df[restaurant_df['island_category'] == island]
            
            # Run analysis for this island
            island_cuisine_stats = self.analyze_cuisine_preferences(island_df)
            island_cultural_elements = self.analyze_cultural_elements(island_df)
            island_meal_analysis = self.analyze_by_meal_type(island_df)
            island_price_value = self.analyze_price_value(island_df)
            
            # Experience aspects analysis
            experience_aspects = {}
            for aspect in self.experience_aspects.keys():
                col = f'aspect_{aspect}'
                aspect_reviews = island_df[island_df[col] == 1]
                if len(aspect_reviews) > 0:
                    experience_aspects[aspect] = {
                        'review_count': len(aspect_reviews),
                        'avg_sentiment': aspect_reviews['sentiment_score'].mean() if 'sentiment_score' in aspect_reviews.columns else 0,
                        'positive_count': len(aspect_reviews[aspect_reviews['sentiment_category'] == 'positive']) if 'sentiment_category' in aspect_reviews.columns else 0,
                        'negative_count': len(aspect_reviews[aspect_reviews['sentiment_category'] == 'negative']) if 'sentiment_category' in aspect_reviews.columns else 0
                    }
            
            # Local vs international cuisine comparison
            local_vs_international = self.analyze_local_vs_international(island_df)
            
            # Store results for this island
            island_results[island] = {
                'review_count': len(island_df),
                'cuisine_preferences': island_cuisine_stats,
                'cultural_elements': island_cultural_elements,
                'meal_analysis': island_meal_analysis,
                'price_value': island_price_value,
                'experience_aspects': experience_aspects,
                'local_vs_international': local_vs_international,
            }
            
            # Generate island-specific visualizations if sufficient data
            if len(island_df) >= 10:
                island_viz_dir = os.path.join(self.island_dir, island)
                if not os.path.exists(island_viz_dir):
                    os.makedirs(island_viz_dir)
                
                # Create island-specific visualizations
                if island_cuisine_stats:
                    self.visualize_cuisine_preferences(island_df, island_cuisine_stats, island_viz_dir)
                if experience_aspects:
                    self.visualize_experience_aspects(island_df, island_viz_dir)
                if island_price_value and island_price_value.get('review_count', 0) > 0:
                    self.visualize_price_sentiment(island_price_value, island_viz_dir)
                if island_cultural_elements and island_cultural_elements.get('review_count', 0) > 0:
                    self.visualize_cultural_elements(island_cultural_elements, island_viz_dir)
                if local_vs_international and all(k in local_vs_international for k in ['local', 'international']):
                    self.visualize_local_vs_international(local_vs_international, island_viz_dir)
        
        # Generate cross-island comparisons
        self.visualize_island_comparisons(island_results)
        
        # Save island analysis results
        self._save_island_results(island_results)
        
        print(f"\nIsland-based restaurant analysis complete. Results saved to {self.island_dir}")
        
        return island_results
    
    def visualize_island_comparisons(self, island_results):
        """
        Generate visualizations comparing restaurant metrics across islands.
        
        Parameters:
        - island_results: Dictionary with island-specific analysis results
        """
        if not island_results:
            return
        
        # Create comparisons directory
        comparison_dir = os.path.join(self.island_dir, 'comparisons')
        if not os.path.exists(comparison_dir):
            os.makedirs(comparison_dir)
        
        # Prepare data for comparisons
        islands = list(island_results.keys())
        
        # 1. Reviews per island
        review_counts = [island_results[island]['review_count'] for island in islands]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(islands, review_counts, color='skyblue')
        plt.title('Restaurant Reviews by Island', fontsize=15)
        plt.xlabel('Island', fontsize=12)
        plt.ylabel('Number of Reviews', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Add count labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, 'restaurant_reviews_by_island.png'), dpi=300)
        plt.close()
        
        # 2. Cuisine preferences comparison
        cuisine_data = {}
        for island in islands:
            cuisine_stats = island_results[island].get('cuisine_preferences', {})
            if cuisine_stats:
                # Collect cuisine counts for this island
                for cuisine, stats in cuisine_stats.items():
                    if cuisine not in cuisine_data:
                        cuisine_data[cuisine] = {}
                    cuisine_data[cuisine][island] = stats.get('review_count', 0)
        
        # Only process if we have cuisine data
        if cuisine_data:
            # Create a DataFrame for easier plotting
            cuisine_df = pd.DataFrame(cuisine_data)
            
            # Fill NAs with 0
            cuisine_df = cuisine_df.fillna(0)
            
            # Plot stacked bar chart
            plt.figure(figsize=(14, 8))
            cuisine_df.plot(kind='bar', stacked=True, colormap='viridis', figsize=(14, 8))
            plt.title('Cuisine Preferences by Island', fontsize=15)
            plt.xlabel('Island', fontsize=12)
            plt.ylabel('Number of Reviews', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Cuisine Type', loc='upper right')
            plt.tight_layout()
            plt.savefig(os.path.join(comparison_dir, 'cuisine_preferences_by_island.png'), dpi=300)
            plt.close()
            
            # Plot normalized stacked bar chart (proportions)
            plt.figure(figsize=(14, 8))
            cuisine_prop_df = cuisine_df.div(cuisine_df.sum(axis=1), axis=0)
            cuisine_prop_df.plot(kind='bar', stacked=True, colormap='viridis', figsize=(14, 8))
            plt.title('Proportion of Cuisine Types by Island', fontsize=15)
            plt.xlabel('Island', fontsize=12)
            plt.ylabel('Proportion', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Cuisine Type', loc='upper right')
            plt.tight_layout()
            plt.savefig(os.path.join(comparison_dir, 'cuisine_proportions_by_island.png'), dpi=300)
            plt.close()
        
        # 3. Price sentiment comparison
        price_data = {}
        for island in islands:
            price_value = island_results[island].get('price_value', {})
            if price_value and price_value.get('review_count', 0) > 0:
                price_data[island] = {
                    'review_count': price_value.get('review_count', 0),
                    'avg_sentiment': price_value.get('avg_sentiment', 0)
                }
        
        # Only process if we have price data
        if price_data:
            # Extract data for plotting
            price_islands = list(price_data.keys())
            price_counts = [price_data[island]['review_count'] for island in price_islands]
            price_sentiments = [price_data[island]['avg_sentiment'] for island in price_islands]
            
            # Create side-by-side plots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Counts
            bars1 = ax1.bar(price_islands, price_counts, color='skyblue')
            ax1.set_title('Price-Mentioning Reviews by Island')
            ax1.set_xlabel('Island')
            ax1.set_ylabel('Number of Reviews')
            ax1.set_xticklabels(price_islands, rotation=45, ha='right')
            
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')
            
            # Sentiments
            bars2 = ax2.bar(price_islands, price_sentiments, color='orange')
            ax2.set_title('Price Sentiment by Island')
            ax2.set_xlabel('Island')
            ax2.set_ylabel('Average Sentiment')
            ax2.set_xticklabels(price_islands, rotation=45, ha='right')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.2)
            
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom' if height >= 0 else 'top')
            
            plt.tight_layout()
            plt.savefig(os.path.join(comparison_dir, 'price_sentiment_by_island.png'), dpi=300)
            plt.close()
        
        # 4. Experience aspects comparison
        aspects_data = {}
        for island in islands:
            experience_aspects = island_results[island].get('experience_aspects', {})
            if experience_aspects:
                for aspect, stats in experience_aspects.items():
                    if aspect not in aspects_data:
                        aspects_data[aspect] = {}
                    aspects_data[aspect][island] = stats.get('review_count', 0)
        
        # Only process if we have aspects data
        if aspects_data:
            # Create a DataFrame for easier plotting
            aspects_df = pd.DataFrame(aspects_data)
            
            # Fill NAs with 0
            aspects_df = aspects_df.fillna(0)
            
            # Plot stacked bar chart
            plt.figure(figsize=(14, 8))
            aspects_df.plot(kind='bar', stacked=True, colormap='viridis', figsize=(14, 8))
            plt.title('Dining Experience Aspects by Island', fontsize=15)
            plt.xlabel('Island', fontsize=12)
            plt.ylabel('Number of Reviews', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Experience Aspect', loc='upper right')
            plt.tight_layout()
            plt.savefig(os.path.join(comparison_dir, 'dining_aspects_by_island.png'), dpi=300)
            plt.close()
            
            # Plot normalized stacked bar chart (proportions)
            plt.figure(figsize=(14, 8))
            aspects_prop_df = aspects_df.div(aspects_df.sum(axis=1), axis=0)
            aspects_prop_df.plot(kind='bar', stacked=True, colormap='viridis', figsize=(14, 8))
            plt.title('Proportion of Dining Experience Aspects by Island', fontsize=15)
            plt.xlabel('Island', fontsize=12)
            plt.ylabel('Proportion', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Experience Aspect', loc='upper right')
            plt.tight_layout()
            plt.savefig(os.path.join(comparison_dir, 'dining_aspects_proportions_by_island.png'), dpi=300)
            plt.close()
        
        # 5. Local vs International comparison across islands
        local_intl_data = {'local': {}, 'international': {}}
        for island in islands:
            local_vs_intl = island_results[island].get('local_vs_international', {})
            if local_vs_intl and all(k in local_vs_intl for k in ['local', 'international']):
                local_count = local_vs_intl['local'].get('review_count', 0)
                intl_count = local_vs_intl['international'].get('review_count', 0)
                
                local_intl_data['local'][island] = local_count
                local_intl_data['international'][island] = intl_count
        
        # Only process if we have local vs international data
        if local_intl_data['local'] and local_intl_data['international']:
            # Create a DataFrame for easier plotting
            local_intl_df = pd.DataFrame(local_intl_data)
            
            # Fill NAs with 0
            local_intl_df = local_intl_df.fillna(0)
            
            # Plot stacked bar chart
            plt.figure(figsize=(14, 8))
            local_intl_df.plot(kind='bar', stacked=True, color=['#1B9E77', '#D95F02'], figsize=(14, 8))
            plt.title('Local vs International Cuisine by Island', fontsize=15)
            plt.xlabel('Island', fontsize=12)
            plt.ylabel('Number of Reviews', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Cuisine Type', loc='upper right')
            plt.tight_layout()
            plt.savefig(os.path.join(comparison_dir, 'local_vs_international_by_island.png'), dpi=300)
            plt.close()
            
            # Plot normalized stacked bar chart (proportions)
            plt.figure(figsize=(14, 8))
            local_intl_prop_df = local_intl_df.div(local_intl_df.sum(axis=1), axis=0)
            local_intl_prop_df.plot(kind='bar', stacked=True, color=['#1B9E77', '#D95F02'], figsize=(14, 8))
            plt.title('Proportion of Local vs International Cuisine by Island', fontsize=15)
            plt.xlabel('Island', fontsize=12)
            plt.ylabel('Proportion', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Cuisine Type', loc='upper right')
            plt.tight_layout()
            plt.savefig(os.path.join(comparison_dir, 'local_vs_international_proportions_by_island.png'), dpi=300)
            plt.close()
        
        print(f"Island comparison visualizations saved to {comparison_dir}")
    
    def _save_island_results(self, island_results):
        """
        Save island analysis results to JSON and Excel.
        
        Parameters:
        - island_results: Dictionary with island-specific analysis results
        """
        # Save to JSON
        json_path = os.path.join(self.island_dir, 'restaurant_island_analysis.json')
        
        # Convert to serializable format
        serializable_results = self._make_serializable(island_results)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Island analysis results saved to {json_path}")
        
        # Try to save to Excel
        try:
            excel_path = os.path.join(self.island_dir, 'restaurant_island_comparison.xlsx')
            
            # Create summary DataFrame
            summary_data = []
            for island, data in island_results.items():
                row = {
                    'Island': island,
                    'Total Restaurant Reviews': data['review_count'],
                    'Local Cuisine Mentions': sum(stats.get('review_count', 0) for stats in 
                                              data.get('cuisine_preferences', {}).values() if 'local' in stats),
                    'International Cuisine Mentions': sum(stats.get('review_count', 0) for stats in 
                                                     data.get('cuisine_preferences', {}).values() if 'international' in stats),
                    'Cultural Elements Mentions': data.get('cultural_elements', {}).get('review_count', 0),
                    'Price Mentions': data.get('price_value', {}).get('review_count', 0),
                    'Price Sentiment': data.get('price_value', {}).get('avg_sentiment', 0),
                }
                summary_data.append(row)
            
            summary_df = pd.DataFrame(summary_data)
            
            # Write to Excel
            with pd.ExcelWriter(excel_path) as writer:
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Add cuisine preferences sheet if data available
                cuisine_data = {}
                for island in island_results:
                    cuisine_stats = island_results[island].get('cuisine_preferences', {})
                    if cuisine_stats:
                        for cuisine, stats in cuisine_stats.items():
                            if cuisine not in cuisine_data:
                                cuisine_data[cuisine] = {}
                            cuisine_data[cuisine][island] = stats.get('review_count', 0)
                
                if cuisine_data:
                    cuisine_df = pd.DataFrame(cuisine_data).fillna(0)
                    cuisine_df.to_excel(writer, sheet_name='Cuisine Preferences')
                
                # Add experience aspects sheet if data available
                aspects_data = {}
                for island in island_results:
                    aspects = island_results[island].get('experience_aspects', {})
                    if aspects:
                        for aspect, stats in aspects.items():
                            if aspect not in aspects_data:
                                aspects_data[aspect] = {}
                            aspects_data[aspect][island] = stats.get('review_count', 0)
                
                if aspects_data:
                    aspects_df = pd.DataFrame(aspects_data).fillna(0)
                    aspects_df.to_excel(writer, sheet_name='Experience Aspects')
            
            print(f"Island comparison data saved to Excel: {excel_path}")
        except Exception as e:
            print(f"Could not save to Excel: {str(e)}")
            print("Consider installing openpyxl for Excel support: pip install openpyxl")
    
    def _make_serializable(self, obj):
        """Helper method to convert objects to JSON serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj