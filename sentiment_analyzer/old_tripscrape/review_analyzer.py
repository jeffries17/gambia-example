import json
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

# Download required NLTK resources
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

class GambiaTripAdvisorAnalyzer:
    def __init__(self, json_file_path, min_year=None):
        """
        Initialize the analyzer with review data
        
        Parameters:
        json_file_path (str): Path to the JSON file with review data
        min_year (int, optional): Minimum year to include reviews from. If None, includes all reviews.
        """
        self.json_file_path = json_file_path
        self.min_year = min_year
        self.reviews = self._load_and_filter_data()
        self.sia = SentimentIntensityAnalyzer()
        
    def _load_and_filter_data(self):
        """Load and filter reviews based on year (if min_year is specified)"""
        with open(self.json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # TripAdvisor exports should be directly in a list format
        filtered_reviews = []
        
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict) and 'reviews' in data:
            items = data['reviews']
        elif isinstance(data, dict) and 'data' in data and 'reviews' in data['data']:
            items = data['data']['reviews']
        else:
            # Log error and return empty list
            print(f"Warning: Cannot identify reviews in JSON structure. Returning empty list.")
            return []
            
        original_count = len(items)
        
        # If no year filter specified, return all reviews
        if self.min_year is None:
            print(f"Loaded all {original_count} reviews")
            return items
            
        # Otherwise, filter by year
        for review in items:
            # TripAdvisor review structure uses 'publishedDate' field
            date_str = None
            if 'publishedDate' in review:
                date_str = review['publishedDate']
            else:
                # Fallback to other possible date fields
                for field in ['date', 'review_date', 'published_date', 'created_at', 'timestamp']:
                    if field in review:
                        date_str = review[field]
                        break
            
            # Skip if no date found
            if not date_str:
                continue
                
            # Try multiple date formats
            parsed_date = None
            for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%d/%m/%Y', 
                        '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S']:
                try:
                    parsed_date = datetime.strptime(date_str, fmt)
                    break
                except ValueError:
                    continue
            
            # If date couldn't be parsed, try to extract year using regex
            if not parsed_date:
                year_match = re.search(r'20\d\d', date_str)
                if year_match:
                    year = int(year_match.group(0))
                    if year >= self.min_year:
                        filtered_reviews.append(review)
                continue
                
            # Check if year meets the minimum year requirement
            if parsed_date and parsed_date.year >= self.min_year:
                filtered_reviews.append(review)
        
        print(f"Found {original_count} total reviews")
        print(f"Loaded {len(filtered_reviews)} reviews from {self.min_year} onwards")
        return filtered_reviews
    
    def get_basic_stats(self):
        """Calculate basic statistics about the reviews"""
        if not self.reviews:
            return {
                'total_reviews': 0,
                'avg_rating': 0,
                'rating_distribution': {},
                'reviews_by_year': {},
                'reviews_by_category': {},
                'reviews_by_traveler_type': {}
            }
            
        # Extract ratings from TripAdvisor data (consistently uses 'rating' field)
        ratings = []
        for r in self.reviews:
            if 'rating' in r:
                ratings.append(r['rating'])
            elif 'stars' in r:
                ratings.append(r['stars'])
            elif 'score' in r:
                ratings.append(r['score'])
                
        avg_rating = sum(ratings) / len(ratings) if ratings else 0
        
        # TripAdvisor data structure specifics
        hotel_name = self.reviews[0].get('placeInfo', {}).get('name', 'Unknown Hotel') if self.reviews else 'Unknown Hotel'
        
        # Get traveler types - TripAdvisor uses 'tripType' field
        traveler_types = []
        for r in self.reviews:
            if 'tripType' in r:
                traveler_types.append(r['tripType'])
            elif 'traveler_type' in r:
                traveler_types.append(r['traveler_type'])
            else:
                traveler_types.append('Not specified')
                
        stats = {
            'total_reviews': len(self.reviews),
            'avg_rating': avg_rating,
            'hotel_name': hotel_name,
            'rating_distribution': Counter(ratings),
            'reviews_by_traveler_type': Counter(traveler_types)
        }
        
        # Extract years from dates - TripAdvisor uses 'publishedDate'
        years = []
        for r in self.reviews:
            date_field = 'publishedDate' if 'publishedDate' in r else next((f for f in ['date', 'review_date', 'published_date'] if f in r), None)
            
            if date_field and r[date_field]:
                try:
                    # Try multiple date formats
                    for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%d/%m/%Y', 
                              '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S']:
                        try:
                            date = datetime.strptime(r[date_field], fmt)
                            years.append(date.year)
                            break
                        except ValueError:
                            continue
                except:
                    # If date parsing fails, try to extract year using regex
                    if isinstance(r[date_field], str):
                        year_match = re.search(r'20\d\d', r[date_field])
                        if year_match:
                            years.append(int(year_match.group(0)))
                                
        stats['reviews_by_year'] = Counter(years)
        
        # Handle the fact that TripAdvisor's 'category' might be embedded differently
        categories = []
        for r in self.reviews:
            if 'category' in r:
                categories.append(r['category'])
            elif 'placeInfo' in r and 'name' in r['placeInfo']:
                categories.append('Hotel: ' + r['placeInfo']['name'])
            else:
                categories.append('Not specified')
                
        stats['reviews_by_category'] = Counter(categories)
        
        return stats
    
    def perform_sentiment_analysis(self):
        """Analyze sentiment in review text"""
        if not self.reviews:
            return {
                'overall_sentiment': {
                    'positive': 0,
                    'negative': 0,
                    'neutral': 0,
                    'compound': 0
                },
                'sentiment_by_category': {}
            }

        # For TripAdvisor data, the field is consistently 'text'            
        for review in self.reviews:
            # Find the review text - TripAdvisor uses 'text' field
            if 'text' in review and review['text']:
                review_text = review['text']
            else:
                # Fallback to other possible text fields
                review_text = ""
                review_text_fields = ['review_text', 'content', 'review', 'comment', 'description']
                for field in review_text_fields:
                    if field in review and review[field]:
                        review_text = review[field]
                        break
                    
            if not review_text:
                # If no review text field is found, skip sentiment analysis
                review['sentiment'] = {'pos': 0, 'neg': 0, 'neu': 1, 'compound': 0}
                continue
                
            sentiment = self.sia.polarity_scores(review_text)
            review['sentiment'] = sentiment
            
        # Overall sentiment stats
        avg_sentiment = {
            'positive': sum(r['sentiment']['pos'] for r in self.reviews) / len(self.reviews),
            'negative': sum(r['sentiment']['neg'] for r in self.reviews) / len(self.reviews),
            'neutral': sum(r['sentiment']['neu'] for r in self.reviews) / len(self.reviews),
            'compound': sum(r['sentiment']['compound'] for r in self.reviews) / len(self.reviews)
        }
        
        # Sentiment by category
        categories = set(r.get('category', 'Not specified') for r in self.reviews)
        sentiment_by_category = {}
        
        for category in categories:
            category_reviews = [r for r in self.reviews if r.get('category', 'Not specified') == category]
            if category_reviews:
                sentiment_by_category[category] = {
                    'count': len(category_reviews),
                    'avg_compound': sum(r['sentiment']['compound'] for r in category_reviews) / len(category_reviews)
                }
        
        return {
            'overall_sentiment': avg_sentiment,
            'sentiment_by_category': sentiment_by_category
        }
    
    def extract_key_themes(self, n_words=20):
        """Extract key themes/topics from reviews using simple word frequency"""
        if not self.reviews:
            return []
        
        try:
            from nltk.corpus import stopwords
            from nltk.tokenize import word_tokenize
            import nltk
            
            # Make sure all necessary NLTK packages are downloaded
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                print("Downloading necessary NLTK data...")
                nltk.download('punkt')
                
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords')
                
            stop_words = set(stopwords.words('english'))
            # Expanded custom stopwords specifically for hotel reviews
            custom_stops = {
                'hotel', 'place', 'gambia', 'the', 'a', 'an', 'it', 'they', 'we', 'i', 'this', 'that',
                'with', 'have', 'from', 'were', 'would', 'their', 'room', 'rooms', 'very', 'beach', 
                'african', 'princess', 'there', 'which', 'stay', 'our', 'for', 'was', 'you', 'and',
                'but', 'had', 'all', 'day', 'when', 'night', 'one', 'two', 'out', 'get', 'got', 'just',
                'back', 'time', 'some', 'every', 'will', 'been', 'your', 'also', 'next', 'like', 'can',
                'holiday', 'during', 'first', 'good', 'great'
            }
            stop_words.update(custom_stops)
            
            all_words = []
            for review in self.reviews:
                # TripAdvisor consistently uses 'text' field
                if 'text' in review and review['text']:
                    review_text = review['text']
                else:
                    # Fallback to other fields
                    review_text = ""
                    review_text_fields = ['review_text', 'content', 'review', 'comment', 'description']
                    for field in review_text_fields:
                        if field in review and review[field]:
                            review_text = review[field]
                            break
                        
                if not review_text:
                    continue
                    
                # Simple word tokenization with regex as fallback if NLTK fails
                try:
                    words = word_tokenize(review_text.lower())
                except:
                    # Fallback to simple regex tokenization
                    words = re.findall(r'\b\w+\b', review_text.lower())
                    
                filtered_words = [word for word in words if word.isalpha() and word not in stop_words and len(word) > 3]
                all_words.extend(filtered_words)
            
            word_freq = Counter(all_words)
            return word_freq.most_common(n_words)
            
        except Exception as e:
            print(f"Warning: Error in key theme extraction: {str(e)}")
            print("Continuing with analysis without key themes...")
            return []
    
    def generate_visualizations(self, output_dir='.'):
        """Generate visualization charts for the analysis results"""
        if not self.reviews:
            print("No reviews to visualize. Skipping visualization generation.")
            return
            
        stats = self.get_basic_stats()
        sentiment = self.perform_sentiment_analysis()
        
        # 1. Rating distribution
        if stats['rating_distribution']:
            plt.figure(figsize=(10, 6))
            ratings = sorted(stats['rating_distribution'].keys())
            counts = [stats['rating_distribution'][r] for r in ratings]
            sns.barplot(x=ratings, y=counts)
            plt.title('Distribution of Ratings')
            plt.xlabel('Rating')
            plt.ylabel('Count')
            plt.savefig(f'{output_dir}/rating_distribution.png')
            plt.close()
        
        # 2. Reviews by year
        if stats['reviews_by_year']:
            plt.figure(figsize=(10, 6))
            years = sorted(stats['reviews_by_year'].keys())
            sns.barplot(x=years, y=[stats['reviews_by_year'][y] for y in years])
            plt.title('Reviews by Year')
            plt.xlabel('Year')
            plt.ylabel('Count')
            plt.savefig(f'{output_dir}/reviews_by_year.png')
            plt.close()
        
        # 3. Sentiment by category
        if sentiment['sentiment_by_category']:
            plt.figure(figsize=(12, 8))
            categories = list(sentiment['sentiment_by_category'].keys())
            sentiment_scores = [sentiment['sentiment_by_category'][c]['avg_compound'] for c in categories]
            review_counts = [sentiment['sentiment_by_category'][c]['count'] for c in categories]
            
            # Sort by count
            sorted_indices = sorted(range(len(review_counts)), key=lambda i: review_counts[i], reverse=True)
            categories = [categories[i] for i in sorted_indices]
            sentiment_scores = [sentiment_scores[i] for i in sorted_indices]
            review_counts = [review_counts[i] for i in sorted_indices]
            
            # Plot
            sentiment_colors = ['green' if s > 0.05 else 'red' if s < -0.05 else 'gray' for s in sentiment_scores]
            plt.figure(figsize=(14, 8))
            bars = plt.bar(categories, sentiment_scores, color=sentiment_colors)
            
            # Add review counts as text
            for i, (bar, count) in enumerate(zip(bars, review_counts)):
                plt.text(i, bar.get_height() + 0.02 if bar.get_height() > 0 else -0.08, 
                        f'n={count}', ha='center', va='bottom', rotation=0)
            
            plt.title('Sentiment Score by Category')
            plt.xlabel('Category')
            plt.ylabel('Average Sentiment Score')
            plt.xticks(rotation=45, ha='right')
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/sentiment_by_category.png')
            plt.close()
        
        # 4. Word cloud for key themes
        themes = self.extract_key_themes(n_words=100)
        if themes:
            try:
                from wordcloud import WordCloud
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(themes))
                
                plt.figure(figsize=(10, 6))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title('Key Themes in Reviews')
                plt.savefig(f'{output_dir}/themes_wordcloud.png')
                plt.close()
            except ImportError:
                print("WordCloud package not installed. Skipping word cloud visualization.")
            except Exception as e:
                print(f"Error generating word cloud: {str(e)}")
    
    def export_to_csv(self, output_file):
        """Export processed review data to CSV for further analysis"""
        if not self.reviews:
            print("No reviews to export. Creating empty CSV file.")
            pd.DataFrame().to_csv(output_file, index=False)
            return
            
        # Convert to pandas DataFrame
        df = pd.json_normalize(self.reviews)
        df.to_csv(output_file, index=False)
        print(f"Exported processed data to {output_file}")

    def generate_report(self, output_file):
        """Generate a comprehensive analysis report in HTML format"""
        stats = self.get_basic_stats()
        sentiment = self.perform_sentiment_analysis()
        themes = self.extract_key_themes(n_words=30)
        
        # Convert to DataFrame for easier HTML formatting
        df = pd.DataFrame({
            'Theme': [theme[0] for theme in themes],
            'Frequency': [theme[1] for theme in themes]
        }) if themes else pd.DataFrame({'Theme': [], 'Frequency': []})
        
        # Create HTML report
        html = f"""
        <html>
        <head>
            <title>Gambia TripAdvisor Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2a5885; }}
                h2 {{ color: #4a76a8; margin-top: 30px; }}
                .stat-box {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin: 10px 0; }}
                .chart {{ margin: 20px 0; }}
                .notice {{ background: #fffacd; padding: 15px; border-radius: 5px; margin: 10px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
            </style>
        </head>
        <body>
            <h1>Gambia TripAdvisor Review Analysis</h1>
        """
        
        if not self.reviews:
            html += f"""
            <div class="notice">
                <h2>No Reviews Found</h2>
                <p>No reviews from {self.min_year} onwards were found in the input file. 
                Please check:</p>
                <ul>
                    <li>The date format in your JSON file</li>
                    <li>If the min_year setting ({self.min_year}) is appropriate</li>
                    <li>The structure of your JSON file</li>
                </ul>
            </div>
            """
        else:
            html += f"""
            <p>Analysis of {stats['total_reviews']} reviews from {self.min_year} onwards</p>
            
            <h2>Overview Statistics</h2>
            <div class="stat-box">
                <p><strong>Average Rating:</strong> {stats['avg_rating']:.2f}/5.0</p>
                <p><strong>Reviews by Category:</strong></p>
                <ul>
                    {
                    ''.join([f'<li>{category}: {count} reviews</li>' 
                            for category, count in sorted(stats['reviews_by_category'].items(), 
                                                     key=lambda x: x[1], reverse=True)])
                    }
                </ul>
                <p><strong>Reviews by Traveler Type:</strong></p>
                <ul>
                    {
                    ''.join([f'<li>{traveler_type}: {count} reviews</li>' 
                            for traveler_type, count in sorted(stats['reviews_by_traveler_type'].items(), 
                                                          key=lambda x: x[1], reverse=True)])
                    }
                </ul>
            </div>
            
            <h2>Sentiment Analysis</h2>
            <div class="stat-box">
                <p><strong>Overall Sentiment:</strong></p>
                <ul>
                    <li>Positive: {sentiment['overall_sentiment']['positive']:.2%}</li>
                    <li>Neutral: {sentiment['overall_sentiment']['neutral']:.2%}</li>
                    <li>Negative: {sentiment['overall_sentiment']['negative']:.2%}</li>
                    <li>Compound Score: {sentiment['overall_sentiment']['compound']:.2f}</li>
                </ul>
            """
            
            if sentiment['sentiment_by_category']:
                html += f"""
                <h3>Sentiment by Category</h3>
                <table>
                    <tr>
                        <th>Category</th>
                        <th>Review Count</th>
                        <th>Sentiment Score</th>
                        <th>Interpretation</th>
                    </tr>
                    {
                    ''.join([f'<tr><td>{category}</td><td>{data["count"]}</td><td>{data["avg_compound"]:.2f}</td><td>{"Positive" if data["avg_compound"] > 0.05 else "Negative" if data["avg_compound"] < -0.05 else "Neutral"}</td></tr>' 
                            for category, data in sorted(sentiment['sentiment_by_category'].items(), 
                                                    key=lambda x: x[1]['count'], reverse=True)])
                    }
                </table>
                """
            
            html += """
            </div>
            """
            
            if themes:
                html += f"""
                <h2>Key Themes in Reviews</h2>
                <div class="stat-box">
                    {df.to_html(index=False)}
                </div>
                """
                
            html += """
            <h2>Visualizations</h2>
            """
            
            if stats['rating_distribution']:
                html += """
                <div class="chart">
                    <img src="rating_distribution.png" alt="Rating Distribution" width="100%">
                </div>
                """
                
            if sentiment['sentiment_by_category']:
                html += """
                <div class="chart">
                    <img src="sentiment_by_category.png" alt="Sentiment by Category" width="100%">
                </div>
                """
                
            if themes:
                html += """
                <div class="chart">
                    <img src="themes_wordcloud.png" alt="Themes Word Cloud" width="100%">
                </div>
                """
        
        html += f"""
            <p><em>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
        </body>
        </html>
        """
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)
            
        print(f"Generated comprehensive report at {output_file}")