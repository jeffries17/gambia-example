#!/usr/bin/env python3
"""
Enhanced Goree Island Analysis with Translation
Translates all reviews to English for unified analysis and processes the complete dataset
"""

import json
import sys
import os
import time
from datetime import datetime
from collections import Counter
import re

# Add the sentiment_analyzer to path
sys.path.append('sentiment_analyzer')
from analysis.tourism_insights_analyzer import TourismInsightsAnalyzer

try:
    from googletrans import Translator
    HAS_TRANSLATION = True
except ImportError:
    HAS_TRANSLATION = False
    print("Warning: Google Translate not available. Install with: pip install googletrans==4.0.0rc1")

class EnhancedGoreeAnalyzer:
    """Enhanced analyzer with translation capabilities."""
    
    def __init__(self):
        self.translator = Translator() if HAS_TRANSLATION else None
        self.language_patterns = {
            'french': ['très', 'bon', 'excellent', 'merci', 'nous', 'avec', 'île', 'visite', 'guide'],
            'german': ['sehr', 'gut', 'ausgezeichnet', 'danke', 'wir', 'mit', 'insel', 'besuch'],
            'spanish': ['muy', 'bueno', 'excelente', 'gracias', 'nosotros', 'con', 'isla', 'visita'],
            'italian': ['molto', 'buono', 'eccellente', 'grazie', 'noi', 'con', 'isola', 'visita'],
            'dutch': ['zeer', 'goed', 'uitstekend', 'dank', 'wij', 'met', 'eiland', 'bezoek'],
            'portuguese': ['muito', 'bom', 'excelente', 'obrigado', 'nós', 'com', 'ilha', 'visita']
        }
    
    def detect_language(self, text):
        """Simple language detection based on common words."""
        text_lower = text.lower()
        scores = {}
        
        for lang, patterns in self.language_patterns.items():
            score = sum(1 for pattern in patterns if pattern in text_lower)
            scores[lang] = score
        
        # Check for English patterns
        english_patterns = ['the', 'and', 'very', 'good', 'great', 'amazing', 'beautiful', 'visit']
        english_score = sum(1 for pattern in english_patterns if pattern in text_lower)
        scores['english'] = english_score
        
        if max(scores.values()) == 0:
            return 'unknown'
        
        return max(scores, key=scores.get)
    
    def translate_text(self, text, source_lang='auto', target_lang='en', max_retries=3):
        """Translate text with error handling and retries."""
        if not self.translator or not text.strip():
            return text
        
        # Skip if already English
        detected_lang = self.detect_language(text)
        if detected_lang == 'english':
            return text
        
        for attempt in range(max_retries):
            try:
                result = self.translator.translate(text, src=source_lang, dest=target_lang)
                if result and result.text:
                    return result.text
            except Exception as e:
                print(f"Translation attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(1)  # Wait before retry
        
        print(f"Translation failed for text: {text[:50]}...")
        return text  # Return original if translation fails
    
    def translate_reviews(self, reviews):
        """Translate all non-English reviews to English."""
        print("Translating reviews to English...")
        translated_reviews = []
        
        language_stats = Counter()
        translation_stats = Counter()
        
        for i, review in enumerate(reviews):
            print(f"Processing review {i+1}/{len(reviews)}", end='\r')
            
            original_text = review.get('text', '')
            original_title = review.get('title', '')
            
            # Detect languages
            text_lang = self.detect_language(original_text)
            title_lang = self.detect_language(original_title)
            
            language_stats[text_lang] += 1
            
            # Create translated review
            translated_review = review.copy()
            
            # Translate text if not English
            if text_lang != 'english' and original_text.strip():
                translated_text = self.translate_text(original_text)
                translated_review['text'] = translated_text
                translated_review['original_text'] = original_text
                translated_review['detected_language'] = text_lang
                translation_stats['text_translated'] += 1
            else:
                translated_review['detected_language'] = 'english'
            
            # Translate title if not English
            if title_lang != 'english' and original_title.strip():
                translated_title = self.translate_text(original_title)
                translated_review['title'] = translated_title
                translated_review['original_title'] = original_title
                translation_stats['title_translated'] += 1
            
            translated_reviews.append(translated_review)
            
            # Small delay to avoid rate limiting
            time.sleep(0.1)
        
        print(f"\nTranslation complete!")
        print(f"Language distribution: {dict(language_stats)}")
        print(f"Translation stats: {dict(translation_stats)}")
        
        return translated_reviews
    
    def prepare_enhanced_data(self):
        """Load and enhance Goree Island reviews with translations."""
        print("Loading Goree Island reviews...")
        
        # Load reviews
        with open('sentiment_analyzer/goree_reviews.json', 'r', encoding='utf-8') as f:
            raw_reviews = json.load(f)
        
        print(f"Found {len(raw_reviews)} reviews (out of {raw_reviews[0]['placeInfo']['numberOfReviews']} total)")
        
        # Translate reviews
        translated_reviews = self.translate_reviews(raw_reviews)
        
        # Extract place info from first review
        place_info = raw_reviews[0]['placeInfo']
        
        # Structure data like TripAdvisor extraction format
        structured_data = {
            'business_info': {
                'name': place_info['name'],
                'category': 'attraction',
                'location': place_info['locationString'],
                'rating': place_info['rating'],
                'review_count': place_info['numberOfReviews'],
                'latitude': place_info['latitude'],
                'longitude': place_info['longitude'],
                'website': place_info.get('website', ''),
                'address': place_info['address']
            },
            'reviews': [],
            'extraction_metadata': {
                'url': place_info['webUrl'],
                'extraction_date': datetime.now().isoformat(),
                'total_reviews_extracted': len(translated_reviews),
                'total_reviews_available': place_info['numberOfReviews'],
                'translation_enabled': HAS_TRANSLATION,
                'languages_detected': len(set([r.get('detected_language', 'unknown') for r in translated_reviews]))
            }
        }
        
        # Convert reviews to expected format
        for review in translated_reviews:
            processed_review = {
                'title': review.get('title', ''),
                'rating': review.get('rating', 0),
                'text': review.get('text', ''),
                'date': review.get('publishedDate', ''),
                'travel_date': review.get('travelDate', ''),
                'user_name': review.get('user', {}).get('name', ''),
                'user_location': review.get('user', {}).get('userLocation', {}).get('name', '') if review.get('user', {}).get('userLocation') else '',
                'user_contributions': review.get('user', {}).get('contributions', {}).get('totalContributions', 0),
                'helpful_votes': review.get('user', {}).get('contributions', {}).get('helpfulVotes', 0),
                'owner_response': review.get('ownerResponse'),
                'review_url': review.get('url', ''),
                # Translation metadata
                'original_text': review.get('original_text'),
                'original_title': review.get('original_title'),
                'detected_language': review.get('detected_language', 'english'),
                'was_translated': 'original_text' in review or 'original_title' in review
            }
            structured_data['reviews'].append(processed_review)
        
        return structured_data

    def analyze_enhanced_goree(self):
        """Analyze Goree Island reviews with translation."""
        print("=== Enhanced Goree Island Analysis with Translation ===")
        
        # Prepare enhanced data
        goree_data = self.prepare_enhanced_data()
        
        # Save translated dataset
        with open('goree_translated_dataset.json', 'w', encoding='utf-8') as f:
            json.dump(goree_data, f, indent=2, ensure_ascii=False)
        print("Translated dataset saved to: goree_translated_dataset.json")
        
        # Run analysis
        print("\nRunning enhanced tourism insights analysis...")
        analyzer = TourismInsightsAnalyzer()
        results = analyzer.analyze_destination_reviews(goree_data)
        
        # Add translation insights
        results['translation_analysis'] = self.generate_translation_insights(goree_data['reviews'])
        
        # Convert numpy types for JSON serialization
        def convert_numpy_types(obj):
            if hasattr(obj, 'item'):
                return obj.item()
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            else:
                return obj
        
        results_serializable = convert_numpy_types(results)
        
        # Save enhanced results
        output_file = 'goree_enhanced_analysis_results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, indent=2, ensure_ascii=False)
        
        print(f"Enhanced analysis complete! Results saved to {output_file}")
        
        # Print summary
        self.print_enhanced_summary(results)
        
        return results

    def generate_translation_insights(self, reviews):
        """Generate insights about the translation process."""
        language_dist = Counter()
        translation_count = 0
        
        for review in reviews:
            language_dist[review.get('detected_language', 'unknown')] += 1
            if review.get('was_translated', False):
                translation_count += 1
        
        return {
            'total_reviews': len(reviews),
            'translations_performed': translation_count,
            'translation_percentage': (translation_count / len(reviews)) * 100,
            'language_distribution': dict(language_dist),
            'dominant_language': max(language_dist, key=language_dist.get),
            'languages_detected': len(language_dist)
        }

    def print_enhanced_summary(self, results):
        """Print enhanced analysis summary."""
        print("\n" + "="*60)
        print("ENHANCED GOREE ISLAND ANALYSIS SUMMARY")
        print("="*60)
        
        # Basic metadata
        if 'analysis_metadata' in results:
            metadata = results['analysis_metadata']
            print(f"Destination: {metadata['destination']}")
            print(f"Location: {metadata['location']}")
            print(f"Total reviews analyzed: {metadata['total_reviews']}")
        
        # Translation insights
        if 'translation_analysis' in results:
            trans = results['translation_analysis']
            print(f"\nTRANSLATION SUMMARY:")
            print(f"- Reviews translated: {trans['translations_performed']}/{trans['total_reviews']} ({trans['translation_percentage']:.1f}%)")
            print(f"- Languages detected: {trans['languages_detected']}")
            print(f"- Dominant language: {trans['dominant_language']}")
            print(f"- Language distribution: {trans['language_distribution']}")
        
        # Sentiment analysis
        if 'overall_sentiment' in results:
            sentiment = results['overall_sentiment']
            print(f"\nSENTIMENT ANALYSIS:")
            print(f"- Average rating: {sentiment.get('average_rating', 'N/A')}/5")
            print(f"- Overall sentiment score: {sentiment.get('overall_score', 'N/A'):.3f}")
            
            dist = sentiment.get('sentiment_distribution', {})
            print(f"- Positive: {dist.get('positive_percentage', 0):.1f}%")
            print(f"- Neutral: {dist.get('neutral_percentage', 0):.1f}%")
            print(f"- Negative: {dist.get('negative_percentage', 0):.1f}%")
        
        # Top aspects
        if 'aspect_sentiment' in results:
            aspects = results['aspect_sentiment']
            print(f"\nTOP MENTIONED ASPECTS:")
            sorted_aspects = sorted(aspects.items(), 
                                  key=lambda x: x[1].get('mention_percentage', 0), 
                                  reverse=True)
            for aspect, data in sorted_aspects[:5]:
                mention_pct = data.get('mention_percentage', 0)
                avg_sentiment = data.get('average_sentiment', 0)
                print(f"- {aspect.replace('_', ' ').title()}: {mention_pct:.1f}% mentions, {avg_sentiment:.3f} sentiment")

def main():
    """Main function."""
    if not HAS_TRANSLATION:
        print("Error: Translation library not available.")
        print("Please install with: pip install googletrans==4.0.0rc1")
        return
    
    analyzer = EnhancedGoreeAnalyzer()
    
    try:
        results = analyzer.analyze_enhanced_goree()
        print("\n✅ Enhanced Goree Island analysis completed successfully!")
        print("📁 Files created:")
        print("  - goree_translated_dataset.json (translated reviews)")
        print("  - goree_enhanced_analysis_results.json (analysis results)")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 