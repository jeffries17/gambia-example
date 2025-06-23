#!/usr/bin/env python3
"""
Manual Translation Enhanced Goree Island Analysis
Provides comprehensive analysis with manual translation for key phrases
"""

import json
import sys
import os
from datetime import datetime
from collections import Counter
import re

# Add the sentiment_analyzer to path
sys.path.append('sentiment_analyzer')
from analysis.tourism_insights_analyzer import TourismInsightsAnalyzer

class ManualTranslationGoreeAnalyzer:
    """Enhanced analyzer with manual translation capabilities."""
    
    def __init__(self):
        # Manual translation dictionaries for key tourism terms
        self.translation_dict = {
            # French to English
            'très': 'very',
            'bon': 'good',
            'bonne': 'good',
            'excellent': 'excellent',
            'magnifique': 'magnificent',
            'belle': 'beautiful',
            'beau': 'beautiful',
            'merci': 'thank you',
            'île': 'island',
            'visite': 'visit',
            'guide': 'guide',
            'histoire': 'history',
            'émouvante': 'moving',
            'recommande': 'recommend',
            'incroyable': 'incredible',
            'superbe': 'superb',
            'fantastique': 'fantastic',
            'parfait': 'perfect',
            'agréable': 'pleasant',
            'sympathique': 'nice',
            'professionnel': 'professional',
            'passionnant': 'fascinating',
            'chargé': 'loaded',
            'dommage': 'pity',
            'déception': 'disappointment',
            'problème': 'problem',
            'catastrophique': 'catastrophic',
            'attente': 'waiting',
            'bateau': 'boat',
            'ferry': 'ferry',
            'transport': 'transport',
            'saleté': 'dirt',
            'propre': 'clean',
            'sale': 'dirty',
            'prix': 'price',
            'cher': 'expensive',
            'gratuit': 'free',
            'rapide': 'fast',
            'lent': 'slow',
            'facile': 'easy',
            'difficile': 'difficult',
            
            # German to English
            'sehr': 'very',
            'gut': 'good',
            'ausgezeichnet': 'excellent',
            'schön': 'beautiful',
            'wunderbar': 'wonderful',
            'perfekt': 'perfect',
            'insel': 'island',
            'besuch': 'visit',
            'geschichte': 'history',
            'empfehlenswert': 'recommended',
            'interessant': 'interesting',
            'sauber': 'clean',
            'schmutzig': 'dirty',
            'teuer': 'expensive',
            'billig': 'cheap',
            
            # Spanish to English
            'muy': 'very',
            'bueno': 'good',
            'excelente': 'excellent',
            'hermoso': 'beautiful',
            'increíble': 'incredible',
            'perfecto': 'perfect',
            'isla': 'island',
            'visita': 'visit',
            'historia': 'history',
            'recomiendo': 'recommend',
            'limpio': 'clean',
            'sucio': 'dirty',
            'caro': 'expensive',
            'barato': 'cheap',
            
            # Common sentiment words
            'harcèle': 'harass',
            'assaillis': 'assailed',
            'profondément': 'deeply',
            'bouleversante': 'moving',
            'impressionnant': 'impressive',
            'décevant': 'disappointing',
            'horrible': 'horrible',
            'merveilleux': 'marvelous',
            'formidable': 'great',
        }
        
        self.language_patterns = {
            'french': ['très', 'bon', 'excellent', 'merci', 'nous', 'avec', 'île', 'visite', 'guide', 'et', 'de', 'la', 'le', 'une', 'sur'],
            'german': ['sehr', 'gut', 'ausgezeichnet', 'danke', 'wir', 'mit', 'insel', 'besuch', 'und', 'der', 'die', 'das', 'ist'],
            'spanish': ['muy', 'bueno', 'excelente', 'gracias', 'nosotros', 'con', 'isla', 'visita', 'y', 'el', 'la', 'de', 'que'],
            'italian': ['molto', 'buono', 'eccellente', 'grazie', 'noi', 'con', 'isola', 'visita', 'e', 'il', 'la', 'di'],
            'dutch': ['zeer', 'goed', 'uitstekend', 'dank', 'wij', 'met', 'eiland', 'bezoek', 'en', 'de', 'het', 'van'],
            'portuguese': ['muito', 'bom', 'excelente', 'obrigado', 'nós', 'com', 'ilha', 'visita', 'e', 'o', 'a', 'de']
        }
    
    def detect_language(self, text):
        """Detect language based on common words."""
        text_lower = text.lower()
        scores = {}
        
        for lang, patterns in self.language_patterns.items():
            score = sum(1 for pattern in patterns if pattern in text_lower)
            scores[lang] = score
        
        # Check for English patterns
        english_patterns = ['the', 'and', 'very', 'good', 'great', 'amazing', 'beautiful', 'visit', 'is', 'was', 'were', 'have', 'had', 'with']
        english_score = sum(1 for pattern in english_patterns if pattern in text_lower)
        scores['english'] = english_score
        
        if max(scores.values()) == 0:
            return 'unknown'
        
        return max(scores, key=scores.get)
    
    def manual_translate_key_phrases(self, text):
        """Manually translate key phrases in text."""
        if not text:
            return text
        
        # Create translated version with key terms replaced
        translated = text
        for foreign_word, english_word in self.translation_dict.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(foreign_word) + r'\b'
            translated = re.sub(pattern, english_word, translated, flags=re.IGNORECASE)
        
        return translated
    
    def enhance_reviews_with_translation(self, reviews):
        """Enhance reviews with manual translation and language detection."""
        print("Enhancing reviews with language detection and manual translation...")
        enhanced_reviews = []
        
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
            
            # Create enhanced review
            enhanced_review = review.copy()
            enhanced_review['detected_language'] = text_lang
            
            # Add manual translation for key phrases
            if text_lang != 'english' and original_text.strip():
                translated_text = self.manual_translate_key_phrases(original_text)
                enhanced_review['enhanced_text'] = translated_text
                enhanced_review['original_text'] = original_text
                translation_stats['text_enhanced'] += 1
            else:
                enhanced_review['enhanced_text'] = original_text
            
            if title_lang != 'english' and original_title.strip():
                translated_title = self.manual_translate_key_phrases(original_title)
                enhanced_review['enhanced_title'] = translated_title
                enhanced_review['original_title'] = original_title
                translation_stats['title_enhanced'] += 1
            else:
                enhanced_review['enhanced_title'] = original_title
            
            enhanced_reviews.append(enhanced_review)
        
        print(f"\nEnhancement complete!")
        print(f"Language distribution: {dict(language_stats)}")
        print(f"Enhancement stats: {dict(translation_stats)}")
        
        return enhanced_reviews
    
    def prepare_enhanced_data(self):
        """Load and enhance Goree Island reviews."""
        print("Loading Goree Island reviews...")
        
        # Load reviews
        with open('sentiment_analyzer/goree_reviews.json', 'r', encoding='utf-8') as f:
            raw_reviews = json.load(f)
        
        print(f"Found {len(raw_reviews)} reviews (subset of {raw_reviews[0]['placeInfo']['numberOfReviews']} total)")
        
        # Enhance reviews with manual translation
        enhanced_reviews = self.enhance_reviews_with_translation(raw_reviews)
        
        # Extract place info
        place_info = raw_reviews[0]['placeInfo']
        
        # Structure data for analysis
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
                'total_reviews_extracted': len(enhanced_reviews),
                'total_reviews_available': place_info['numberOfReviews'],
                'enhancement_enabled': True,
                'languages_detected': len(set([r.get('detected_language', 'unknown') for r in enhanced_reviews]))
            }
        }
        
        # Convert reviews to expected format using enhanced text
        for review in enhanced_reviews:
            processed_review = {
                'title': review.get('enhanced_title', review.get('title', '')),
                'rating': review.get('rating', 0),
                'text': review.get('enhanced_text', review.get('text', '')),
                'date': review.get('publishedDate', ''),
                'travel_date': review.get('travelDate', ''),
                'user_name': review.get('user', {}).get('name', ''),
                'user_location': review.get('user', {}).get('userLocation', {}).get('name', '') if review.get('user', {}).get('userLocation') else '',
                'user_contributions': review.get('user', {}).get('contributions', {}).get('totalContributions', 0),
                'helpful_votes': review.get('user', {}).get('contributions', {}).get('helpfulVotes', 0),
                'owner_response': review.get('ownerResponse'),
                'review_url': review.get('url', ''),
                # Enhancement metadata
                'original_text': review.get('original_text'),
                'original_title': review.get('original_title'),
                'detected_language': review.get('detected_language', 'english'),
                'was_enhanced': review.get('enhanced_text') != review.get('text')
            }
            structured_data['reviews'].append(processed_review)
        
        return structured_data

    def analyze_enhanced_goree(self):
        """Analyze Goree Island reviews with enhancement."""
        print("=== Enhanced Goree Island Analysis with Manual Translation ===")
        
        # Prepare enhanced data
        goree_data = self.prepare_enhanced_data()
        
        # Save enhanced dataset
        with open('goree_enhanced_dataset.json', 'w', encoding='utf-8') as f:
            json.dump(goree_data, f, indent=2, ensure_ascii=False)
        print("Enhanced dataset saved to: goree_enhanced_dataset.json")
        
        # Run analysis on enhanced data
        print("\nRunning enhanced tourism insights analysis...")
        analyzer = TourismInsightsAnalyzer()
        results = analyzer.analyze_destination_reviews(goree_data)
        
        # Add enhancement insights
        results['enhancement_analysis'] = self.generate_enhancement_insights(goree_data['reviews'])
        
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

    def generate_enhancement_insights(self, reviews):
        """Generate insights about the enhancement process."""
        language_dist = Counter()
        enhancement_count = 0
        
        for review in reviews:
            language_dist[review.get('detected_language', 'unknown')] += 1
            if review.get('was_enhanced', False):
                enhancement_count += 1
        
        return {
            'total_reviews': len(reviews),
            'enhancements_performed': enhancement_count,
            'enhancement_percentage': (enhancement_count / len(reviews)) * 100,
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
        
        # Enhancement insights
        if 'enhancement_analysis' in results:
            enhance = results['enhancement_analysis']
            print(f"\nENHANCEMENT SUMMARY:")
            print(f"- Reviews enhanced: {enhance['enhancements_performed']}/{enhance['total_reviews']} ({enhance['enhancement_percentage']:.1f}%)")
            print(f"- Languages detected: {enhance['languages_detected']}")
            print(f"- Dominant language: {enhance['dominant_language']}")
            print(f"- Language distribution: {enhance['language_distribution']}")
        
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
    analyzer = ManualTranslationGoreeAnalyzer()
    
    try:
        results = analyzer.analyze_enhanced_goree()
        print("\n✅ Enhanced Goree Island analysis completed successfully!")
        print("📁 Files created:")
        print("  - goree_enhanced_dataset.json (enhanced reviews with manual translation)")
        print("  - goree_enhanced_analysis_results.json (analysis results)")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 