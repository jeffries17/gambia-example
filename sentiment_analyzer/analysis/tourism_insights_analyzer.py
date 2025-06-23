#!/usr/bin/env python3
"""
Tourism Insights Analyzer

This module provides specialized analysis for tourism boards, extracting key insights
from TripAdvisor reviews that are most relevant for destination marketing and improvement.
"""

import os
import json
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging

try:
    from textblob import TextBlob
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.util import ngrams
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    HAS_NLP = True
except ImportError:
    HAS_NLP = False

logger = logging.getLogger(__name__)

class TourismInsightsAnalyzer:
    """
    Specialized analyzer for extracting tourism board insights from TripAdvisor reviews.
    Focuses on the key factors that tourism boards need for strategic decision making.
    """
    
    def __init__(self, output_dir='outputs/tourism_insights'):
        """
        Initialize the tourism insights analyzer.
        
        Args:
            output_dir (str): Directory for tourism insights outputs
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize NLP components if available
        if HAS_NLP:
            try:
                self.stop_words = set(stopwords.words('english'))
                # Add tourism-specific stop words
                self.stop_words.update([
                    'hotel', 'room', 'stay', 'place', 'time', 'day', 'night',
                    'trip', 'visit', 'go', 'went', 'come', 'came', 'get', 'got'
                ])
            except:
                self.stop_words = set()
        else:
            self.stop_words = set()
        
        # Define tourism aspects for analysis
        self.tourism_aspects = {
            'accommodation': {
                'keywords': ['hotel', 'room', 'bed', 'bathroom', 'shower', 'accommodation', 
                           'lodge', 'resort', 'guesthouse', 'service', 'staff', 'reception'],
                'sub_aspects': {
                    'cleanliness': ['clean', 'dirty', 'spotless', 'hygiene', 'maintenance'],
                    'comfort': ['comfortable', 'bed', 'pillow', 'quiet', 'peaceful', 'cozy'],
                    'service': ['service', 'staff', 'friendly', 'helpful', 'professional'],
                    'amenities': ['wifi', 'ac', 'pool', 'breakfast', 'parking', 'restaurant']
                }
            },
            'restaurants': {
                'keywords': ['food', 'restaurant', 'meal', 'dinner', 'lunch', 'breakfast',
                           'eat', 'taste', 'delicious', 'menu', 'chef', 'cuisine'],
                'sub_aspects': {
                    'food_quality': ['delicious', 'tasty', 'fresh', 'flavor', 'quality'],
                    'service': ['service', 'waiter', 'staff', 'friendly', 'fast', 'slow'],
                    'value': ['price', 'expensive', 'cheap', 'value', 'worth', 'cost'],
                    'atmosphere': ['atmosphere', 'ambiance', 'music', 'view', 'setting']
                }
            },
            'attractions': {
                'keywords': ['attraction', 'tour', 'guide', 'museum', 'beach', 'park',
                           'activity', 'experience', 'visit', 'see', 'beautiful', 'amazing'],
                'sub_aspects': {
                    'experience_quality': ['amazing', 'beautiful', 'stunning', 'incredible', 'wonderful'],
                    'accessibility': ['easy', 'difficult', 'accessible', 'reach', 'transport'],
                    'value': ['price', 'worth', 'expensive', 'free', 'ticket', 'cost'],
                    'facilities': ['facilities', 'toilets', 'parking', 'restaurant', 'shop']
                }
            },
            'transportation': {
                'keywords': ['transport', 'taxi', 'bus', 'car', 'drive', 'road', 'airport',
                           'transfer', 'journey', 'travel', 'distance'],
                'sub_aspects': {
                    'accessibility': ['easy', 'difficult', 'convenient', 'accessible'],
                    'quality': ['good', 'bad', 'comfortable', 'safe', 'reliable'],
                    'cost': ['expensive', 'cheap', 'reasonable', 'cost', 'price']
                }
            },
            'safety': {
                'keywords': ['safe', 'safety', 'secure', 'dangerous', 'crime', 'theft',
                           'police', 'security', 'worry', 'comfortable', 'trust'],
                'sub_aspects': {
                    'personal_safety': ['safe', 'dangerous', 'secure', 'theft', 'crime'],
                    'comfort_level': ['comfortable', 'worried', 'relaxed', 'confident']
                }
            },
            'local_culture': {
                'keywords': ['culture', 'local', 'people', 'friendly', 'tradition', 'authentic',
                           'history', 'heritage', 'customs', 'community'],
                'sub_aspects': {
                    'friendliness': ['friendly', 'welcoming', 'warm', 'helpful', 'kind'],
                    'authenticity': ['authentic', 'traditional', 'genuine', 'real', 'original'],
                    'cultural_richness': ['culture', 'history', 'heritage', 'tradition']
                }
            }
        }
        
        # Language detection patterns (simple approach)
        self.language_patterns = {
            'spanish': ['muy', 'bueno', 'excelente', 'gracias', 'hotel', 'comida'],
            'french': ['très', 'bon', 'excellent', 'merci', 'hôtel', 'nourriture'],
            'german': ['sehr', 'gut', 'ausgezeichnet', 'danke', 'hotel', 'essen'],
            'italian': ['molto', 'buono', 'eccellente', 'grazie', 'albergo', 'cibo'],
            'portuguese': ['muito', 'bom', 'excelente', 'obrigado', 'hotel', 'comida'],
            'dutch': ['zeer', 'goed', 'uitstekend', 'dank', 'hotel', 'eten'],
            'chinese': ['很好', '非常', '谢谢', '酒店', '食物'],
            'japanese': ['とても', '良い', '素晴らしい', 'ありがとう', 'ホテル'],
            'korean': ['매우', '좋은', '훌륭한', '감사합니다', '호텔'],
            'arabic': ['جيد', 'ممتاز', 'شكرا', 'فندق', 'طعام']
        }
    
    def analyze_destination_reviews(self, tripadvisor_data: Dict) -> Dict:
        """
        Perform comprehensive tourism insights analysis on TripAdvisor data.
        
        Args:
            tripadvisor_data (Dict): Extracted TripAdvisor data
            
        Returns:
            Dict: Comprehensive tourism insights
        """
        logger.info("Starting comprehensive tourism insights analysis...")
        
        # Convert to DataFrame for analysis
        reviews_df = self._prepare_dataframe(tripadvisor_data)
        
        if len(reviews_df) == 0:
            logger.warning("No reviews found for analysis")
            return {}
        
        # Perform all analyses
        insights = {
            'analysis_metadata': {
                'destination': tripadvisor_data.get('business_info', {}).get('name', 'Unknown'),
                'category': tripadvisor_data.get('business_info', {}).get('category', 'Unknown'),
                'location': tripadvisor_data.get('business_info', {}).get('location', 'Unknown'),
                'total_reviews': len(reviews_df),
                'analysis_date': datetime.now().isoformat(),
                'source_url': tripadvisor_data.get('extraction_metadata', {}).get('url', 'Unknown')
            }
        }
        
        # 1. Overall Sentiment Score
        insights['overall_sentiment'] = self._calculate_overall_sentiment(reviews_df)
        
        # 2. Aspect-Based Sentiment Scores
        insights['aspect_sentiment'] = self._analyze_aspect_sentiment(reviews_df)
        
        # 3. Keywords & Phrases Analysis
        insights['keywords_phrases'] = self._extract_keywords_phrases(reviews_df)
        
        # 4. Recurring Themes
        insights['recurring_themes'] = self._identify_recurring_themes(reviews_df)
        
        # 5. Language Diversification
        insights['language_analysis'] = self._analyze_language_diversity(reviews_df)
        
        # 6. Responsiveness Analysis
        insights['responsiveness_analysis'] = self._analyze_responsiveness(reviews_df)
        
        # 7. Temporal Patterns
        insights['temporal_patterns'] = self._analyze_temporal_patterns(reviews_df)
        
        # Generate summary report
        insights['executive_summary'] = self._generate_executive_summary(insights)
        
        logger.info("Tourism insights analysis complete")
        return insights
    
    def _prepare_dataframe(self, tripadvisor_data: Dict) -> pd.DataFrame:
        """Prepare DataFrame with sentiment analysis."""
        reviews = tripadvisor_data.get('reviews', [])
        df = pd.DataFrame(reviews)
        
        if len(df) == 0:
            return df
        
        # Add sentiment analysis
        if HAS_NLP and 'text' in df.columns:
            df['sentiment_score'] = df['text'].apply(self._get_sentiment_score)
            df['sentiment_category'] = df['sentiment_score'].apply(self._categorize_sentiment)
        else:
            # Fallback to rating-based sentiment
            if 'rating' in df.columns:
                df['sentiment_score'] = (df['rating'] - 3) / 2
                df['sentiment_category'] = df['sentiment_score'].apply(self._categorize_sentiment)
        
        return df
    
    def _get_sentiment_score(self, text: str) -> float:
        """Calculate sentiment score for text."""
        if not text or pd.isna(text):
            return 0.0
        
        if HAS_NLP:
            try:
                blob = TextBlob(str(text))
                return blob.sentiment.polarity
            except:
                pass
        
        return 0.0
    
    def _categorize_sentiment(self, score: float) -> str:
        """Categorize sentiment score."""
        if score > 0.1:
            return 'positive'
        elif score < -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    def _calculate_overall_sentiment(self, df: pd.DataFrame) -> Dict:
        """Calculate overall sentiment metrics."""
        if 'sentiment_score' not in df.columns:
            return {}
        
        sentiment_counts = df['sentiment_category'].value_counts()
        
        return {
            'overall_score': float(df['sentiment_score'].mean()),
            'sentiment_distribution': {
                'positive_percentage': float((sentiment_counts.get('positive', 0) / len(df)) * 100),
                'negative_percentage': float((sentiment_counts.get('negative', 0) / len(df)) * 100),
                'neutral_percentage': float((sentiment_counts.get('neutral', 0) / len(df)) * 100)
            },
            'rating_distribution': dict(df['rating'].value_counts()) if 'rating' in df.columns else {},
            'average_rating': float(df['rating'].mean()) if 'rating' in df.columns else None,
            'total_reviews': len(df),
            'confidence_level': self._calculate_confidence_level(len(df))
        }
    
    def _analyze_aspect_sentiment(self, df: pd.DataFrame) -> Dict:
        """Analyze sentiment for each tourism aspect."""
        aspect_results = {}
        
        for aspect, config in self.tourism_aspects.items():
            aspect_sentiment = self._analyze_single_aspect(df, aspect, config)
            if aspect_sentiment:
                aspect_results[aspect] = aspect_sentiment
        
        return aspect_results
    
    def _analyze_single_aspect(self, df: pd.DataFrame, aspect: str, config: Dict) -> Dict:
        """Analyze sentiment for a single aspect."""
        if 'text' not in df.columns:
            return {}
        
        # Find reviews mentioning this aspect
        keywords = config['keywords']
        pattern = '|'.join(keywords)
        aspect_mask = df['text'].str.lower().str.contains(pattern, na=False)
        aspect_df = df[aspect_mask]
        
        if len(aspect_df) < 3:  # Need minimum reviews for reliable analysis
            return {}
        
        result = {
            'mention_count': len(aspect_df),
            'mention_percentage': float((len(aspect_df) / len(df)) * 100),
            'average_sentiment': float(aspect_df['sentiment_score'].mean()) if 'sentiment_score' in aspect_df.columns else None,
            'sentiment_distribution': dict(aspect_df['sentiment_category'].value_counts()) if 'sentiment_category' in aspect_df.columns else {},
            'sub_aspects': {}
        }
        
        # Analyze sub-aspects
        for sub_aspect, sub_keywords in config.get('sub_aspects', {}).items():
            sub_pattern = '|'.join(sub_keywords)
            sub_mask = aspect_df['text'].str.lower().str.contains(sub_pattern, na=False)
            sub_df = aspect_df[sub_mask]
            
            if len(sub_df) > 0:
                result['sub_aspects'][sub_aspect] = {
                    'mention_count': len(sub_df),
                    'average_sentiment': float(sub_df['sentiment_score'].mean()) if 'sentiment_score' in sub_df.columns else None,
                    'key_phrases': self._extract_key_phrases_for_aspect(sub_df, sub_keywords)
                }
        
        return result
    
    def _extract_keywords_phrases(self, df: pd.DataFrame) -> Dict:
        """Extract commonly mentioned keywords and phrases."""
        if 'text' not in df.columns or not HAS_NLP:
            return {}
        
        # Combine all review text
        all_text = ' '.join(df['text'].fillna('').astype(str))
        
        # Extract keywords by sentiment
        positive_df = df[df['sentiment_category'] == 'positive'] if 'sentiment_category' in df.columns else df
        negative_df = df[df['sentiment_category'] == 'negative'] if 'sentiment_category' in df.columns else pd.DataFrame()
        
        results = {
            'positive_keywords': self._extract_top_keywords(positive_df, sentiment_type='positive'),
            'negative_keywords': self._extract_top_keywords(negative_df, sentiment_type='negative'),
            'overall_keywords': self._extract_top_keywords(df, sentiment_type='overall'),
            'bigrams': self._extract_bigrams(all_text),
            'trigrams': self._extract_trigrams(all_text)
        }
        
        return results
    
    def _extract_top_keywords(self, df: pd.DataFrame, sentiment_type: str, limit: int = 20) -> List[Dict]:
        """Extract top keywords for a sentiment category."""
        if len(df) == 0 or 'text' not in df.columns:
            return []
        
        # Combine text and tokenize
        text = ' '.join(df['text'].fillna('').astype(str))
        words = word_tokenize(text.lower())
        
        # Filter words
        filtered_words = [
            word for word in words 
            if word.isalpha() and len(word) > 2 and word not in self.stop_words
        ]
        
        # Count frequencies
        word_freq = Counter(filtered_words)
        
        # Return top words with context
        top_words = []
        for word, count in word_freq.most_common(limit):
            # Calculate sentiment for this word
            word_sentiment = self._calculate_word_sentiment(df, word)
            
            top_words.append({
                'word': word,
                'frequency': count,
                'percentage': float((count / len(filtered_words)) * 100),
                'sentiment_score': word_sentiment,
                'example_phrases': self._get_example_phrases(df, word)
            })
        
        return top_words
    
    def _extract_bigrams(self, text: str, limit: int = 15) -> List[Dict]:
        """Extract meaningful bigrams (two-word phrases)."""
        if not HAS_NLP:
            return []
        
        words = word_tokenize(text.lower())
        filtered_words = [w for w in words if w.isalpha() and len(w) > 2 and w not in self.stop_words]
        
        bigram_list = list(ngrams(filtered_words, 2))
        bigram_freq = Counter(bigram_list)
        
        results = []
        for bigram, count in bigram_freq.most_common(limit):
            phrase = ' '.join(bigram)
            results.append({
                'phrase': phrase,
                'frequency': count,
                'words': list(bigram)
            })
        
        return results
    
    def _extract_trigrams(self, text: str, limit: int = 10) -> List[Dict]:
        """Extract meaningful trigrams (three-word phrases)."""
        if not HAS_NLP:
            return []
        
        words = word_tokenize(text.lower())
        filtered_words = [w for w in words if w.isalpha() and len(w) > 2 and w not in self.stop_words]
        
        trigram_list = list(ngrams(filtered_words, 3))
        trigram_freq = Counter(trigram_list)
        
        results = []
        for trigram, count in trigram_freq.most_common(limit):
            phrase = ' '.join(trigram)
            results.append({
                'phrase': phrase,
                'frequency': count,
                'words': list(trigram)
            })
        
        return results
    
    def _identify_recurring_themes(self, df: pd.DataFrame) -> Dict:
        """Identify recurring themes in reviews using topic modeling approach."""
        if 'text' not in df.columns or not HAS_NLP or len(df) < 10:
            return {}
        
        # Simple theme identification based on keyword clustering
        themes = {
            'infrastructure_challenges': {
                'keywords': ['road', 'transport', 'difficult', 'access', 'infrastructure', 'maintenance'],
                'description': 'Issues related to infrastructure and accessibility'
            },
            'natural_beauty': {
                'keywords': ['beautiful', 'stunning', 'nature', 'beach', 'view', 'scenery', 'landscape'],
                'description': 'Appreciation for natural attractions and scenery'
            },
            'cultural_authenticity': {
                'keywords': ['authentic', 'traditional', 'culture', 'local', 'heritage', 'history'],
                'description': 'Experiences related to local culture and authenticity'
            },
            'service_quality': {
                'keywords': ['service', 'staff', 'friendly', 'helpful', 'professional', 'customer'],
                'description': 'Quality of service and staff interactions'
            },
            'value_concerns': {
                'keywords': ['expensive', 'overpriced', 'value', 'money', 'worth', 'cost'],
                'description': 'Concerns about pricing and value for money'
            },
            'safety_security': {
                'keywords': ['safe', 'security', 'dangerous', 'theft', 'crime', 'worry'],
                'description': 'Safety and security experiences'
            }
        }
        
        theme_results = {}
        
        for theme_name, theme_config in themes.items():
            pattern = '|'.join(theme_config['keywords'])
            theme_mask = df['text'].str.lower().str.contains(pattern, na=False)
            theme_df = df[theme_mask]
            
            if len(theme_df) > 0:
                theme_results[theme_name] = {
                    'description': theme_config['description'],
                    'mention_count': len(theme_df),
                    'percentage': float((len(theme_df) / len(df)) * 100),
                    'average_sentiment': float(theme_df['sentiment_score'].mean()) if 'sentiment_score' in theme_df.columns else None,
                    'key_examples': self._get_theme_examples(theme_df, theme_config['keywords']),
                    'sentiment_breakdown': dict(theme_df['sentiment_category'].value_counts()) if 'sentiment_category' in theme_df.columns else {}
                }
        
        return theme_results
    
    def _analyze_language_diversity(self, df: pd.DataFrame) -> Dict:
        """Analyze language diversity in reviews."""
        if 'text' not in df.columns:
            return {}
        
        language_counts = defaultdict(int)
        total_reviews = len(df)
        
        for text in df['text'].fillna(''):
            detected_lang = self._detect_language(text)
            language_counts[detected_lang] += 1
        
        # Convert to percentages and insights
        language_analysis = {}
        for lang, count in language_counts.items():
            percentage = (count / total_reviews) * 100
            language_analysis[lang] = {
                'review_count': count,
                'percentage': float(percentage)
            }
        
        # Calculate diversity metrics
        diversity_score = len([lang for lang, data in language_analysis.items() 
                             if data['percentage'] > 5])  # Languages with >5% representation
        
        return {
            'language_distribution': language_analysis,
            'diversity_score': diversity_score,
            'primary_language': max(language_counts.items(), key=lambda x: x[1])[0],
            'international_appeal': float(sum(data['percentage'] for lang, data in language_analysis.items() 
                                            if lang != 'english')),
            'insights': self._generate_language_insights(language_analysis)
        }
    
    def _analyze_responsiveness(self, df: pd.DataFrame) -> Dict:
        """Analyze areas showing high responsiveness or lack thereof."""
        if 'text' not in df.columns:
            return {}
        
        # Define responsiveness indicators
        responsiveness_indicators = {
            'management_response': ['management', 'owner', 'response', 'replied', 'thank you', 'address'],
            'service_recovery': ['apologize', 'sorry', 'improve', 'fixed', 'resolved', 'better'],
            'proactive_service': ['anticipate', 'exceeded', 'surprise', 'unexpected', 'extra'],
            'feedback_acknowledgment': ['feedback', 'suggestion', 'comment', 'noted', 'appreciated']
        }
        
        responsiveness_results = {}
        
        for indicator, keywords in responsiveness_indicators.items():
            pattern = '|'.join(keywords)
            responsive_mask = df['text'].str.lower().str.contains(pattern, na=False)
            responsive_df = df[responsive_mask]
            
            if len(responsive_df) > 0:
                responsiveness_results[indicator] = {
                    'mention_count': len(responsive_df),
                    'percentage': float((len(responsive_df) / len(df)) * 100),
                    'average_sentiment': float(responsive_df['sentiment_score'].mean()) if 'sentiment_score' in responsive_df.columns else None,
                    'examples': self._get_responsiveness_examples(responsive_df, keywords)
                }
        
        # Calculate overall responsiveness score
        total_responsive_mentions = sum(data['mention_count'] for data in responsiveness_results.values())
        responsiveness_score = (total_responsive_mentions / len(df)) * 100
        
        return {
            'overall_responsiveness_score': float(responsiveness_score),
            'responsiveness_areas': responsiveness_results,
            'recommendations': self._generate_responsiveness_recommendations(responsiveness_results, df)
        }
    
    def _analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze temporal patterns in reviews."""
        if 'date' not in df.columns:
            return {}
        
        # This is a simplified temporal analysis
        # In a full implementation, you'd parse dates properly
        temporal_data = {
            'review_frequency': 'Analysis would require proper date parsing',
            'seasonal_patterns': 'Requires date field parsing',
            'recent_trends': 'Would compare recent vs older reviews'
        }
        
        return temporal_data
    
    def _generate_executive_summary(self, insights: Dict) -> Dict:
        """Generate executive summary for tourism board."""
        summary = {
            'key_findings': [],
            'strengths': [],
            'areas_for_improvement': [],
            'strategic_recommendations': [],
            'competitive_positioning': []
        }
        
        # Overall sentiment insights
        overall = insights.get('overall_sentiment', {})
        if overall:
            sentiment_score = overall.get('overall_score', 0)
            positive_pct = overall.get('sentiment_distribution', {}).get('positive_percentage', 0)
            
            if sentiment_score > 0.3:
                summary['strengths'].append(f"Strong overall visitor satisfaction (sentiment: {sentiment_score:.2f})")
            elif sentiment_score < 0:
                summary['areas_for_improvement'].append(f"Below-average visitor satisfaction (sentiment: {sentiment_score:.2f})")
            
            summary['key_findings'].append(f"{positive_pct:.1f}% of reviews express positive sentiment")
        
        # Aspect-based insights
        aspects = insights.get('aspect_sentiment', {})
        for aspect, data in aspects.items():
            avg_sentiment = data.get('average_sentiment', 0)
            mention_pct = data.get('mention_percentage', 0)
            
            if avg_sentiment > 0.4 and mention_pct > 10:
                summary['strengths'].append(f"{aspect.title()}: Strong performance with {avg_sentiment:.2f} sentiment score")
            elif avg_sentiment < 0 and mention_pct > 5:
                summary['areas_for_improvement'].append(f"{aspect.title()}: Needs attention (negative sentiment: {avg_sentiment:.2f})")
        
        # Theme-based insights
        themes = insights.get('recurring_themes', {})
        for theme, data in themes.items():
            if data.get('percentage', 0) > 15:
                avg_sentiment = data.get('average_sentiment', 0)
                if avg_sentiment > 0.2:
                    summary['strengths'].append(f"Positive theme: {data['description']}")
                elif avg_sentiment < -0.2:
                    summary['areas_for_improvement'].append(f"Concerning theme: {data['description']}")
        
        # Language diversity insights
        lang_analysis = insights.get('language_analysis', {})
        if lang_analysis:
            international_appeal = lang_analysis.get('international_appeal', 0)
            if international_appeal > 30:
                summary['strengths'].append(f"Strong international appeal ({international_appeal:.1f}% non-English reviews)")
            elif international_appeal < 10:
                summary['strategic_recommendations'].append("Consider marketing to international audiences")
        
        # Responsiveness insights
        responsiveness = insights.get('responsiveness_analysis', {})
        if responsiveness:
            resp_score = responsiveness.get('overall_responsiveness_score', 0)
            if resp_score > 20:
                summary['strengths'].append(f"Good service responsiveness ({resp_score:.1f}% of reviews mention responsive service)")
            elif resp_score < 5:
                summary['areas_for_improvement'].append("Low service responsiveness - consider staff training")
        
        return summary
    
    # Helper methods
    def _calculate_confidence_level(self, sample_size: int) -> str:
        """Calculate confidence level based on sample size."""
        if sample_size >= 100:
            return "High"
        elif sample_size >= 30:
            return "Medium"
        else:
            return "Low"
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection based on keyword patterns."""
        if not text:
            return 'unknown'
        
        text_lower = text.lower()
        
        for language, patterns in self.language_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                return language
        
        return 'english'  # Default assumption
    
    def _calculate_word_sentiment(self, df: pd.DataFrame, word: str) -> float:
        """Calculate average sentiment for reviews containing a specific word."""
        if 'text' not in df.columns or 'sentiment_score' not in df.columns:
            return 0.0
        
        word_mask = df['text'].str.lower().str.contains(word, na=False)
        word_df = df[word_mask]
        
        if len(word_df) > 0:
            return float(word_df['sentiment_score'].mean())
        return 0.0
    
    def _get_example_phrases(self, df: pd.DataFrame, word: str, limit: int = 3) -> List[str]:
        """Get example phrases containing a specific word."""
        if 'text' not in df.columns:
            return []
        
        examples = []
        for text in df['text'].fillna(''):
            sentences = sent_tokenize(str(text))
            for sentence in sentences:
                if word.lower() in sentence.lower() and len(sentence) < 150:
                    examples.append(sentence.strip())
                    if len(examples) >= limit:
                        break
            if len(examples) >= limit:
                break
        
        return examples
    
    def _extract_key_phrases_for_aspect(self, df: pd.DataFrame, keywords: List[str]) -> List[str]:
        """Extract key phrases for a specific aspect."""
        phrases = []
        
        for text in df['text'].fillna('').head(10):  # Limit to first 10 reviews
            sentences = sent_tokenize(str(text))
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in keywords):
                    if len(sentence) < 100:  # Keep phrases reasonably short
                        phrases.append(sentence.strip())
        
        return phrases[:5]  # Return top 5 phrases
    
    def _get_theme_examples(self, df: pd.DataFrame, keywords: List[str]) -> List[str]:
        """Get example reviews for a theme."""
        examples = []
        
        for text in df['text'].fillna('').head(5):
            sentences = sent_tokenize(str(text))
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in keywords) and len(sentence) > 20:
                    examples.append(sentence.strip())
                    break
        
        return examples
    
    def _get_responsiveness_examples(self, df: pd.DataFrame, keywords: List[str]) -> List[str]:
        """Get examples of responsiveness mentions."""
        examples = []
        
        for text in df['text'].fillna('').head(3):
            sentences = sent_tokenize(str(text))
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in keywords):
                    examples.append(sentence.strip())
                    break
        
        return examples
    
    def _generate_language_insights(self, language_analysis: Dict) -> List[str]:
        """Generate insights about language diversity."""
        insights = []
        
        total_languages = len(language_analysis)
        if total_languages > 5:
            insights.append(f"High language diversity with {total_languages} languages represented")
        
        non_english_pct = sum(data['percentage'] for lang, data in language_analysis.items() 
                             if lang != 'english')
        
        if non_english_pct > 40:
            insights.append("Strong international visitor base - consider multilingual services")
        elif non_english_pct < 10:
            insights.append("Opportunity to attract more international visitors")
        
        return insights
    
    def _generate_responsiveness_recommendations(self, responsiveness_results: Dict, df: pd.DataFrame) -> List[str]:
        """Generate recommendations based on responsiveness analysis."""
        recommendations = []
        
        mgmt_response = responsiveness_results.get('management_response', {}).get('percentage', 0)
        if mgmt_response < 5:
            recommendations.append("Increase management engagement with guest feedback")
        
        service_recovery = responsiveness_results.get('service_recovery', {}).get('percentage', 0)
        if service_recovery < 3:
            recommendations.append("Implement service recovery training for staff")
        
        if not responsiveness_results:
            recommendations.append("Consider implementing guest feedback response system")
        
        return recommendations
    
    def save_insights_report(self, insights: Dict, filename: str = None) -> str:
        """Save tourism insights to JSON file."""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            destination = insights.get('analysis_metadata', {}).get('destination', 'unknown')
            filename = f"tourism_insights_{destination}_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(insights, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Tourism insights saved to: {filepath}")
        return filepath

# Convenience function
def analyze_tourism_insights(tripadvisor_data: Dict, output_dir: str = 'outputs/tourism_insights') -> Dict:
    """
    Quick function to analyze tourism insights from TripAdvisor data.
    
    Args:
        tripadvisor_data (Dict): TripAdvisor extraction data
        output_dir (str): Output directory for results
        
    Returns:
        Dict: Tourism insights analysis
    """
    analyzer = TourismInsightsAnalyzer(output_dir=output_dir)
    insights = analyzer.analyze_destination_reviews(tripadvisor_data)
    
    # Save the report
    if insights:
        analyzer.save_insights_report(insights)
    
    return insights 