#!/usr/bin/env python3
"""
Sentiment Analysis Module

This module provides the core sentiment analysis functionality,
leveraging the existing sentiment analysis codebase while adapting
it for use in a web application context.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Configure logger
logger = logging.getLogger(__name__)

# Add the research code to the path
RESEARCH_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'tonga_analysis')
if os.path.exists(RESEARCH_DIR):
    sys.path.append(RESEARCH_DIR)
    logger.info(f"Added research directory to path: {RESEARCH_DIR}")
else:
    logger.warning(f"Research directory not found: {RESEARCH_DIR}")

# Import existing analysis code (will be implemented once integrated)
try:
    # These imports will be updated to match the actual structure
    # import tonga_analysis.sentiment_analyzer as existing_analyzer
    pass
except ImportError as e:
    logger.error(f"Error importing research code: {str(e)}")

class ReviewAnalyzer:
    """
    Main review analyzer class that processes customer reviews
    and generates sentiment analysis reports.
    """
    
    def __init__(self):
        """Initialize the analyzer with default configuration"""
        self.output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Define analysis parameters
        self.aspect_categories = {
            'accommodations': [
                'cleanliness', 'comfort', 'location', 'facilities',
                'staff', 'value', 'breakfast', 'room', 'bathroom'
            ],
            'restaurants': [
                'food_quality', 'service', 'ambiance', 'value',
                'menu_variety', 'portion_size', 'taste', 'presentation'
            ],
            'attractions': [
                'experience', 'staff', 'value', 'facilities',
                'accessibility', 'uniqueness', 'education', 'enjoyment'
            ]
        }
    
    def analyze_text(self, text, category='general'):
        """
        Analyze raw text input (e.g., pasted reviews)
        
        Args:
            text (str): The review text to analyze
            category (str): The business category (accommodations/restaurants/attractions)
            
        Returns:
            dict: Analysis results
        """
        logger.info(f"Analyzing text input ({len(text)} chars) for category: {category}")
        
        # For demo purposes, return mock results
        # In production, this will use the existing sentiment analysis code
        return self._generate_mock_results(text, category)
    
    def analyze_file(self, file_path, category='general'):
        """
        Analyze reviews from a file (CSV, Excel, JSON)
        
        Args:
            file_path (str): Path to the file containing reviews
            category (str): The business category
            
        Returns:
            dict: Analysis results
        """
        logger.info(f"Analyzing file: {file_path} for category: {category}")
        
        # For demo, return mock results
        # In production, this will load the file and process it
        return self._generate_mock_results(f"File analysis: {file_path}", category)
    
    def analyze_url(self, url, max_reviews=50):
        """
        Analyze reviews from a URL (TripAdvisor, etc.)
        Note: This is for Phase 2
        
        Args:
            url (str): URL to scrape reviews from
            max_reviews (int): Maximum number of reviews to scrape
            
        Returns:
            dict: Analysis results
        """
        logger.info(f"Analyzing URL: {url} (max {max_reviews} reviews)")
        
        # For demo, return mock results
        # In production, this will scrape the URL and process reviews
        return self._generate_mock_results(f"URL analysis: {url}", "url_scrape")
    
    def _generate_mock_results(self, input_data, category):
        """
        Generate mock results for demonstration purposes
        
        Args:
            input_data: The input data (text, file, or URL)
            category: The business category
            
        Returns:
            dict: Mock analysis results
        """
        # Create a unique ID for this analysis
        analysis_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Choose aspects based on category
        if category in self.aspect_categories:
            aspects = self.aspect_categories[category]
        else:
            aspects = ['service', 'quality', 'value', 'location', 'experience']
        
        # Generate mock sentiment scores for aspects
        aspect_sentiments = {}
        for aspect in aspects:
            aspect_sentiments[aspect] = {
                'sentiment': round(np.random.uniform(0.3, 0.9), 2),
                'count': np.random.randint(5, 30),
                'positive_examples': [
                    f"Positive example for {aspect} 1",
                    f"Positive example for {aspect} 2"
                ],
                'negative_examples': [
                    f"Negative example for {aspect} 1"
                ]
            }
        
        # Generate mock results
        results = {
            'analysis_id': analysis_id,
            'timestamp': datetime.now().isoformat(),
            'input_type': 'text' if isinstance(input_data, str) else 'file',
            'category': category,
            'overall': {
                'review_count': np.random.randint(20, 100),
                'average_sentiment': round(np.random.uniform(0.5, 0.8), 2),
                'positive_percentage': round(np.random.uniform(60, 80), 1),
                'negative_percentage': round(np.random.uniform(10, 30), 1),
                'neutral_percentage': round(np.random.uniform(5, 15), 1)
            },
            'aspects': aspect_sentiments,
            'top_phrases': [
                {'text': 'great experience', 'sentiment': 0.92, 'count': 12},
                {'text': 'friendly staff', 'sentiment': 0.88, 'count': 10},
                {'text': 'beautiful location', 'sentiment': 0.85, 'count': 9},
                {'text': 'poor service', 'sentiment': -0.65, 'count': 5},
                {'text': 'clean rooms', 'sentiment': 0.78, 'count': 8}
            ]
        }
        
        # Save the results
        output_path = os.path.join(self.output_dir, f"{analysis_id}.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Analysis complete. Results saved to {output_path}")
        return results

# Singleton instance
analyzer = ReviewAnalyzer()

def analyze(data, input_type='text', category='general', **kwargs):
    """
    Unified analysis function for all input types
    
    Args:
        data: The input data (text, file path, or URL)
        input_type: Type of input ('text', 'file', 'url')
        category: Business category for analysis
        **kwargs: Additional arguments based on input type
        
    Returns:
        dict: Analysis results
    """
    if input_type == 'text':
        return analyzer.analyze_text(data, category)
    elif input_type == 'file':
        return analyzer.analyze_file(data, category)
    elif input_type == 'url':
        max_reviews = kwargs.get('max_reviews', 50)
        return analyzer.analyze_url(data, max_reviews)
    else:
        raise ValueError(f"Unsupported input type: {input_type}")

if __name__ == "__main__":
    # Simple test
    test_text = "The hotel was amazing. Great service and beautiful rooms. However, the breakfast was disappointing."
    results = analyze(test_text, category='accommodations')
    print(json.dumps(results, indent=2)) 