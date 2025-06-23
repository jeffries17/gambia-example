#!/usr/bin/env python3
"""
Goree Island Analysis Script
Analyzes the Goree Island reviews for competitor comparison
"""

import json
import sys
import os
from datetime import datetime

# Add the sentiment_analyzer to path
sys.path.append('sentiment_analyzer')
from analysis.tourism_insights_analyzer import TourismInsightsAnalyzer

def prepare_goree_data():
    """Load and structure Goree Island reviews for analysis."""
    print("Loading Goree Island reviews...")
    
    # Load reviews
    with open('sentiment_analyzer/goree_reviews.json', 'r', encoding='utf-8') as f:
        raw_reviews = json.load(f)
    
    print(f"Found {len(raw_reviews)} reviews")
    
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
            'total_reviews_extracted': len(raw_reviews)
        }
    }
    
    # Convert reviews to expected format
    for review in raw_reviews:
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
            'review_url': review.get('url', '')
        }
        structured_data['reviews'].append(processed_review)
    
    return structured_data

def analyze_goree():
    """Analyze Goree Island reviews."""
    print("Preparing Goree Island data...")
    goree_data = prepare_goree_data()
    
    print("Running tourism insights analysis...")
    analyzer = TourismInsightsAnalyzer()
    
    # Run analysis
    results = analyzer.analyze_destination_reviews(goree_data)
    
    # Save results (convert numpy types to native Python types)
    def convert_numpy_types(obj):
        """Convert numpy types to native Python types for JSON serialization."""
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
    
    output_file = 'goree_analysis_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)
    
    print(f"Analysis complete! Results saved to {output_file}")
    
    # Print summary
    if 'analysis_metadata' in results:
        metadata = results['analysis_metadata']
        print(f"\nDestination: {metadata['destination']}")
        print(f"Location: {metadata['location']}")
        print(f"Total reviews: {metadata['total_reviews']}")
    
    if 'overall_sentiment' in results:
        sentiment = results['overall_sentiment']
        print(f"Overall sentiment score: {sentiment.get('average_sentiment_score', 'N/A')}")
        print(f"Sentiment distribution: {sentiment.get('sentiment_distribution', {})}")
    
    return results

if __name__ == "__main__":
    try:
        results = analyze_goree()
        print("\nGoree Island analysis completed successfully!")
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc() 