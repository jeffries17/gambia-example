#!/usr/bin/env python3
"""
Apify vs Our Approach Comparison

This script analyzes the Gambia reviews extracted by Apify and shows 
how we can replicate their sophisticated extraction approach.
"""

import json
import sys
import os
from collections import Counter
from datetime import datetime

# Add the sentiment_analyzer directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'sentiment_analyzer'))

def analyze_apify_data(filepath: str):
    """Analyze Apify's extracted data structure and quality."""
    
    print("🔍 ANALYZING APIFY'S TRIPADVISOR EXTRACTION")
    print("=" * 60)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reviews = json.load(f)
    
    print(f"📊 BASIC STATISTICS:")
    print(f"   Total reviews: {len(reviews)}")
    
    if not reviews:
        print("   No reviews found in file")
        return
    
    # Analyze data completeness
    print(f"\n📋 DATA COMPLETENESS ANALYSIS:")
    
    fields_analysis = {
        'title': 0,
        'text': 0,
        'rating': 0,
        'travelDate': 0,
        'publishedDate': 0,
        'user.username': 0,
        'user.userLocation': 0,
        'user.contributions': 0,
        'user.avatar': 0,
        'placeInfo.name': 0,
        'placeInfo.rating': 0,
        'placeInfo.numberOfReviews': 0,
        'url': 0,
        'ownerResponse': 0
    }
    
    for review in reviews:
        if review.get('title'): fields_analysis['title'] += 1
        if review.get('text'): fields_analysis['text'] += 1
        if review.get('rating'): fields_analysis['rating'] += 1
        if review.get('travelDate'): fields_analysis['travelDate'] += 1
        if review.get('publishedDate'): fields_analysis['publishedDate'] += 1
        if review.get('url'): fields_analysis['url'] += 1
        if review.get('ownerResponse'): fields_analysis['ownerResponse'] += 1
        
        user = review.get('user', {})
        if user and user.get('username'): fields_analysis['user.username'] += 1
        if user and user.get('userLocation'): fields_analysis['user.userLocation'] += 1
        if user and user.get('contributions'): fields_analysis['user.contributions'] += 1
        if user and user.get('avatar'): fields_analysis['user.avatar'] += 1
        
        place = review.get('placeInfo', {})
        if place and place.get('name'): fields_analysis['placeInfo.name'] += 1
        if place and place.get('rating'): fields_analysis['placeInfo.rating'] += 1
        if place and place.get('numberOfReviews'): fields_analysis['placeInfo.numberOfReviews'] += 1
    
    total_reviews = len(reviews)
    for field, count in fields_analysis.items():
        percentage = (count / total_reviews) * 100
        status = "✅" if percentage > 90 else "⚠️" if percentage > 50 else "❌"
        print(f"   {status} {field}: {count}/{total_reviews} ({percentage:.1f}%)")
    
    # Analyze place information
    print(f"\n🏢 PLACE INFORMATION:")
    if reviews:
        place_info = reviews[0].get('placeInfo', {})
        print(f"   Name: {place_info.get('name', 'Unknown')}")
        print(f"   Overall Rating: {place_info.get('rating', 'Unknown')}")
        print(f"   Total Reviews: {place_info.get('numberOfReviews', 'Unknown')}")
        print(f"   Location: {place_info.get('locationString', 'Unknown')}")
        
        # Rating histogram
        histogram = place_info.get('ratingHistogram', {})
        if histogram:
            print(f"   Rating Distribution:")
            for i in range(1, 6):
                count = histogram.get(f'count{i}', 0)
                print(f"     {i} stars: {count}")
    
    # Analyze review ratings
    print(f"\n⭐ REVIEW RATINGS:")
    ratings = [r.get('rating') for r in reviews if r.get('rating')]
    if ratings:
        rating_counts = Counter(ratings)
        avg_rating = sum(ratings) / len(ratings)
        print(f"   Average Rating: {avg_rating:.2f}")
        print(f"   Rating Distribution:")
        for rating in sorted(rating_counts.keys()):
            count = rating_counts[rating]
            percentage = (count / len(ratings)) * 100
            print(f"     {rating} stars: {count} ({percentage:.1f}%)")
    
    # Analyze user locations
    print(f"\n🌍 USER LOCATIONS:")
    locations = []
    for review in reviews:
        user = review.get('user', {})
        location = user.get('userLocation', {})
        if location and location.get('name'):
            locations.append(location['name'])
    
    if locations:
        location_counts = Counter(locations)
        print(f"   Total unique locations: {len(location_counts)}")
        print(f"   Top locations:")
        for location, count in location_counts.most_common(5):
            percentage = (count / len(locations)) * 100
            print(f"     {location}: {count} ({percentage:.1f}%)")
    
    # Analyze languages
    print(f"\n🗣️ REVIEW LANGUAGES:")
    languages = []
    for review in reviews:
        text = review.get('text', '')
        # Simple language detection based on common words
        if any(word in text.lower() for word in ['the', 'and', 'is', 'was', 'were']):
            languages.append('English')
        elif any(word in text.lower() for word in ['de', 'het', 'een', 'is', 'was', 'waren']):
            languages.append('Dutch')
        elif any(word in text.lower() for word in ['le', 'la', 'les', 'est', 'était']):
            languages.append('French')
        else:
            languages.append('Other')
    
    if languages:
        lang_counts = Counter(languages)
        for lang, count in lang_counts.items():
            percentage = (count / len(languages)) * 100
            print(f"   {lang}: {count} ({percentage:.1f}%)")
    
    # Analyze date patterns
    print(f"\n📅 DATE ANALYSIS:")
    travel_dates = [r.get('travelDate') for r in reviews if r.get('travelDate')]
    published_dates = [r.get('publishedDate') for r in reviews if r.get('publishedDate')]
    
    print(f"   Travel dates available: {len(travel_dates)}/{total_reviews}")
    print(f"   Published dates available: {len(published_dates)}/{total_reviews}")
    
    if travel_dates:
        # Extract years from travel dates
        years = []
        for date in travel_dates:
            if date and len(date) >= 4:
                year = date[:4]
                if year.isdigit():
                    years.append(int(year))
        
        if years:
            year_counts = Counter(years)
            print(f"   Travel date range: {min(years)} - {max(years)}")
            print(f"   Most common travel years:")
            for year, count in year_counts.most_common(3):
                print(f"     {year}: {count} reviews")

def show_apify_advantages():
    """Show what makes Apify's approach superior."""
    
    print(f"\n🚀 APIFY'S KEY ADVANTAGES:")
    print("=" * 60)
    
    advantages = [
        ("Rich Data Structure", "Comprehensive user profiles, place metadata, rating histograms"),
        ("Professional Anti-Detection", "Proxy rotation, browser fingerprinting, smart delays"),
        ("Scalability", "Can handle thousands of reviews across multiple pages"),
        ("Data Quality", "90%+ completion rates for most fields"),
        ("Multilingual Support", "Handles reviews in multiple languages"),
        ("Business Intelligence", "Includes rating histograms, contribution counts, etc."),
        ("Reliability", "99%+ success rate according to their metrics"),
        ("API Integration", "RESTful API, webhooks, multiple output formats"),
        ("Cost Efficiency", "$2/1000 reviews with infrastructure included"),
        ("Maintenance", "No need to update selectors when TripAdvisor changes")
    ]
    
    for i, (title, description) in enumerate(advantages, 1):
        print(f"   {i:2d}. {title}")
        print(f"       {description}")

def show_replication_strategy():
    """Show how we can replicate Apify's approach."""
    
    print(f"\n🛠️ HOW TO REPLICATE APIFY'S APPROACH:")
    print("=" * 60)
    
    strategies = [
        ("Data Structure", "Match their JSON schema exactly", "✅ Implemented in apify_inspired_extractor.py"),
        ("Anti-Detection", "Implement proxy rotation and better headers", "⚠️ Partial - need proxy service"),
        ("User Profiles", "Extract detailed user information", "✅ Implemented with user details"),
        ("Place Metadata", "Extract rating histograms and business info", "✅ Implemented place info extraction"),
        ("Error Handling", "Robust retry logic with exponential backoff", "✅ Implemented"),
        ("Pagination", "Smart page limit detection", "✅ Implemented"),
        ("Language Detection", "Identify review languages", "🔄 Can be added"),
        ("URL Construction", "Handle all TripAdvisor URL patterns", "✅ Implemented"),
        ("Data Validation", "Ensure data quality and completeness", "🔄 Can be improved"),
        ("Performance", "Parallel processing and caching", "🔄 Future enhancement")
    ]
    
    for strategy, description, status in strategies:
        print(f"   {status} {strategy}: {description}")

def main():
    """Main function to run the analysis."""
    
    gambia_file = "sentiment_analyzer/gambia_reviews.json"
    
    if not os.path.exists(gambia_file):
        print(f"❌ File not found: {gambia_file}")
        print("   Please ensure the Gambia reviews file exists")
        return
    
    # Analyze Apify's data
    analyze_apify_data(gambia_file)
    
    # Show advantages
    show_apify_advantages()
    
    # Show replication strategy
    show_replication_strategy()
    
    print(f"\n💡 RECOMMENDATIONS:")
    print("=" * 60)
    print("1. Use our apify_inspired_extractor.py for Apify-compatible data")
    print("2. Consider using a proxy service for large-scale extractions")
    print("3. Implement language detection for multilingual reviews")
    print("4. Add parallel processing for multiple URLs")
    print("5. Create a validation layer to ensure data quality")
    print("6. Monitor TripAdvisor's selector changes and update accordingly")
    
    print(f"\n🎯 NEXT STEPS:")
    print("=" * 60)
    print("1. Test the apify_inspired_extractor.py with Gambia URLs")
    print("2. Compare output quality with Apify's results")
    print("3. Move to sentiment analysis of the extracted reviews")
    print("4. Scale to multiple destinations for comparison")

if __name__ == "__main__":
    main() 