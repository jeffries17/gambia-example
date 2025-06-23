#!/usr/bin/env python3
"""
Test script for the improved TripAdvisor extractor
Demonstrates usage with Gambia destination example
"""

import sys
import os

# Add the sentiment_analyzer directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'sentiment_analyzer'))

from sentiment_analyzer.extractors.improved_tripadvisor_extractor import extract_reviews_from_url

def test_gambia_extraction():
    """Test extraction with a Gambia destination URL."""
    
    # Example URLs for The Gambia (you'll need to find actual TripAdvisor URLs)
    test_urls = [
        # You would replace these with actual TripAdvisor URLs for Gambia destinations
        # "https://www.tripadvisor.com/Hotel_Review-g293794-d123456-Reviews-Kunta_Kinteh_Island-Gambia.html",
        # "https://www.tripadvisor.com/Restaurant_Review-g293794-d789012-Reviews-Restaurant_Name-Gambia.html",
        # "https://www.tripadvisor.com/Attraction_Review-g293794-d345678-Reviews-Kunta_Kinteh_Island-Gambia.html"
    ]
    
    print("=== Testing Improved TripAdvisor Extractor ===")
    print("Note: Add actual TripAdvisor URLs for Gambia destinations to test")
    
    if not test_urls:
        print("\nTo test the extractor:")
        print("1. Find TripAdvisor URLs for Gambia destinations")
        print("2. Add them to the test_urls list in this script")
        print("3. Run the script again")
        
        print("\nExample usage:")
        print("python test_improved_extractor.py 'https://www.tripadvisor.com/Hotel_Review-...'")
        return
    
    for i, url in enumerate(test_urls, 1):
        print(f"\n--- Test {i}: Extracting from {url} ---")
        
        try:
            # Extract with limits for testing
            result = extract_reviews_from_url(
                url=url,
                max_reviews=50,  # Limit for testing
                max_pages=5,     # Limit pages for testing
                save_file=True   # Save results to file
            )
            
            # Print summary
            business_info = result.get('business_info', {})
            reviews = result.get('reviews', [])
            
            print(f"✅ Success!")
            print(f"   Business: {business_info.get('name', 'Unknown')}")
            print(f"   Category: {business_info.get('category', 'Unknown')}")
            print(f"   Location: {business_info.get('location', 'Unknown')}")
            print(f"   Reviews extracted: {len(reviews)}")
            
            if reviews:
                avg_rating = sum(r.get('rating', 0) for r in reviews if r.get('rating')) / len([r for r in reviews if r.get('rating')])
                print(f"   Average rating: {avg_rating:.1f}")
                
                # Show sample review
                sample_review = reviews[0]
                print(f"   Sample review title: {sample_review.get('title', 'No title')[:100]}...")
                print(f"   Sample review text: {sample_review.get('text', 'No text')[:200]}...")
            
        except Exception as e:
            print(f"❌ Failed: {str(e)}")

def test_single_url(url):
    """Test extraction from a single URL provided as command line argument."""
    print(f"=== Testing Single URL ===")
    print(f"URL: {url}")
    
    try:
        result = extract_reviews_from_url(
            url=url,
            max_reviews=100,  # Moderate limit
            max_pages=10,     # Moderate page limit
            save_file=True    # Save results
        )
        
        # Print detailed results
        metadata = result.get('extraction_metadata', {})
        business_info = result.get('business_info', {})
        reviews = result.get('reviews', [])
        
        print(f"\n=== Extraction Results ===")
        print(f"Business Name: {business_info.get('name', 'Unknown')}")
        print(f"Category: {business_info.get('category', 'Unknown')}")
        print(f"Location: {business_info.get('location', 'Unknown')}")
        print(f"Total Reviews Found: {metadata.get('total_reviews_found', 0)}")
        print(f"Reviews Extracted: {len(reviews)}")
        print(f"Pages Processed: {metadata.get('pages_processed', 0)}")
        
        if reviews:
            # Calculate statistics
            ratings = [r.get('rating') for r in reviews if r.get('rating')]
            if ratings:
                avg_rating = sum(ratings) / len(ratings)
                print(f"Average Rating: {avg_rating:.2f}")
            
            # Show distribution of trip types
            trip_types = [r.get('trip_type', 'Unknown') for r in reviews]
            trip_type_counts = {}
            for tt in trip_types:
                trip_type_counts[tt] = trip_type_counts.get(tt, 0) + 1
            
            print(f"\nTrip Type Distribution:")
            for trip_type, count in sorted(trip_type_counts.items()):
                print(f"  {trip_type}: {count}")
            
            # Show sample reviews
            print(f"\n=== Sample Reviews ===")
            for i, review in enumerate(reviews[:3], 1):
                print(f"\nReview {i}:")
                print(f"  Title: {review.get('title', 'No title')}")
                print(f"  Rating: {review.get('rating', 'No rating')}")
                print(f"  Date: {review.get('date', 'No date')}")
                print(f"  Trip Type: {review.get('trip_type', 'Unknown')}")
                review_text = review.get('text', 'No text')
                print(f"  Text: {review_text[:300]}{'...' if len(review_text) > 300 else ''}")
        
        return result
        
    except Exception as e:
        print(f"❌ Extraction failed: {str(e)}")
        return None

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Test with URL provided as command line argument
        url = sys.argv[1]
        test_single_url(url)
    else:
        # Run the predefined tests
        test_gambia_extraction()
        
        # Show usage instructions
        print("\n" + "="*60)
        print("USAGE INSTRUCTIONS:")
        print("="*60)
        print("1. To test with a specific URL:")
        print("   python test_improved_extractor.py 'https://www.tripadvisor.com/Hotel_Review-...'")
        print("\n2. To find TripAdvisor URLs for Gambia:")
        print("   - Go to tripadvisor.com")
        print("   - Search for 'Gambia hotels' or 'Gambia attractions'")
        print("   - Click on a specific business page")
        print("   - Copy the URL from your browser")
        print("\n3. The extractor will:")
        print("   - Extract business information")
        print("   - Collect reviews with ratings, dates, text")
        print("   - Handle pagination automatically")
        print("   - Save results to JSON file")
        print("   - Apply anti-detection measures") 