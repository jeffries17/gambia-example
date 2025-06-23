import json
import os
import sys
from review_analyzer import GambiaTripAdvisorAnalyzer

def main():
    try:
        from review_analyzer import GambiaTripAdvisorAnalyzer
    except ImportError as e:
        print(f"Error importing required packages: {str(e)}")
        print("Make sure you've installed the required packages with:")
        print("pip install pandas nltk matplotlib seaborn")
        sys.exit(1)
    
    # Configuration
    input_file = 'your_hotel_reviews.json'  # Your downloaded JSON file
    output_dir = 'analysis_results'
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize and run the analyzer without year filter
    analyzer = GambiaTripAdvisorAnalyzer(input_file)
    
    try:
        # Generate charts
        analyzer.generate_visualizations(output_dir)
        
        # Export processed data to CSV
        analyzer.export_to_csv(os.path.join(output_dir, 'processed_reviews.csv'))
        
        # Generate HTML report
        analyzer.generate_report(os.path.join(output_dir, 'analysis_report.html'))
        
        print(f"Analysis complete! Results saved to {output_dir}")
        print(f"Open {os.path.join(output_dir, 'analysis_report.html')} to view the report")
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        print("Please check for missing dependencies and ensure all NLTK packages are downloaded.")
        print("You may need to run:")
        print("python -c \"import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')\"")

if __name__ == "__main__":
    main()

def convert_format(data):
    """
    Convert different JSON formats to the expected format.
    Modify this function based on your specific JSON structure.
    """
    converted = []
    
    # Example conversion for TripAdvisor API format
    if isinstance(data, dict) and 'data' in data and 'reviews' in data['data']:
        # TripAdvisor API format
        for review in data['data']['reviews']:
            converted.append({
                'establishment_name': data['data'].get('name', 'Unknown'),
                'category': 'Accommodation',
                'location': data['data'].get('location', {}).get('name', 'Unknown'),
                'rating': review.get('rating', 0),
                'date': review.get('published_date', '2000-01-01'),
                'title': review.get('title', ''),
                'review_text': review.get('text', ''),
                'traveler_type': review.get('travel_type', 'Not specified'),
                'visit_date': review.get('travel_date', '')
            })
    
    # Another example for a different format
    elif isinstance(data, dict) and 'reviews' in data:
        for review in data['reviews']:
            converted.append({
                'establishment_name': data.get('hotelName', 'Unknown'),
                'category': 'Accommodation',
                'location': data.get('location', 'Unknown'),
                'rating': review.get('overallRating', 0),
                'date': review.get('reviewDate', '2000-01-01'),
                'title': review.get('reviewTitle', ''),
                'review_text': review.get('reviewText', ''),
                'traveler_type': review.get('travelerType', 'Not specified')
            })
    
    # If data is already an array of reviews but missing fields
    elif isinstance(data, list):
        for item in data:
            review = {
                'establishment_name': item.get('hotel_name', item.get('name', 'Unknown')),
                'category': 'Accommodation',
                'rating': item.get('rating', item.get('stars', 0)),
                'date': item.get('date', item.get('review_date', '2000-01-01')),
                'review_text': item.get('review_text', item.get('text', item.get('content', ''))),
                'traveler_type': item.get('traveler_type', 'Not specified')
            }
            converted.append(review)
    
    return converted

if __name__ == "__main__":
    main()