import pandas as pd
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import os
import re
from urllib.parse import urlparse, unquote

class ReviewWorkflow:
    def __init__(self, scraper=None, output_dir: str = 'output'):
        """Initialize workflow with custom scraper or use default"""
        # Import inside method to avoid circular imports
        if scraper is None:
            from tripscrape import TripScape
            self.scraper = TripScape()
        else:
            self.scraper = scraper
            
        self.output_dir = output_dir
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

    def extract_business_name(self, url: str) -> str:
        """Extract business name from TripAdvisor URL"""
        try:
            # Parse URL and get the path
            path = urlparse(url).path
            
            # Extract the business name part using regex
            match = re.search(r'Reviews-([^-]+)-', path)
            if match:
                # Get encoded name and decode it
                encoded_name = match.group(1)
                return unquote(encoded_name)
                
            # Alternative extraction from the last segment
            segments = [s for s in path.split('-') if s]
            if segments:
                # Last segment is often the location, second-to-last is the business
                business_part = segments[-2] if len(segments) >= 2 else segments[-1]
                return business_part.replace('_', ' ')
                
            return "unknown_business"
        except Exception as e:
            print(f"Error extracting business name: {e}")
            return "unknown_business"

    def detect_page_type(self, url: str) -> str:
        """Detect if URL is for hotel, restaurant, or attraction"""
        if 'Hotel_Review' in url:
            return 'hotel'
        elif 'Restaurant_Review' in url:
            return 'restaurant'
        elif 'Attraction_Review' in url:
            return 'attraction'
        else:
            return 'unknown'

    def process_url(self, url: str) -> Tuple[pd.DataFrame, str]:
        """Process a single URL and return results as DataFrame with business name"""
        start_time = time.time()
        business_name = self.extract_business_name(url)
        page_type = self.detect_page_type(url)
        
        print(f"\nProcessing: {business_name} ({page_type})")
        print(f"URL: {url}")
        
        try:
            # Get all reviews
            reviews = self.scraper.get_all_reviews(url)
            if not reviews:
                print(f"No reviews found for {business_name}")
                return None, business_name
            
            # Convert to DataFrame
            df = pd.DataFrame(reviews)
            
            # Add metadata
            df['scraped_date'] = datetime.now().strftime('%Y-%m-%d')
            df['source_url'] = url
            df['business_name'] = business_name
            df['page_type'] = page_type
            
            process_time = time.time() - start_time
            print(f"Retrieved {len(df)} reviews in {process_time:.1f} seconds")
            
            return df, business_name
            
        except Exception as e:
            print(f"Error processing {url}: {str(e)}")
            return None, business_name

    def save_results(self, df: pd.DataFrame, business_name: str = None):
        """Save results with detailed statistics using business name"""
        if df is None or df.empty:
            print("No data to save!")
            return
        
        # Create safe filename from business name
        if business_name:
            safe_name = re.sub(r'[^\w\s-]', '', business_name).strip().replace(' ', '_')
        else:
            safe_name = "reviews"
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{safe_name}_{timestamp}.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        # Save main CSV file
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"\nSaved {len(df)} reviews to {filepath}")
        
        # Generate and save statistics
        self._save_statistics(df, filepath)
        
        return filepath
    
    def _save_statistics(self, df: pd.DataFrame, filepath: str):
        """Generate and save detailed statistics"""
        stats_file = filepath.replace('.csv', '_stats.txt')
        
        try:
            with open(stats_file, 'w', encoding='utf-8') as f:
                f.write(f"Statistics Report\n")
                f.write(f"=================\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Business info
                if 'business_name' in df.columns:
                    f.write(f"Business: {df['business_name'].iloc[0]}\n")
                if 'page_type' in df.columns:
                    f.write(f"Type: {df['page_type'].iloc[0]}\n")
                
                # Basic metrics
                f.write(f"Total Reviews: {len(df)}\n")
                f.write(f"Average Rating: {df['rating'].mean():.2f}/5.0\n\n")
                
                # Reviews by Rating
                f.write("Reviews by Rating:\n")
                f.write("----------------\n")
                rating_counts = df['rating'].value_counts().sort_index()
                for rating, count in rating_counts.items():
                    percentage = (count / len(df)) * 100
                    f.write(f"{rating:.1f} stars: {count} ({percentage:.1f}%)\n")
                
                # Reviews by Month
                f.write("\nReviews by Month:\n")
                f.write("---------------\n")
                try:
                    df['month'] = pd.to_datetime(df['date'], format='%b %Y', errors='coerce').dt.strftime('%Y-%m')
                    month_counts = df['month'].value_counts().sort_index()
                    for month, count in month_counts.items():
                        if pd.notna(month):
                            f.write(f"{month}: {count}\n")
                except Exception as e:
                    f.write(f"Error processing dates: {str(e)}\n")
                
                # Trip Types
                if 'trip_type' in df.columns:
                    f.write("\nTrip Types:\n")
                    f.write("-----------\n")
                    trip_types = df['trip_type'].fillna('Unknown').value_counts()
                    for trip_type, count in trip_types.items():
                        percentage = (count / len(df)) * 100
                        f.write(f"{trip_type}: {count} ({percentage:.1f}%)\n")
                
                # Location distribution
                if 'location' in df.columns:
                    locations = df['location'].fillna('Unknown').value_counts()
                    if len(locations) <= 20:  # Only show if not too many locations
                        f.write("\nReviewer Locations:\n")
                        f.write("-----------------\n")
                        for location, count in locations.items()[:10]:  # Top 10
                            percentage = (count / len(df)) * 100
                            f.write(f"{location}: {count} ({percentage:.1f}%)\n")
                
            print(f"Statistics saved to {stats_file}")
            
        except Exception as e:
            print(f"Error saving statistics: {str(e)}")


# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='TripAdvisor Review Scraper')
    parser.add_argument('url', help='TripAdvisor URL to scrape')
    parser.add_argument('--output', '-o', default='tripadvisor_data', 
                      help='Output directory for scraped data')
    parser.add_argument('--max-pages', '-m', type=int, default=None,
                      help='Maximum number of pages to scrape (default: all)')
    
    args = parser.parse_args()
    
    workflow = ReviewWorkflow(output_dir=args.output)
    results, business_name = workflow.process_url(args.url)
    if results is not None:
        workflow.save_results(results, business_name)
    else:
        print("Scraping failed - no data retrieved")