import requests
from typing import List, Dict, Optional
from parsel import Selector
import pandas as pd
from datetime import datetime
import time
import random
import logging
import re

class TripScape:
    def __init__(self):
        """Initialize the TripScape scraper with necessary configuration."""
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Origin': 'https://www.tripadvisor.com',
            'Referer': 'https://www.tripadvisor.com',
            'sec-ch-ua': '"Not A(Brand";v="99", "Google Chrome";v="121", "Chromium";v="121"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'Upgrade-Insecure-Requests': '1'
        }
        self.seen_reviews = set()
        self.setup_logging()

    def setup_logging(self):
        """Configure logging for tracking scraping progress."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('tripadvisor_scraping.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def get_page(self, url: str, retry_count: int = 3) -> Optional[Selector]:
        """Fetch a page with retry logic."""
        for attempt in range(retry_count):
            try:
                self.logger.info(f"Fetching URL: {url}")
                time.sleep(random.uniform(2, 4))
                response = requests.get(url, headers=self.headers, timeout=30)
                response.raise_for_status()
                self.logger.info(f"Successfully fetched page: {response.status_code}")
                return Selector(response.text)
            except requests.RequestException as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < retry_count - 1:
                    sleep_time = 5 * (attempt + 1)
                    self.logger.info(f"Waiting {sleep_time} seconds before retry...")
                    time.sleep(sleep_time)
        return None

    def extract_review_count(self, selector: Selector) -> int:
        """
        Extract the total number of reviews from the page.
        Returns 0 if no valid count is found.
        """
        try:
            # First try to get the review count text
            review_count_text = selector.css('[data-automation="reviewCount"]::text').get('')
            self.logger.info(f"Found review count text: {review_count_text}")
            
            # Extract just the number using regex
            match = re.search(r'(\d+(?:,\d+)?)', review_count_text)
            if match:
                # Remove commas and convert to integer
                count = int(match.group(1).replace(',', ''))
                self.logger.info(f"Extracted review count: {count}")
                return count
            
            self.logger.warning(f"Could not extract number from review count text: {review_count_text}")
            return 0
            
        except Exception as e:
            self.logger.error(f"Error extracting review count: {str(e)}")
            return 0

    def extract_single_review(self, review: Selector) -> Optional[Dict]:
        """Extract data from a single review element."""
        try:
            # Extract basic review information with proper error handling
            username = review.css('div.QIHsu.Zb span.biGQs._P.fiohW.fOtGX a::text').get('').strip()
            location = review.css('div.QIHsu.Zb div.biGQs._P.pZUbB.osNWb span::text').get('').strip()
            
            # Get review title and text
            title = review.css('div.biGQs._P.fiohW.qWPrE.ncFvv.fOtGX span.yCeTE::text').get('').strip()
            text = review.css('div.biGQs._P.pZUbB.KxBGd span.JguWG span.yCeTE::text').get('')
            if not text:
                text = review.css('div.fIrGe._T.bgMZj div.biGQs._P.pZUbB.KxBGd span.JguWG span.yCeTE::text').get('')
            text = text.strip() if text else ''

            # Extract rating
            rating_text = review.css('svg.UctUV title::text').get('')
            rating = None
            if rating_text:
                match = re.search(r'(\d+(?:\.\d+)?)', rating_text)
                if match:
                    rating = float(match.group(1))

            # Get date and trip type
            date_element = review.css('div.RpeCd::text').get('')
            date = ''
            trip_type = ''
            if date_element:
                parts = date_element.split('•')
                date = parts[0].strip()
                trip_type = parts[1].strip() if len(parts) > 1 else ''

            review_data = {
                'username': username,
                'location': location,
                'title': title,
                'text': text,
                'rating': rating,
                'date': date,
                'trip_type': trip_type
            }

            self.logger.debug(f"Extracted review from {username}")
            return review_data

        except Exception as e:
            self.logger.error(f"Error extracting review: {str(e)}")
            return None

    def get_all_reviews(self, url: str, max_pages: Optional[int] = None) -> List[Dict]:
        """Get all reviews from the page with pagination."""
        self.logger.info(f"Starting review collection from: {url}")
        
        # Get first page
        first_page = self.get_page(url)
        if not first_page:
            self.logger.error("Failed to fetch first page")
            return []

        # Get total review count
        total_reviews = self.extract_review_count(first_page)
        if total_reviews == 0:
            self.logger.error("Could not determine total number of reviews")
            return []

        self.logger.info(f"Found {total_reviews} total reviews")
        reviews_per_page = 10
        total_pages = (total_reviews + reviews_per_page - 1) // reviews_per_page

        if max_pages:
            total_pages = min(total_pages, max_pages)

        all_reviews = []
        
        # Process each page
        for page_num in range(total_pages):
            current_url = url if page_num == 0 else url.replace('-Reviews-', f'-Reviews-or{page_num*10}-')
            self.logger.info(f"Processing page {page_num + 1}/{total_pages}")
            
            page_content = first_page if page_num == 0 else self.get_page(current_url)
            if not page_content:
                continue

            review_elements = page_content.css('div._c[data-automation="reviewCard"]')
            self.logger.info(f"Found {len(review_elements)} reviews on page {page_num + 1}")

            for review_element in review_elements:
                review_data = self.extract_single_review(review_element)
                if review_data:
                    all_reviews.append(review_data)

            if page_num < total_pages - 1:
                delay = random.uniform(3, 5)
                self.logger.info(f"Waiting {delay:.1f} seconds before next page")
                time.sleep(delay)

        self.logger.info(f"Finished collecting reviews. Total collected: {len(all_reviews)}")
        return all_reviews

    def save_to_csv(self, reviews: List[Dict], filename: str):
        """Save reviews to CSV with statistics."""
        if not reviews:
            self.logger.warning("No reviews to save!")
            return
            
        df = pd.DataFrame(reviews)
        df['scraped_date'] = datetime.now().strftime('%Y-%m-%d')
        
        # Save main CSV file
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        self.logger.info(f"Saved {len(reviews)} reviews to {filename}")
        
        # Save statistics
        stats_filename = filename.replace('.csv', '_stats.txt')
        with open(stats_filename, 'w', encoding='utf-8') as f:
            f.write(f"Statistics for {filename}\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Reviews by Rating:\n")
            f.write(df['rating'].value_counts().sort_index().to_string())
            
            f.write("\n\nReviews by Month:\n")
            df['month'] = pd.to_datetime(df['date'], format='%b %Y', errors='coerce').dt.strftime('%Y-%m')
            f.write(df['month'].value_counts().sort_index().to_string())
            
            f.write("\n\nTrip Types:\n")
            f.write(df['trip_type'].value_counts().to_string())