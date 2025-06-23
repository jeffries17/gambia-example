#!/usr/bin/env python3
"""
Improved TripAdvisor Review Extractor

This module combines the best practices from the old tripscrape implementation
with modern techniques for more reliable and efficient review extraction.
"""

import re
import json
import time
import random
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlparse, parse_qs
import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent


class ImprovedTripAdvisorExtractor:
    """
    Enhanced TripAdvisor extractor using requests + BeautifulSoup with robust anti-detection.
    Combines best practices from old_tripscrape with modern improvements.
    """
    
    def __init__(self, max_reviews: int = 500, max_pages: int = None):
        """
        Initialize the extractor.
        
        Args:
            max_reviews (int): Maximum number of reviews to extract
            max_pages (int): Maximum number of pages to scrape (None = unlimited)
        """
        self.max_reviews = max_reviews
        self.max_pages = max_pages
        self.session = requests.Session()
        self.seen_reviews = set()
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize user agent
        try:
            self.ua = UserAgent()
        except Exception:
            self.ua = None
            self.logger.warning("Failed to initialize UserAgent, using fallback")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('tripadvisor_extraction.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def _get_user_agent(self) -> str:
        """Generate a random user agent or use fallback."""
        try:
            if self.ua:
                return self.ua.random
        except Exception:
            pass
        
        # Fallback to modern user agents
        fallback_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1'
        ]
        return random.choice(fallback_agents)
    
    def _get_headers(self) -> Dict[str, str]:
        """Get realistic browser headers."""
        return {
            'User-Agent': self._get_user_agent(),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'same-origin',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
            'Referer': 'https://www.tripadvisor.com/'
        }
    
    def _make_request(self, url: str, retry_count: int = 3) -> Optional[str]:
        """Make request with robust error handling and anti-blocking measures."""
        for attempt in range(retry_count):
            try:
                # Random delay with jitter
                delay = random.uniform(2, 5) + (attempt * 2)
                self.logger.info(f"Waiting {delay:.1f} seconds before request...")
                time.sleep(delay)
                
                # Fresh headers for each request
                headers = self._get_headers()
                
                self.logger.info(f"Sending request to {url}")
                response = self.session.get(url, headers=headers, timeout=30)
                
                if response.status_code == 200:
                    self.logger.info(f"Successfully fetched page: {response.status_code}")
                    return response.text
                elif response.status_code == 403:
                    self.logger.warning(f"Access forbidden (403) - attempt {attempt+1}/{retry_count}")
                    time.sleep(10 + random.uniform(5, 15))
                elif response.status_code == 429:
                    self.logger.warning(f"Rate limited (429) - attempt {attempt+1}/{retry_count}")
                    time.sleep(30 + random.uniform(15, 45))
                else:
                    self.logger.warning(f"HTTP error {response.status_code} - attempt {attempt+1}/{retry_count}")
                    
            except requests.RequestException as e:
                self.logger.warning(f"Request failed on attempt {attempt+1}: {str(e)}")
            
            # Exponential backoff with jitter
            backoff_time = (2 ** attempt) + random.uniform(1, 5)
            self.logger.info(f"Backing off for {backoff_time:.1f} seconds...")
            time.sleep(backoff_time)
        
        self.logger.error(f"Failed to fetch {url} after {retry_count} attempts")
        return None
    
    def extract_from_url(self, url: str) -> Dict:
        """
        Extract reviews from a TripAdvisor URL.
        
        Args:
            url (str): TripAdvisor URL to extract from
            
        Returns:
            Dict: Structured data with business info and reviews
        """
        self.logger.info(f"Starting extraction from: {url}")
        
        # Get first page
        html = self._make_request(url)
        if not html:
            raise Exception(f"Failed to fetch initial page: {url}")
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract business information
        business_info = self._extract_business_info(soup, url)
        
        # Extract total review count and calculate pages
        total_reviews = self._extract_review_count(soup)
        reviews_per_page = 10  # TripAdvisor standard
        max_pages_from_count = (total_reviews + reviews_per_page - 1) // reviews_per_page
        
        # Limit pages if specified
        if self.max_pages:
            max_pages_from_count = min(max_pages_from_count, self.max_pages)
        
        self.logger.info(f"Found {total_reviews} total reviews across {max_pages_from_count} pages")
        
        # Extract reviews from all pages
        all_reviews = []
        
        for page_num in range(max_pages_from_count):
            if len(all_reviews) >= self.max_reviews:
                break
                
            page_url = self._construct_page_url(url, page_num)
            
            if page_num == 0:
                page_soup = soup  # Use already fetched first page
            else:
                page_html = self._make_request(page_url)
                if not page_html:
                    continue
                page_soup = BeautifulSoup(page_html, 'html.parser')
            
            page_reviews = self._extract_page_reviews(page_soup)
            
            if not page_reviews:
                self.logger.info(f"No reviews found on page {page_num + 1}, stopping")
                break
            
            all_reviews.extend(page_reviews)
            self.logger.info(f"Page {page_num + 1}: extracted {len(page_reviews)} reviews (total: {len(all_reviews)})")
            
            # Random delay between pages
            if page_num < max_pages_from_count - 1:
                time.sleep(random.uniform(3, 6))
        
        # Limit to max_reviews
        if len(all_reviews) > self.max_reviews:
            all_reviews = all_reviews[:self.max_reviews]
        
        # Structure the output
        result = {
            'extraction_metadata': {
                'url': url,
                'extraction_date': datetime.now().isoformat(),
                'total_reviews_found': total_reviews,
                'total_reviews_extracted': len(all_reviews),
                'pages_processed': min(page_num + 1, max_pages_from_count),
                'extractor_version': '2.0.0'
            },
            'business_info': business_info,
            'reviews': all_reviews
        }
        
        self.logger.info(f"Extraction complete. Extracted {len(all_reviews)} reviews.")
        return result
    
    def _extract_business_info(self, soup: BeautifulSoup, url: str) -> Dict:
        """Extract business information from the page."""
        business_info = {
            'name': 'Unknown',
            'category': 'unknown',
            'location': 'Unknown',
            'url': url
        }
        
        # Multiple selectors for business name
        name_selectors = [
            'h1[data-test-target="top-info-header"]',
            'h1.biGQs._P.fiohW.fOtGX',
            'h1.QdLfr.bgMZj._a',
            'h1'
        ]
        
        for selector in name_selectors:
            elem = soup.select_one(selector)
            if elem and elem.get_text().strip():
                business_info['name'] = elem.get_text().strip()
                break
        
        # Infer category from URL
        business_info['category'] = self._infer_category_from_url(url)
        
        # Extract location - multiple approaches
        location_selectors = [
            '[data-test-target="location"]',
            '.fHvkI.PTrfg',
            '.biGQs._P.pZUbB.osNWb'
        ]
        
        for selector in location_selectors:
            elem = soup.select_one(selector)
            if elem and elem.get_text().strip():
                business_info['location'] = elem.get_text().strip()
                break
        
        return business_info
    
    def _extract_review_count(self, soup: BeautifulSoup) -> int:
        """Extract total number of reviews."""
        # Multiple patterns to find review count
        patterns = [
            r'(\d+(?:,\d+)*)\s+reviews',
            r'(\d+(?:,\d+)*)\s+Reviews',
            r'Showing.*?of\s+(\d+(?:,\d+)*)',
            r'(\d+(?:,\d+)*)\s+matching'
        ]
        
        text_content = soup.get_text()
        
        for pattern in patterns:
            matches = re.findall(pattern, text_content)
            if matches:
                try:
                    count = int(matches[0].replace(',', ''))
                    self.logger.info(f"Found review count: {count}")
                    return count
                except ValueError:
                    continue
        
        # Fallback: count review elements on first page and estimate
        review_elements = self._find_review_elements(soup)
        if review_elements:
            # Estimate based on pagination
            pagination = soup.select('.pageNumbers a, .pagination a')
            if pagination:
                try:
                    last_page_text = pagination[-1].get_text().strip()
                    if last_page_text.isdigit():
                        estimated = int(last_page_text) * 10  # 10 reviews per page
                        self.logger.info(f"Estimated review count from pagination: {estimated}")
                        return estimated
                except:
                    pass
            
            # Very rough estimate if no pagination
            estimated = len(review_elements) * 10
            self.logger.info(f"Rough estimated review count: {estimated}")
            return estimated
        
        self.logger.warning("Could not determine review count")
        return 0
    
    def _find_review_elements(self, soup: BeautifulSoup) -> List:
        """Find review elements using multiple selectors."""
        selectors = [
            '[data-test-target="HR_CC_CARD"]',
            'div._c[data-automation="reviewCard"]',
            '.review-container',
            '.prw_rup_resp_vertical_review',
            'div[data-reviewid]'
        ]
        
        for selector in selectors:
            elements = soup.select(selector)
            if elements:
                self.logger.debug(f"Found {len(elements)} reviews using selector: {selector}")
                return elements
        
        return []
    
    def _extract_page_reviews(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract reviews from current page."""
        reviews = []
        review_elements = self._find_review_elements(soup)
        
        for element in review_elements:
            review_data = self._extract_single_review(element)
            if review_data and review_data.get('review_id') not in self.seen_reviews:
                reviews.append(review_data)
                self.seen_reviews.add(review_data.get('review_id', ''))
        
        return reviews
    
    def _extract_single_review(self, element) -> Optional[Dict]:
        """Extract data from a single review element."""
        try:
            # Generate unique review ID
            review_id = f"review_{hash(str(element))}"
            
            # Extract text - multiple selectors
            text_selectors = [
                '[data-test-target="review-body"]',
                '.fIrGe._T .KgQgP',
                'span.yCeTE',
                '.partial_entry'
            ]
            
            text = ''
            for selector in text_selectors:
                elem = element.select_one(selector)
                if elem:
                    text = elem.get_text().strip()
                    break
            
            # Extract title
            title_selectors = [
                '[data-test-target="review-title"]',
                '.noQuotes',
                '.quote'
            ]
            
            title = ''
            for selector in title_selectors:
                elem = element.select_one(selector)
                if elem:
                    title = elem.get_text().strip()
                    break
            
            # Extract rating
            rating = self._extract_rating(element)
            
            # Extract date
            date = self._extract_date(element)
            
            # Extract reviewer info
            reviewer_info = self._extract_reviewer_info(element)
            
            # Extract trip type
            trip_type = self._extract_trip_type(element)
            
            if not text and not title:
                return None  # Skip if no meaningful content
            
            return {
                'review_id': review_id,
                'title': title,
                'text': text,
                'rating': rating,
                'date': date,
                'trip_type': trip_type,
                **reviewer_info
            }
            
        except Exception as e:
            self.logger.warning(f"Error extracting review: {str(e)}")
            return None
    
    def _extract_rating(self, element) -> Optional[float]:
        """Extract rating from review element."""
        # Multiple approaches for rating extraction
        rating_selectors = [
            '[data-test-target="review-rating"]',
            '.ui_star_rating',
            'svg title'
        ]
        
        for selector in rating_selectors:
            elem = element.select_one(selector)
            if elem:
                # Check for title attribute with rating
                title = elem.get('title', '')
                if title:
                    match = re.search(r'(\d+(?:\.\d+)?)', title)
                    if match:
                        return float(match.group(1))
                
                # Check for class-based rating
                class_attr = elem.get('class', [])
                for cls in class_attr:
                    if 'bubble_' in cls:
                        match = re.search(r'bubble_(\d+)', cls)
                        if match:
                            return float(match.group(1)) / 10  # TripAdvisor uses 50 for 5 stars
        
        return None
    
    def _extract_date(self, element) -> str:
        """Extract review date."""
        date_selectors = [
            '[data-test-target="review-date"]',
            '.ratingDate',
            '.prw_rup_meta_hsx_review_resp_additional_info_author'
        ]
        
        for selector in date_selectors:
            elem = element.select_one(selector)
            if elem:
                date_text = elem.get_text().strip()
                # Clean up date text
                date_text = re.sub(r'Date of visit:|Visited|Review collected in partnership with.*', '', date_text).strip()
                return date_text
        
        return ''
    
    def _extract_reviewer_info(self, element) -> Dict:
        """Extract reviewer information."""
        reviewer_selectors = [
            '[data-test-target="review-username"]',
            '.info_text .username',
            '.memberOverlayLink'
        ]
        
        username = 'Anonymous'
        for selector in reviewer_selectors:
            elem = element.select_one(selector)
            if elem:
                username = elem.get_text().strip()
                break
        
        # Extract location if available
        location_selectors = [
            '.default_text',
            '.userLoc'
        ]
        
        location = ''
        for selector in location_selectors:
            elem = element.select_one(selector)
            if elem:
                location = elem.get_text().strip()
                break
        
        return {
            'reviewer_username': self._anonymize_username(username),
            'reviewer_location': location
        }
    
    def _extract_trip_type(self, element) -> str:
        """Extract trip type information."""
        trip_selectors = [
            '[data-test-target="trip-type"]',
            '.recommend-titleInline'
        ]
        
        for selector in trip_selectors:
            elem = element.select_one(selector)
            if elem:
                return elem.get_text().strip()
        
        return ''
    
    def _anonymize_username(self, username: str) -> str:
        """Anonymize username for privacy."""
        if not username or username.lower() in ['anonymous', 'unknown']:
            return 'Anonymous'
        
        # Keep first letter and length, replace rest with asterisks
        if len(username) <= 2:
            return '*' * len(username)
        
        return username[0] + '*' * (len(username) - 2) + username[-1]
    
    def _construct_page_url(self, base_url: str, page_num: int) -> str:
        """Construct URL for specific page number."""
        if page_num == 0:
            return base_url
        
        offset = page_num * 10
        
        # Different URL patterns for different types
        if '-Reviews-' in base_url:
            if '-or' in base_url:
                return re.sub(r'-or\d+-', f'-or{offset}-', base_url)
            else:
                return base_url.replace('-Reviews-', f'-Reviews-or{offset}-')
        elif '.html' in base_url:
            return base_url.replace('.html', f'-or{offset}.html')
        else:
            return f"{base_url}-or{offset}"
    
    def _infer_category_from_url(self, url: str) -> str:
        """Infer business category from URL pattern."""
        if 'Hotel_Review' in url or 'Hotels-' in url:
            return 'accommodation'
        elif 'Restaurant_Review' in url or 'Restaurants-' in url:
            return 'restaurant'
        elif 'Attraction_Review' in url or 'Attractions-' in url:
            return 'attraction'
        else:
            return 'unknown'
    
    def save_to_json(self, data: Dict, filename: str = None) -> str:
        """Save extracted data to JSON file."""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            business_name = data.get('business_info', {}).get('name', 'unknown')
            safe_name = re.sub(r'[^\w\s-]', '', business_name).strip().replace(' ', '_')
            filename = f"{safe_name}_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Data saved to {filename}")
        return filename


def extract_reviews_from_url(url: str, max_reviews: int = 500, max_pages: int = None, save_file: bool = True) -> Dict:
    """
    Convenience function to extract reviews from a TripAdvisor URL.
    
    Args:
        url (str): TripAdvisor URL
        max_reviews (int): Maximum reviews to extract
        max_pages (int): Maximum pages to process
        save_file (bool): Whether to save results to file
    
    Returns:
        Dict: Extracted review data
    """
    extractor = ImprovedTripAdvisorExtractor(max_reviews=max_reviews, max_pages=max_pages)
    
    try:
        result = extractor.extract_from_url(url)
        
        if save_file:
            extractor.save_to_json(result)
        
        return result
        
    except Exception as e:
        logging.error(f"Extraction failed: {str(e)}")
        raise


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python improved_tripadvisor_extractor.py <tripadvisor_url>")
        sys.exit(1)
    
    url = sys.argv[1]
    result = extract_reviews_from_url(url)
    print(f"Extracted {len(result['reviews'])} reviews from {result['business_info']['name']}") 