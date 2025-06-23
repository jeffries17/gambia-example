#!/usr/bin/env python3
"""
Apify-Inspired TripAdvisor Review Extractor

This extractor is inspired by Apify's sophisticated approach and data structure,
incorporating their best practices while remaining open-source and customizable.
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


class ApifyInspiredExtractor:
    """
    TripAdvisor extractor inspired by Apify's data structure and quality.
    Focuses on matching their rich data extraction while being open-source.
    """
    
    def __init__(self, max_reviews: int = 500, enable_user_details: bool = True):
        """
        Initialize the extractor.
        
        Args:
            max_reviews (int): Maximum number of reviews to extract
            enable_user_details (bool): Whether to extract detailed user information
        """
        self.max_reviews = max_reviews
        self.enable_user_details = enable_user_details
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
                logging.FileHandler('apify_inspired_extraction.log'),
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
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0'
        ]
        return random.choice(fallback_agents)
    
    def _get_headers(self) -> Dict[str, str]:
        """Get realistic browser headers."""
        return {
            'User-Agent': self._get_user_agent(),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
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
        """Make request with robust error handling."""
        for attempt in range(retry_count):
            try:
                # Random delay with jitter (Apify-style)
                delay = random.uniform(3, 7) + (attempt * 2)
                self.logger.info(f"Waiting {delay:.1f} seconds before request...")
                time.sleep(delay)
                
                headers = self._get_headers()
                self.logger.info(f"Sending request to {url}")
                response = self.session.get(url, headers=headers, timeout=30)
                
                if response.status_code == 200:
                    self.logger.info(f"Successfully fetched page: {response.status_code}")
                    return response.text
                elif response.status_code == 403:
                    self.logger.warning(f"Access forbidden (403) - attempt {attempt+1}/{retry_count}")
                    time.sleep(15 + random.uniform(5, 20))
                elif response.status_code == 429:
                    self.logger.warning(f"Rate limited (429) - attempt {attempt+1}/{retry_count}")
                    time.sleep(45 + random.uniform(15, 30))
                else:
                    self.logger.warning(f"HTTP error {response.status_code}")
                    
            except requests.RequestException as e:
                self.logger.warning(f"Request failed on attempt {attempt+1}: {str(e)}")
            
            # Exponential backoff
            backoff_time = (2 ** attempt) + random.uniform(2, 8)
            time.sleep(backoff_time)
        
        self.logger.error(f"Failed to fetch {url} after {retry_count} attempts")
        return None
    
    def extract_from_url(self, url: str) -> List[Dict]:
        """
        Extract reviews from a TripAdvisor URL in Apify's format.
        
        Args:
            url (str): TripAdvisor URL to extract from
            
        Returns:
            List[Dict]: Reviews in Apify-compatible format
        """
        self.logger.info(f"Starting extraction from: {url}")
        
        # Get first page
        html = self._make_request(url)
        if not html:
            raise Exception(f"Failed to fetch initial page: {url}")
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract place information (Apify-style)
        place_info = self._extract_place_info(soup, url)
        
        # Calculate total pages needed
        total_reviews = place_info.get('numberOfReviews', 0)
        reviews_per_page = 10
        max_pages = min((total_reviews + reviews_per_page - 1) // reviews_per_page, 50)  # Cap at 50 pages
        
        if total_reviews > self.max_reviews:
            max_pages = min(max_pages, (self.max_reviews + reviews_per_page - 1) // reviews_per_page)
        
        self.logger.info(f"Processing {max_pages} pages for {total_reviews} total reviews")
        
        # Extract reviews from all pages
        all_reviews = []
        
        for page_num in range(max_pages):
            if len(all_reviews) >= self.max_reviews:
                break
            
            page_url = self._construct_page_url(url, page_num)
            
            if page_num == 0:
                page_soup = soup
            else:
                page_html = self._make_request(page_url)
                if not page_html:
                    continue
                page_soup = BeautifulSoup(page_html, 'html.parser')
            
            page_reviews = self._extract_page_reviews(page_soup, place_info)
            
            if not page_reviews:
                self.logger.info(f"No reviews found on page {page_num + 1}, stopping")
                break
            
            all_reviews.extend(page_reviews)
            self.logger.info(f"Page {page_num + 1}: extracted {len(page_reviews)} reviews")
            
            # Delay between pages
            if page_num < max_pages - 1:
                time.sleep(random.uniform(4, 8))
        
        # Limit to max_reviews
        if len(all_reviews) > self.max_reviews:
            all_reviews = all_reviews[:self.max_reviews]
        
        self.logger.info(f"Extraction complete. Extracted {len(all_reviews)} reviews.")
        return all_reviews
    
    def _extract_place_info(self, soup: BeautifulSoup, url: str) -> Dict:
        """Extract place information in Apify format."""
        place_info = {
            'id': self._extract_place_id(url),
            'name': 'Unknown',
            'rating': None,
            'numberOfReviews': 0,
            'locationString': 'Unknown',
            'latitude': None,
            'longitude': None,
            'webUrl': url,
            'website': None,
            'address': None,
            'addressObj': {
                'street1': None,
                'street2': None,
                'city': None,
                'state': None,
                'country': None,
                'postalcode': None
            },
            'ratingHistogram': {
                'count1': 0,
                'count2': 0,
                'count3': 0,
                'count4': 0,
                'count5': 0
            }
        }
        
        # Extract business name
        name_selectors = [
            'h1[data-test-target="top-info-header"]',
            'h1.biGQs._P.fiohW.fOtGX',
            'h1.QdLfr.bgMZj._a',
            'h1'
        ]
        
        for selector in name_selectors:
            elem = soup.select_one(selector)
            if elem and elem.get_text().strip():
                place_info['name'] = elem.get_text().strip()
                break
        
        # Extract overall rating
        rating_selectors = [
            '[data-test-target="review-rating-badge"]',
            '.reviewsBlackText .overallRating',
            '.ui_star_rating'
        ]
        
        for selector in rating_selectors:
            elem = soup.select_one(selector)
            if elem:
                rating_text = elem.get_text().strip()
                match = re.search(r'(\d+(?:\.\d+)?)', rating_text)
                if match:
                    place_info['rating'] = float(match.group(1))
                    break
        
        # Extract review count
        review_count_patterns = [
            r'(\d+(?:,\d+)*)\s+reviews',
            r'(\d+(?:,\d+)*)\s+Reviews'
        ]
        
        text_content = soup.get_text()
        for pattern in review_count_patterns:
            matches = re.findall(pattern, text_content)
            if matches:
                try:
                    place_info['numberOfReviews'] = int(matches[0].replace(',', ''))
                    break
                except ValueError:
                    continue
        
        # Extract location string
        location_selectors = [
            '[data-test-target="location"]',
            '.fHvkI.PTrfg',
            '.biGQs._P.pZUbB.osNWb'
        ]
        
        for selector in location_selectors:
            elem = soup.select_one(selector)
            if elem and elem.get_text().strip():
                place_info['locationString'] = elem.get_text().strip()
                break
        
        # Extract rating histogram (if available)
        histogram = self._extract_rating_histogram(soup)
        if histogram:
            place_info['ratingHistogram'] = histogram
        
        return place_info
    
    def _extract_place_id(self, url: str) -> str:
        """Extract place ID from URL."""
        match = re.search(r'-d(\d+)-', url)
        return match.group(1) if match else 'unknown'
    
    def _extract_rating_histogram(self, soup: BeautifulSoup) -> Optional[Dict]:
        """Extract rating distribution histogram."""
        histogram = {'count1': 0, 'count2': 0, 'count3': 0, 'count4': 0, 'count5': 0}
        
        # Look for rating distribution elements
        rating_elements = soup.select('.rating_histogram .row')
        
        for element in rating_elements:
            rating_text = element.get_text()
            # Extract rating level and count
            rating_match = re.search(r'(\d+)\s+star.*?(\d+)', rating_text)
            if rating_match:
                rating_level = int(rating_match.group(1))
                count = int(rating_match.group(2))
                histogram[f'count{rating_level}'] = count
        
        # Return None if no histogram data found
        if all(count == 0 for count in histogram.values()):
            return None
        
        return histogram
    
    def _extract_page_reviews(self, soup: BeautifulSoup, place_info: Dict) -> List[Dict]:
        """Extract reviews from current page in Apify format."""
        reviews = []
        review_elements = self._find_review_elements(soup)
        
        for element in review_elements:
            review_data = self._extract_single_review(element, place_info)
            if review_data:
                review_id = review_data.get('url', '').split('-r')[-1].split('-')[0]
                if review_id not in self.seen_reviews:
                    reviews.append(review_data)
                    self.seen_reviews.add(review_id)
        
        return reviews
    
    def _find_review_elements(self, soup: BeautifulSoup) -> List:
        """Find review elements using multiple selectors."""
        selectors = [
            '[data-test-target="HR_CC_CARD"]',
            'div._c[data-automation="reviewCard"]',
            '.review-container',
            'div[data-reviewid]'
        ]
        
        for selector in selectors:
            elements = soup.select(selector)
            if elements:
                return elements
        
        return []
    
    def _extract_single_review(self, element, place_info: Dict) -> Optional[Dict]:
        """Extract single review in Apify format."""
        try:
            # Basic review data
            title = self._extract_text_by_selectors(element, [
                '[data-test-target="review-title"]',
                '.noQuotes',
                '.quote'
            ])
            
            text = self._extract_text_by_selectors(element, [
                '[data-test-target="review-body"]',
                '.fIrGe._T .KgQgP',
                'span.yCeTE',
                '.partial_entry'
            ])
            
            if not text and not title:
                return None
            
            rating = self._extract_rating_from_element(element)
            dates = self._extract_dates(element)
            user_info = self._extract_user_info(element) if self.enable_user_details else None
            review_url = self._extract_review_url(element, place_info.get('webUrl', ''))
            owner_response = self._extract_owner_response(element)
            
            # Build review object in Apify format
            review = {
                'title': title,
                'rating': rating,
                'travelDate': dates.get('travel_date', ''),
                'publishedDate': dates.get('published_date', ''),
                'text': text,
                'url': review_url,
                'user': user_info,
                'ownerResponse': owner_response,
                'placeInfo': place_info
            }
            
            return review
            
        except Exception as e:
            self.logger.warning(f"Error extracting review: {str(e)}")
            return None
    
    def _extract_text_by_selectors(self, element, selectors: List[str]) -> str:
        """Extract text using multiple selector fallbacks."""
        for selector in selectors:
            elem = element.select_one(selector)
            if elem:
                return elem.get_text().strip()
        return ''
    
    def _extract_rating_from_element(self, element) -> Optional[int]:
        """Extract rating from review element."""
        selectors = [
            '[data-test-target="review-rating"]',
            '.ui_star_rating',
            'svg title'
        ]
        
        for selector in selectors:
            elem = element.select_one(selector)
            if elem:
                # Check title attribute
                title = elem.get('title', '')
                if title:
                    match = re.search(r'(\d+)', title)
                    if match:
                        return int(match.group(1))
                
                # Check class-based rating
                class_attr = elem.get('class', [])
                for cls in class_attr:
                    if 'bubble_' in cls:
                        match = re.search(r'bubble_(\d+)', cls)
                        if match:
                            return int(int(match.group(1)) / 10)  # Convert 50 to 5
        
        return None
    
    def _extract_dates(self, element) -> Dict[str, str]:
        """Extract travel and published dates."""
        dates = {'travel_date': '', 'published_date': ''}
        
        date_selectors = [
            '[data-test-target="review-date"]',
            '.ratingDate',
            '.prw_rup_meta_hsx_review_resp_additional_info_author'
        ]
        
        for selector in date_selectors:
            elem = element.select_one(selector)
            if elem:
                date_text = elem.get_text().strip()
                
                # Try to extract both travel and published dates
                if 'Date of visit' in date_text:
                    dates['travel_date'] = date_text.replace('Date of visit:', '').strip()
                elif 'Visited' in date_text:
                    dates['travel_date'] = date_text.replace('Visited', '').strip()
                else:
                    dates['published_date'] = date_text
                
                break
        
        return dates
    
    def _extract_user_info(self, element) -> Optional[Dict]:
        """Extract user information in Apify format."""
        if not self.enable_user_details:
            return None
        
        user_info = {
            'userId': None,
            'name': None,
            'contributions': {
                'totalContributions': 0,
                'helpfulVotes': 0
            },
            'username': None,
            'userLocation': None,
            'avatar': None,
            'link': None
        }
        
        # Extract username
        username_selectors = [
            '[data-test-target="review-username"]',
            '.info_text .username',
            '.memberOverlayLink'
        ]
        
        for selector in username_selectors:
            elem = element.select_one(selector)
            if elem:
                username = elem.get_text().strip()
                user_info['username'] = username
                user_info['name'] = username
                break
        
        # Extract user location
        location_selectors = [
            '.default_text',
            '.userLoc'
        ]
        
        for selector in location_selectors:
            elem = element.select_one(selector)
            if elem:
                location_text = elem.get_text().strip()
                user_info['userLocation'] = {
                    'name': location_text,
                    'shortName': location_text.split(',')[0] if ',' in location_text else location_text
                }
                break
        
        # Extract contribution count (if available)
        contrib_text = element.get_text()
        contrib_match = re.search(r'(\d+)\s+contributions?', contrib_text)
        if contrib_match:
            user_info['contributions']['totalContributions'] = int(contrib_match.group(1))
        
        return user_info if user_info['username'] else None
    
    def _extract_review_url(self, element, base_url: str) -> str:
        """Extract or construct review URL."""
        # Look for review ID in data attributes
        review_id = element.get('data-reviewid')
        if review_id and base_url:
            return f"{base_url}#review{review_id}"
        
        # Try to find a direct link
        link_elem = element.select_one('a[href*="ShowUserReviews"]')
        if link_elem:
            href = link_elem.get('href')
            if href.startswith('http'):
                return href
            else:
                return f"https://www.tripadvisor.com{href}"
        
        return base_url
    
    def _extract_owner_response(self, element) -> Optional[Dict]:
        """Extract owner response if available."""
        response_selectors = [
            '.owner_response',
            '.mgrRspnInline'
        ]
        
        for selector in response_selectors:
            elem = element.select_one(selector)
            if elem:
                response_text = elem.get_text().strip()
                if response_text:
                    return {
                        'text': response_text,
                        'date': '',  # Would need additional parsing
                        'title': 'Response from management'
                    }
        
        return None
    
    def _construct_page_url(self, base_url: str, page_num: int) -> str:
        """Construct URL for specific page number."""
        if page_num == 0:
            return base_url
        
        offset = page_num * 10
        
        if '-Reviews-' in base_url:
            if '-or' in base_url:
                return re.sub(r'-or\d+-', f'-or{offset}-', base_url)
            else:
                return base_url.replace('-Reviews-', f'-Reviews-or{offset}-')
        elif '.html' in base_url:
            return base_url.replace('.html', f'-or{offset}.html')
        else:
            return f"{base_url}-or{offset}"
    
    def save_to_json(self, reviews: List[Dict], filename: str = None) -> str:
        """Save reviews to JSON file in Apify format."""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            if reviews:
                place_name = reviews[0].get('placeInfo', {}).get('name', 'unknown')
                safe_name = re.sub(r'[^\w\s-]', '', place_name).strip().replace(' ', '_')
                filename = f"{safe_name}_reviews_{timestamp}.json"
            else:
                filename = f"reviews_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(reviews, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved {len(reviews)} reviews to {filename}")
        return filename


def extract_reviews_apify_style(url: str, max_reviews: int = 500, save_file: bool = True) -> List[Dict]:
    """
    Extract reviews in Apify's format.
    
    Args:
        url (str): TripAdvisor URL
        max_reviews (int): Maximum reviews to extract
        save_file (bool): Whether to save results to file
    
    Returns:
        List[Dict]: Reviews in Apify format
    """
    extractor = ApifyInspiredExtractor(max_reviews=max_reviews)
    
    try:
        reviews = extractor.extract_from_url(url)
        
        if save_file and reviews:
            extractor.save_to_json(reviews)
        
        return reviews
        
    except Exception as e:
        logging.error(f"Extraction failed: {str(e)}")
        raise


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python apify_inspired_extractor.py <tripadvisor_url>")
        sys.exit(1)
    
    url = sys.argv[1]
    reviews = extract_reviews_apify_style(url)
    print(f"Extracted {len(reviews)} reviews in Apify format") 