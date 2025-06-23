#!/usr/bin/env python3
"""
TripAdvisor Review Extractor

This module provides functionality to extract reviews from TripAdvisor URLs
and convert them to a standardized JSON format for sentiment analysis.
"""

import re
import json
import time
import random
import logging
from datetime import datetime
from typing import List, Dict, Optional
from urllib.parse import urlparse, parse_qs
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TripAdvisorExtractor:
    """
    Extracts reviews from TripAdvisor URLs and converts to standardized format.
    """
    
    def __init__(self, headless=True, max_reviews=100):
        """
        Initialize the extractor.
        
        Args:
            headless (bool): Whether to run browser in headless mode
            max_reviews (int): Maximum number of reviews to extract per URL
        """
        self.headless = headless
        self.max_reviews = max_reviews
        self.driver = None
        
        # TripAdvisor selectors (these may need updating as TA changes their site)
        self.selectors = {
            'reviews': '[data-test-target="HR_CC_CARD"]',
            'review_text': '[data-test-target="review-body"]',
            'rating': '[data-test-target="review-rating"] span',
            'title': '[data-test-target="review-title"]',
            'date': '[data-test-target="review-date"]',
            'reviewer': '[data-test-target="review-username"]',
            'trip_type': '[data-test-target="trip-type"]',
            'next_button': '[aria-label="Next page"]',
            'business_name': 'h1[data-test-target="top-info-header"]',
            'business_type': '[data-test-target="category"]',
            'location': '[data-test-target="location"]'
        }
    
    def setup_driver(self):
        """Set up Chrome driver with appropriate options."""
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
        
        self.driver = webdriver.Chrome(options=chrome_options)
        return self.driver
    
    def close_driver(self):
        """Close the driver if it exists."""
        if self.driver:
            self.driver.quit()
            self.driver = None
    
    def extract_from_url(self, url: str) -> Dict:
        """
        Extract reviews from a TripAdvisor URL.
        
        Args:
            url (str): TripAdvisor URL to extract from
            
        Returns:
            Dict: Structured data with business info and reviews
        """
        logger.info(f"Starting extraction from: {url}")
        
        if not self.driver:
            self.setup_driver()
        
        try:
            # Navigate to the URL
            self.driver.get(url)
            time.sleep(3)  # Wait for page to load
            
            # Extract business information
            business_info = self._extract_business_info()
            
            # Extract reviews
            reviews = self._extract_reviews()
            
            # Structure the output
            result = {
                'extraction_metadata': {
                    'url': url,
                    'extraction_date': datetime.now().isoformat(),
                    'total_reviews_extracted': len(reviews),
                    'extractor_version': '1.0.0'
                },
                'business_info': business_info,
                'reviews': reviews
            }
            
            logger.info(f"Extraction complete. Found {len(reviews)} reviews.")
            return result
            
        except Exception as e:
            logger.error(f"Error during extraction: {str(e)}")
            raise
    
    def _extract_business_info(self) -> Dict:
        """Extract business information from the page."""
        business_info = {}
        
        try:
            # Business name
            name_elem = self.driver.find_element(By.CSS_SELECTOR, self.selectors['business_name'])
            business_info['name'] = name_elem.text.strip()
        except NoSuchElementException:
            business_info['name'] = 'Unknown'
        
        try:
            # Business type/category
            category_elem = self.driver.find_element(By.CSS_SELECTOR, self.selectors['business_type'])
            business_info['category'] = self._categorize_business(category_elem.text.strip())
        except NoSuchElementException:
            business_info['category'] = self._infer_category_from_url()
        
        try:
            # Location
            location_elem = self.driver.find_element(By.CSS_SELECTOR, self.selectors['location'])
            business_info['location'] = location_elem.text.strip()
        except NoSuchElementException:
            business_info['location'] = 'Unknown'
        
        return business_info
    
    def _extract_reviews(self) -> List[Dict]:
        """Extract all reviews from the page(s)."""
        all_reviews = []
        page_count = 0
        
        while len(all_reviews) < self.max_reviews:
            # Extract reviews from current page
            page_reviews = self._extract_page_reviews()
            
            if not page_reviews:
                logger.info("No more reviews found on this page")
                break
            
            all_reviews.extend(page_reviews)
            page_count += 1
            
            logger.info(f"Extracted {len(page_reviews)} reviews from page {page_count}. Total: {len(all_reviews)}")
            
            # Check if we have enough reviews
            if len(all_reviews) >= self.max_reviews:
                all_reviews = all_reviews[:self.max_reviews]
                break
            
            # Try to go to next page
            if not self._go_to_next_page():
                logger.info("No more pages available")
                break
            
            # Random delay between pages
            time.sleep(random.uniform(2, 4))
        
        return all_reviews
    
    def _extract_page_reviews(self) -> List[Dict]:
        """Extract reviews from the current page."""
        reviews = []
        
        # Wait for reviews to load
        try:
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, self.selectors['reviews']))
            )
        except TimeoutException:
            logger.warning("Timeout waiting for reviews to load")
            return reviews
        
        review_elements = self.driver.find_elements(By.CSS_SELECTOR, self.selectors['reviews'])
        logger.info(f"Found {len(review_elements)} review elements on current page")
        
        for elem in review_elements:
            try:
                review_data = self._extract_single_review(elem)
                if review_data:
                    reviews.append(review_data)
            except Exception as e:
                logger.warning(f"Error extracting review: {str(e)}")
                continue
        
        return reviews
    
    def _extract_single_review(self, review_elem) -> Optional[Dict]:
        """Extract data from a single review element."""
        review_data = {}
        
        try:
            # Review text
            text_elem = review_elem.find_element(By.CSS_SELECTOR, self.selectors['review_text'])
            review_data['text'] = text_elem.text.strip()
            
            # Rating
            rating_elem = review_elem.find_element(By.CSS_SELECTOR, self.selectors['rating'])
            rating_class = rating_elem.get_attribute('class')
            rating = self._parse_rating(rating_class)
            review_data['rating'] = rating
            
            # Title
            try:
                title_elem = review_elem.find_element(By.CSS_SELECTOR, self.selectors['title'])
                review_data['title'] = title_elem.text.strip()
            except NoSuchElementException:
                review_data['title'] = ''
            
            # Date
            try:
                date_elem = review_elem.find_element(By.CSS_SELECTOR, self.selectors['date'])
                review_data['date'] = self._parse_date(date_elem.text.strip())
            except NoSuchElementException:
                review_data['date'] = None
            
            # Reviewer (anonymized)
            try:
                reviewer_elem = review_elem.find_element(By.CSS_SELECTOR, self.selectors['reviewer'])
                reviewer_name = reviewer_elem.text.strip()
                review_data['reviewer'] = self._anonymize_reviewer(reviewer_name)
            except NoSuchElementException:
                review_data['reviewer'] = 'Anonymous'
            
            # Trip type
            try:
                trip_elem = review_elem.find_element(By.CSS_SELECTOR, self.selectors['trip_type'])
                review_data['trip_type'] = trip_elem.text.strip()
            except NoSuchElementException:
                review_data['trip_type'] = 'Unknown'
            
            # Add unique ID
            review_data['id'] = f"review_{hash(review_data['text'])}_{int(time.time())}"
            
            return review_data
            
        except Exception as e:
            logger.warning(f"Error extracting review data: {str(e)}")
            return None
    
    def _parse_rating(self, rating_class: str) -> int:
        """Parse rating from CSS class or other indicators."""
        # TripAdvisor often uses classes like 'ui_bubble_rating bubble_50'
        # where 50 means 5.0 stars, 40 means 4.0 stars, etc.
        rating_match = re.search(r'bubble_(\d+)', rating_class)
        if rating_match:
            rating_value = int(rating_match.group(1))
            return rating_value // 10  # Convert 50 -> 5, 40 -> 4, etc.
        
        # Fallback: look for numbers in the class
        numbers = re.findall(r'\d+', rating_class)
        if numbers:
            return min(int(numbers[0]), 5)
        
        return 0  # Default if can't parse
    
    def _parse_date(self, date_text: str) -> str:
        """Parse and standardize date format."""
        # Handle different date formats from TripAdvisor
        date_patterns = [
            r'(\w+) (\d{4})',  # "March 2024"
            r'(\d{1,2})/(\d{1,2})/(\d{4})',  # "3/15/2024"
            r'(\d{1,2})-(\d{1,2})-(\d{4})',  # "3-15-2024"
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, date_text)
            if match:
                try:
                    # This is a simplified date parser
                    # In production, you'd want more robust date parsing
                    return date_text.strip()
                except:
                    continue
        
        return date_text.strip()
    
    def _anonymize_reviewer(self, reviewer_name: str) -> str:
        """Anonymize reviewer name to initials."""
        if not reviewer_name or reviewer_name.lower() in ['anonymous', 'a tripadvisor member']:
            return 'Anonymous'
        
        # Extract initials
        words = reviewer_name.split()
        if len(words) >= 2:
            return f"{words[0][0]}.{words[1][0]}."
        elif len(words) == 1:
            return f"{words[0][0]}."
        else:
            return 'A.'
    
    def _categorize_business(self, category_text: str) -> str:
        """Categorize business based on TripAdvisor category."""
        category_lower = category_text.lower()
        
        if any(word in category_lower for word in ['hotel', 'resort', 'inn', 'lodge', 'accommodation']):
            return 'accommodation'
        elif any(word in category_lower for word in ['restaurant', 'cafe', 'bar', 'dining']):
            return 'restaurant'
        elif any(word in category_lower for word in ['attraction', 'tour', 'activity', 'museum', 'park']):
            return 'attraction'
        else:
            return 'other'
    
    def _infer_category_from_url(self) -> str:
        """Infer business category from URL structure."""
        current_url = self.driver.current_url.lower()
        
        if 'hotel' in current_url or 'accommodation' in current_url:
            return 'accommodation'
        elif 'restaurant' in current_url or 'dining' in current_url:
            return 'restaurant'
        elif 'attraction' in current_url or 'activity' in current_url:
            return 'attraction'
        else:
            return 'other'
    
    def _go_to_next_page(self) -> bool:
        """Navigate to the next page of reviews."""
        try:
            next_button = self.driver.find_element(By.CSS_SELECTOR, self.selectors['next_button'])
            if next_button.is_enabled():
                self.driver.execute_script("arguments[0].click();", next_button)
                time.sleep(3)  # Wait for page to load
                return True
        except NoSuchElementException:
            pass
        
        return False
    
    def save_to_json(self, data: Dict, filename: str = None) -> str:
        """
        Save extracted data to JSON file.
        
        Args:
            data (Dict): Extracted data
            filename (str): Optional custom filename
            
        Returns:
            str: Path to saved file
        """
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            business_name = data.get('business_info', {}).get('name', 'unknown').replace(' ', '_')
            filename = f"tripadvisor_{business_name}_{timestamp}.json"
        
        # Ensure the filename is safe
        filename = re.sub(r'[^\w\-_\.]', '_', filename)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Data saved to: {filename}")
        return filename

# Convenience function for quick extraction
def extract_reviews_from_url(url: str, max_reviews: int = 100, save_file: bool = True) -> Dict:
    """
    Quick function to extract reviews from a TripAdvisor URL.
    
    Args:
        url (str): TripAdvisor URL
        max_reviews (int): Maximum number of reviews to extract
        save_file (bool): Whether to save results to JSON file
        
    Returns:
        Dict: Extracted review data
    """
    extractor = TripAdvisorExtractor(max_reviews=max_reviews)
    
    try:
        data = extractor.extract_from_url(url)
        
        if save_file:
            extractor.save_to_json(data)
        
        return data
        
    finally:
        extractor.close_driver()

if __name__ == "__main__":
    # Example usage
    url = input("Enter TripAdvisor URL: ").strip()
    max_reviews = int(input("Max reviews to extract (default 50): ") or 50)
    
    try:
        data = extract_reviews_from_url(url, max_reviews)
        print(f"\nExtraction complete!")
        print(f"Business: {data['business_info']['name']}")
        print(f"Category: {data['business_info']['category']}")
        print(f"Reviews extracted: {len(data['reviews'])}")
        
    except Exception as e:
        print(f"Error: {str(e)}") 