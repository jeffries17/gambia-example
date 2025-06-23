import requests
from bs4 import BeautifulSoup
import time
import random
import re
import os
import json
from fake_useragent import UserAgent
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tripadvisor_scraping.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_user_agent():
    """Generate a random user agent or use a fallback if fake_useragent fails"""
    try:
        ua = UserAgent()
        return ua.random
    except Exception as e:
        logger.warning(f"Failed to generate random user agent: {e}")
        # Fallback user agents (modern browsers)
        fallback_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36 Edg/95.0.1020.44',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:94.0) Gecko/20100101 Firefox/94.0',
            'Mozilla/5.0 (iPhone; CPU iPhone OS 15_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1'
        ]
        return random.choice(fallback_agents)

def get_request_headers():
    """Get headers that better mimic a real browser"""
    user_agent = get_user_agent()
    headers = {
        'User-Agent': user_agent,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
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
    return headers

def make_request(url, retry_count=5, session=None):
    """Make a request with advanced error handling and anti-blocking measures"""
    if session is None:
        session = requests.Session()
    
    # Implement exponential backoff with jitter
    for attempt in range(retry_count):
        try:
            # Random delay between 2-5 seconds with jitter
            delay = random.uniform(2, 5) + (attempt * 2)
            logger.info(f"Waiting {delay:.1f} seconds before request...")
            time.sleep(delay)
            
            # Get fresh headers for each request
            headers = get_request_headers()
            
            # Make the request
            logger.info(f"Sending request to {url}")
            response = session.get(url, headers=headers, timeout=30)
            
            # Handle status codes
            if response.status_code == 200:
                logger.info(f"Successfully fetched page: status code {response.status_code}")
                return response.text
            elif response.status_code == 403:
                logger.warning(f"Access forbidden (403) - attempt {attempt+1}/{retry_count}")
                # Extra long delay after a 403
                time.sleep(10 + random.uniform(5, 15))
            elif response.status_code == 429:
                logger.warning(f"Rate limited (429) - attempt {attempt+1}/{retry_count}")
                # Very long delay after a rate limit
                time.sleep(30 + random.uniform(15, 45))
            else:
                logger.warning(f"HTTP error {response.status_code} - attempt {attempt+1}/{retry_count}")
                
        except requests.RequestException as e:
            logger.warning(f"Request failed on attempt {attempt+1}: {str(e)}")
        
        # Calculate backoff time with jitter
        backoff_time = (2 ** attempt) + random.uniform(1, 5)
        logger.info(f"Backing off for {backoff_time:.1f} seconds...")
        time.sleep(backoff_time)
    
    logger.error(f"Failed to fetch {url} after {retry_count} attempts")
    return None

def extract_destination_code(url):
    """Extract the destination code from a TripAdvisor URL"""
    match = re.search(r'[-/](g\d+)[-/]', url)
    if match:
        return match.group(1)
    return None

def extract_links(html, pattern):
    """Extract links from the HTML"""
    if not html:
        return []
        
    soup = BeautifulSoup(html, 'html.parser')
    base_url = "https://www.tripadvisor.com"
    links = []
    
    # Try multiple selectors to find links
    selectors = [
        f'a[href*="{pattern}"]',  # Generic pattern match
        '.result_wrap a',  # Results wrapper links
        '.result a',  # Simple result links
        '.listItem a',  # List item links
        '.property_title',  # Property title links (hotels)
        '.BMQDV._F',  # Current TripAdvisor link format
        'a.review_count',  # Review count links
        '.result_title a',  # Result title links
    ]
    
    for selector in selectors:
        elements = soup.select(selector)
        for element in elements:
            href = element.get('href')
            if href and pattern in href:
                # Ensure absolute URL
                full_url = f"{base_url}{href}" if href.startswith('/') else href
                links.append(full_url)
    
    # Remove duplicates while preserving order
    unique_links = []
    seen = set()
    for link in links:
        if link not in seen:
            seen.add(link)
            unique_links.append(link)
    
    return unique_links

def get_pagination_info(html):
    """Extract total count and pagination info using multiple approaches"""
    if not html:
        return None
        
    soup = BeautifulSoup(html, 'html.parser')
    
    # Approach 1: Look for the showing results text
    patterns = [
        r'Showing results .*?of (\d+(?:,\d+)*)',
        r'(\d+(?:,\d+)*) matching properties',
        r'(\d+(?:,\d+)*) results found',
        r'(\d+(?:,\d+)*) properties'
    ]
    
    for pattern in patterns:
        regex = re.compile(pattern)
        text_elements = soup.find_all(text=regex)
        
        for element in text_elements:
            match = regex.search(element)
            if match:
                count = int(match.group(1).replace(',', ''))
                logger.info(f"Found total count: {count}")
                return count
    
    # Approach 2: Check pagination elements
    pagination_selectors = [
        '.pageNumbers a',  # Standard pagination
        '.pagination a',   # Alternative pagination
        '.unified a',      # Unified pagination
        '.page_num a',     # Legacy pagination
    ]
    
    for selector in pagination_selectors:
        elements = soup.select(selector)
        if elements:
            try:
                # Get the text from the last pagination element
                last_page_text = elements[-1].get_text().strip()
                # Extract digits from the text
                last_page_match = re.search(r'\d+', last_page_text)
                if last_page_match:
                    last_page = int(last_page_match.group(0))
                    items_per_page = 30  # Standard for TripAdvisor
                    estimated_count = last_page * items_per_page
                    logger.info(f"Estimated count from pagination: {estimated_count}")
                    return estimated_count
            except Exception as e:
                logger.warning(f"Error extracting pagination info: {e}")
    
    # Approach 3: Count items on the page and assume standard pagination
    result_selectors = [
        '.listItem',  # Standard list items
        '.result',    # Results
        '.content',   # Content elements
        '.property_listing', # Property listings
    ]
    
    for selector in result_selectors:
        elements = soup.select(selector)
        if elements:
            items_on_page = len(elements)
            if items_on_page > 0:
                # Assume 10 pages as conservative estimate
                estimated_count = items_on_page * 10
                logger.info(f"Estimated count from page items: {estimated_count}")
                return estimated_count
    
    # If all else fails, return a conservative estimate
    logger.warning("Could not determine count - using conservative estimate")
    return 300

def fetch_all_links(start_url, link_pattern, session=None):
    """Fetch all links by following pagination with improved robustness"""
    if session is None:
        session = requests.Session()
        
    all_links = []
    current_url = start_url
    
    # Try to extract destination code for proper pagination
    destination_code = extract_destination_code(start_url)
    if not destination_code:
        logger.error(f"Could not extract destination code from {start_url}")
        return []
    
    # First page
    logger.info(f"Fetching initial page: {current_url}")
    html = make_request(current_url, session=session)
    if not html:
        logger.error("Failed to fetch initial page - aborting")
        return []
    
    # Save HTML for debugging
    with open(f"debug_{destination_code}_page1.html", "w", encoding="utf-8") as f:
        f.write(html)
    
    # Get initial links
    links = extract_links(html, link_pattern)
    all_links.extend(links)
    logger.info(f"Page 1: Found {len(links)} links")
    
    # If no links found on first page, likely blocked - abort
    if not links:
        logger.error("No links found on first page - likely blocked or wrong selectors")
        return []
    
    # Get total count for pagination
    total_count = get_pagination_info(html)
    if not total_count:
        logger.warning("Could not determine total count, using conservative estimate")
        total_pages = 15  # Conservative estimate
    else:
        # Calculate total pages (items per page is typically 30)
        total_pages = (total_count + 29) // 30  # Ceiling division
        logger.info(f"Found approximately {total_count} total items ({total_pages} pages)")
    
    # Process remaining pages
    page_size = 30
    for page_num in range(1, min(total_pages, 50)):  # Cap at 50 pages for safety
        offset = page_num * page_size
        
        # Construct pagination URL based on URL format
        if 'oa0-' in current_url:
            next_url = current_url.replace('oa0-', f'oa{offset}-')
        elif '-oa' in current_url:
            # Replace existing offset
            next_url = re.sub(r'-oa\d+-', f'-oa{offset}-', current_url)
        else:
            # Standard format without existing pagination
            if '.html' in current_url:
                if '-Activities' in current_url:
                    next_url = current_url.replace('-Activities', f'-Activities-oa{offset}')
                else:
                    next_url = current_url.replace('.html', f'-oa{offset}.html')
            else:
                next_url = f"{current_url}-oa{offset}"
                
        logger.info(f"Fetching page {page_num+1}: {next_url}")
        html = make_request(next_url, session=session)
        
        # Save HTML for debugging
        if html:
            with open(f"debug_{destination_code}_page{page_num+1}.html", "w", encoding="utf-8") as f:
                f.write(html)
        
        if not html:
            logger.warning(f"Failed to fetch page {page_num+1}, stopping pagination")
            break
            
        links = extract_links(html, link_pattern)
        if not links:
            logger.warning(f"No links found on page {page_num+1}, reached end of results")
            break
            
        all_links.extend(links)
        logger.info(f"Page {page_num+1}: Found {len(links)} links")
        
        # If we've collected a significant number of links, we can stop
        # This helps with partial success even if we hit blocking later
        if len(all_links) >= total_count * 0.8:
            logger.info(f"Collected {len(all_links)} links (80%+ of estimated total) - stopping pagination")
            break
    
    # Remove duplicates
    unique_links = list(dict.fromkeys(all_links))
    logger.info(f"Total unique links: {len(unique_links)}")
    return unique_links

def save_links(links, filename):
    """Save links to a text file"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        for link in links:
            f.write(f"{link}\n")
    logger.info(f"Saved {len(links)} links to {filename}")

def main():
    """Main function with improved error handling and session management"""
    # Create a persistent session for better performance and cookie handling
    session = requests.Session()
    
    # URLs for Gambia - using different variations to improve chances of success
    urls = {
        'hotels': [
            'https://www.tripadvisor.com/Hotels-g293794-Gambia-Hotels.html',
            'https://www.tripadvisor.com/Hotels-g293794.html',
            'https://www.tripadvisor.com/Search?q=Gambia&searchSessionId=&searchNearby=false&sid=&blockRedirect=true&geo=293794&geoId=293794',
        ],
        'restaurants': [
            'https://www.tripadvisor.com/Restaurants-g293794-Gambia.html',
            'https://www.tripadvisor.com/Restaurants-g293794.html',
            'https://www.tripadvisor.com/Search?q=restaurants%20gambia&searchSessionId=&searchNearby=false',
        ],
        'attractions': [
            'https://www.tripadvisor.com/Attractions-g293794-Activities-Gambia.html',
            'https://www.tripadvisor.com/Attractions-g293794-Activities.html',
            'https://www.tripadvisor.com/Search?q=attractions%20gambia&searchSessionId=&searchNearby=false',
        ]
    }
    
    # Link patterns to look for
    patterns = {
        'hotels': 'Hotel_Review',
        'restaurants': 'Restaurant_Review',
        'attractions': 'Attraction_Review'
    }
    
    # Create output directory with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f'tripadvisor_links_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    # Try each type with multiple URL variations
    for category, url_variations in urls.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"EXTRACTING {category.upper()} LINKS")
        logger.info(f"{'='*50}")
        
        links = []
        for i, url in enumerate(url_variations):
            logger.info(f"Trying URL variation {i+1}/{len(url_variations)}: {url}")
            
            try:
                variation_links = fetch_all_links(url, patterns[category], session=session)
                if variation_links:
                    logger.info(f"Success! Found {len(variation_links)} links with URL variation {i+1}")
                    links.extend(variation_links)
                    # If we found links, no need to try other variations
                    break
                else:
                    logger.warning(f"No links found with URL variation {i+1}")
            except Exception as e:
                logger.error(f"Error processing URL variation {i+1}: {str(e)}")
        
        # Remove duplicates
        unique_links = list(dict.fromkeys(links))
        results[category] = unique_links
        
        # Save category links
        save_links(unique_links, f"{output_dir}/{category}_links.txt")
        
        # Add a longer delay between categories
        delay = random.uniform(10, 20)
        logger.info(f"Waiting {delay:.1f} seconds before processing next category...")
        time.sleep(delay)
    
    # Save summary
    with open(f"{output_dir}/summary.json", 'w', encoding='utf-8') as f:
        json.dump({
            'total_links': sum(len(links) for links in results.values()),
            'counts': {category: len(links) for category, links in results.items()},
            'timestamp': timestamp
        }, f, indent=2)
        
    logger.info("\nSUMMARY:")
    logger.info(f"Total links extracted: {sum(len(links) for links in results.values())}")
    for category, links in results.items():
        logger.info(f"- {category.capitalize()}: {len(links)} links")
    logger.info(f"\nAll links saved to {output_dir}/ directory")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}", exc_info=True)