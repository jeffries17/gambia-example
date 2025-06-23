import requests
from bs4 import BeautifulSoup
import time
import random
import re
import os
import json

def get_random_user_agent():
    """Return a random modern user agent"""
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36 Edg/96.0.1054.29',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:94.0) Gecko/20100101 Firefox/94.0',
        'Mozilla/5.0 (iPhone; CPU iPhone OS 15_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1',
        'Mozilla/5.0 (iPad; CPU OS 15_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/96.0.4664.53 Mobile/15E148 Safari/604.1',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36',
        'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:94.0) Gecko/20100101 Firefox/94.0',
    ]
    return random.choice(user_agents)

def make_request(url, session=None):
    """Make a request with anti-blocking measures"""
    # Use provided session or create a new one
    if session is None:
        session = requests.Session()
    
    # Random delay to avoid rate limiting (4-8 seconds)
    delay = random.uniform(4, 8)
    print(f"Waiting {delay:.1f} seconds before request...")
    time.sleep(delay)
    
    # Create headers that look like a real browser
    headers = {
        'User-Agent': get_random_user_agent(),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'max-age=0',
        'Referer': 'https://www.tripadvisor.com/',
        'DNT': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'same-origin',
        'Sec-Fetch-User': '?1'
    }
    
    # Try up to 3 times
    for attempt in range(3):
        try:
            print(f"Attempt {attempt+1}: Requesting {url}")
            response = session.get(url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                print(f"Success! Status code: {response.status_code}")
                return response.text
            else:
                print(f"Got status code {response.status_code} on attempt {attempt+1}")
                if attempt < 2:
                    # Wait longer between retries (increasing backoff)
                    retry_delay = 10 + (10 * attempt) + random.uniform(1, 5)
                    print(f"Waiting {retry_delay:.1f} seconds before retry...")
                    time.sleep(retry_delay)
                    # Rotate user agent for retry
                    headers['User-Agent'] = get_random_user_agent()
        
        except Exception as e:
            print(f"Error on attempt {attempt+1}: {str(e)}")
            if attempt < 2:
                # Wait between retries
                time.sleep(5 + (5 * attempt))
    
    print(f"Failed to fetch {url} after 3 attempts")
    return None

def extract_links(html, pattern):
    """Extract links from HTML content"""
    if not html:
        return []
        
    print(f"Extracting links matching pattern: {pattern}")
    soup = BeautifulSoup(html, 'html.parser')
    base_url = "https://www.tripadvisor.com"
    all_links = []
    
    # Try to find links using various selectors
    try:
        # Find all links that contain the pattern
        for link in soup.find_all('a', href=True):
            href = link.get('href', '')
            if pattern in href:
                full_url = href if href.startswith('http') else f"{base_url}{href}"
                all_links.append(full_url)
    except Exception as e:
        print(f"Error extracting links: {e}")
    
    # Remove duplicates
    unique_links = list(dict.fromkeys(all_links))
    print(f"Found {len(unique_links)} unique links")
    return unique_links

def main():
    print("=== Simple TripAdvisor Link Extractor ===")
    
    # Create a session to maintain cookies
    session = requests.Session()
    
    # Output directory
    output_dir = "tripadvisor_links"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get Gambia destination code
    destination_code = 'g293794'
    
    # Mapping of category to URL and pattern
    categories = {
        'hotels': ('https://www.tripadvisor.com/Hotels-g293794-Gambia-Hotels.html', 'Hotel_Review'),
        'restaurants': ('https://www.tripadvisor.com/Restaurants-g293794-Gambia.html', 'Restaurant_Review'),
        'attractions': ('https://www.tripadvisor.com/Attractions-g293794-Activities-Gambia.html', 'Attraction_Review')
    }
    
    results = {}
    
    for category, (url, pattern) in categories.items():
        print(f"\n{'='*50}")
        print(f"EXTRACTING {category.upper()} LINKS")
        print(f"{'='*50}")
        
        # Get the first page
        html = make_request(url, session)
        
        # Save the HTML for debugging
        if html:
            with open(f"{output_dir}/{category}_page1.html", 'w', encoding='utf-8') as f:
                f.write(html)
            print(f"Saved first page HTML to {output_dir}/{category}_page1.html")
            
            # Extract links from first page
            links = extract_links(html, pattern)
            
            # Try to extract pagination info to estimate total pages
            soup = BeautifulSoup(html, 'html.parser')
            max_pages = 10  # Default conservative estimate
            
            # Look for pagination elements
            pagination = soup.select('.pageNumbers a')
            if pagination and len(pagination) > 0:
                try:
                    last_page_text = pagination[-1].get_text().strip()
                    if last_page_text.isdigit():
                        max_pages = min(int(last_page_text), 20)  # Cap at 20 pages
                        print(f"Found pagination: {max_pages} pages")
                except Exception as e:
                    print(f"Error parsing pagination: {e}")
            
            # Process additional pages
            for page in range(1, max_pages):
                offset = page * 30
                
                # Construct pagination URL
                if 'oa0-' in url:
                    next_url = url.replace('oa0-', f'oa{offset}-')
                elif '-oa' in url:
                    next_url = re.sub(r'-oa\d+-', f'-oa{offset}-', url)
                elif '-Activities' in url:
                    next_url = url.replace('-Activities', f'-Activities-oa{offset}')
                elif '.html' in url:
                    next_url = url.replace('.html', f'-oa{offset}.html')
                else:
                    next_url = f"{url}-oa{offset}"
                
                print(f"\nProcessing page {page+1}: {next_url}")
                page_html = make_request(next_url, session)
                
                if page_html:
                    # Save this page's HTML for debugging
                    with open(f"{output_dir}/{category}_page{page+1}.html", 'w', encoding='utf-8') as f:
                        f.write(page_html)
                    
                    # Extract links from this page
                    page_links = extract_links(page_html, pattern)
                    links.extend(page_links)
                    
                    if not page_links:
                        print(f"No links found on page {page+1}. Stopping pagination.")
                        break
                else:
                    print(f"Failed to fetch page {page+1}. Stopping pagination.")
                    break
                
                # Add a delay between pages
                time.sleep(random.uniform(8, 12))
            
            # Remove duplicates
            unique_links = list(dict.fromkeys(links))
            results[category] = unique_links
            
            # Save links to file
            links_file = f"{output_dir}/{category}_links.txt"
            with open(links_file, 'w', encoding='utf-8') as f:
                for link in unique_links:
                    f.write(f"{link}\n")
            print(f"Saved {len(unique_links)} links to {links_file}")
        else:
            print(f"Failed to fetch the first page for {category}")
            results[category] = []
        
        # Add a longer delay between categories
        time.sleep(random.uniform(15, 25))
    
    # Save summary
    summary = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'total_links': sum(len(links) for links in results.values()),
        'counts': {category: len(links) for category, links in results.items()}
    }
    
    with open(f"{output_dir}/summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print("\nSUMMARY:")
    print(f"Total links extracted: {summary['total_links']}")
    for category, count in summary['counts'].items():
        print(f"- {category.capitalize()}: {count} links")
    print(f"\nAll results saved to {output_dir}/ directory")


if __name__ == "__main__":
    main()