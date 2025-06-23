import requests
from bs4 import BeautifulSoup
import csv
import time
import re
import os

def get_club_info(url):
    """Scrape club information from a single page"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all club entries - based on the structure we see in the screenshot
        clubs = []
        # Each club is presented in a row with 4 TD cells
        club_entries = soup.select('table tr')
        
        for entry in club_entries:
            # Skip header rows
            if entry.find('th'):
                continue
                
            club_data = {}
            
            # Get club name and URL from the second column
            name_link = entry.select_one('td:nth-of-type(2) a')
            if name_link:
                club_data['name'] = name_link.get_text(strip=True)
                club_data['idpa_url'] = name_link.get('href', '')
            
            # Get location from paragraphs in second column
            location_elements = entry.select('td:nth-of-type(2) p')
            if len(location_elements) >= 1:
                club_data['location'] = location_elements[0].get_text(strip=True)
            
            # Get phone if available (usually in second paragraph of second column)
            if len(location_elements) >= 2:
                phone = location_elements[1].get_text(strip=True)
                if re.search(r'\d', phone):  # Check if it contains digits (likely a phone number)
                    club_data['phone'] = phone
            
            # Get external website from third column - this is what we're looking for
            website_cell = entry.select_one('td:nth-of-type(3)')
            if website_cell and website_cell.select_one('a'):
                website_link = website_cell.select_one('a')
                club_data['website'] = website_link.get('href', '')
                
                # Sometimes there's text in addition to the link
                website_text = website_cell.get_text(strip=True)
                if website_text and website_text != club_data['website']:
                    club_data['website_text'] = website_text
            
            if club_data.get('name'):  # Only add if we have at least a name
                clubs.append(club_data)
        
        return clubs
    except requests.exceptions.RequestException as e:
        print(f"Request error scraping {url}: {e}")
        return []
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return []

def scrape_all_club_pages():
    """Scrape all club pages"""
    base_url = "https://www.idpa.com/clubs/page/{}/?search-location&search-radius&search-word&search-id&search_country&search_state"
    page = 1
    max_pages = 50  # Set a reasonable upper limit
    all_clubs = []
    
    while page <= max_pages:
        url = base_url.format(page)
        print(f"Scraping page {page}...")
        
        # Retry logic for getting club info
        retry_count = 0
        max_retries = 3
        clubs = []
        
        while retry_count < max_retries:
            clubs = get_club_info(url)
            if clubs:
                break
            
            retry_count += 1
            if retry_count < max_retries:
                print(f"Retrying page {page} (attempt {retry_count+1}/{max_retries})...")
                time.sleep(5)  # Wait longer between retries
        
        if not clubs:
            # If no clubs were found after retries, we've probably reached the end
            print("No clubs found on this page after multiple attempts, stopping.")
            break
            
        all_clubs.extend(clubs)
        print(f"Found {len(clubs)} clubs on page {page}")
        
        # Check if there's a next page by looking at the pagination
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for text indicating total number of clubs/pages
            results_text = soup.select_one('h2, .woocommerce-result-count')
            if results_text:
                match = re.search(r'Showing \(.*? of (\d+)\)', results_text.text)
                if match:
                    total_clubs = int(match.group(1))
                    clubs_per_page = 10  # Typically 10 clubs per page
                    estimated_pages = (total_clubs + clubs_per_page - 1) // clubs_per_page
                    max_pages = min(max_pages, estimated_pages)
                    print(f"Detected approximately {max_pages} total pages")
            
            # Look for next page link
            next_page = False
            pagination = soup.select('a[href*="/clubs/page/"]')
            for link in pagination:
                if str(page + 1) in link.get('href', ''):
                    next_page = True
                    break
            
            if not next_page:
                print("No next page link found, stopping.")
                break
                
        except Exception as e:
            print(f"Error checking pagination: {e}")
        
        page += 1
        time.sleep(2)  # Be nice to the server
    
    return all_clubs

def save_to_csv(clubs, filename="idpa_clubs.csv"):
    """Save club information to CSV file"""
    if not clubs:
        print("No clubs to save.")
        return
        
    fieldnames = ['name', 'location', 'phone', 'website', 'website_text', 'idpa_url']
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for club in clubs:
            writer.writerow(club)
    
    print(f"Saved {len(clubs)} clubs to {filename}")

def main():
    """Main function"""
    print("Starting IDPA club scraper...")
    
    clubs = scrape_all_club_pages()
    print(f"Found {len(clubs)} clubs in total")
    
    save_to_csv(clubs)
    print("Scraping complete!")

if __name__ == "__main__":
    main() 