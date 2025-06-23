import requests
from bs4 import BeautifulSoup
import csv
import time
import re
import os
import urllib.parse
import sys

def extract_emails_from_page(url):
    """Extract email addresses from a web page"""
    emails = []
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        # Common email pattern
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        
        # Find emails in the HTML
        found_emails = re.findall(email_pattern, response.text)
        
        # Clean up and deduplicate emails
        for email in found_emails:
            # Make sure it's a valid email and not part of a script or code
            if (
                '@' in email 
                and '.' in email 
                and not email.endswith('.png') 
                and not email.endswith('.jpg') 
                and not email.endswith('.js')
                and not email.endswith('.css')
                and not 'example' in email.lower()
                and not email.lower() == 'name@email.com'
                and not email.lower() == 'your@email.com'
                and not email == 'user@domain.com'
                and not email == 'email@domain.com'
            ):
                emails.append(email.lower())
        
        return list(set(emails))  # Return unique emails
    except Exception as e:
        print(f"Error extracting emails from {url}: {e}")
        return []

def check_contact_pages(base_url):
    """Check common contact pages for emails"""
    all_emails = []
    try:
        # Parse the URL to get the base domain
        parsed_url = urllib.parse.urlparse(base_url)
        domain_base = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        # Common contact page paths to try
        contact_paths = [
            '/contact', 
            '/contact-us', 
            '/about', 
            '/about-us', 
            '/contact.html', 
            '/contact-us.html', 
            '/about.html', 
            '/about-us.html',
            '/get-in-touch',
            '/reach-us',
            '/staff',
            '/team',
            '/club-info',
            '/membership'
        ]
        
        # Try the main URL first
        print(f"Checking main URL: {base_url}")
        main_emails = extract_emails_from_page(base_url)
        if main_emails:
            all_emails.extend(main_emails)
            print(f"  Found {len(main_emails)} emails")
        
        # If no emails found on the main page and it's a Facebook page, skip further attempts
        if not all_emails and 'facebook.com' in base_url.lower():
            print("  Facebook page - skipping further checks")
            return all_emails
        
        # If no emails found on main page, try common contact pages
        if not all_emails:
            for path in contact_paths:
                contact_url = domain_base + path
                print(f"  Checking {contact_url}")
                try:
                    contact_emails = extract_emails_from_page(contact_url)
                    if contact_emails:
                        all_emails.extend(contact_emails)
                        print(f"    Found {len(contact_emails)} emails")
                        break  # Found emails, no need to check more pages
                except Exception as e:
                    print(f"    Error: {e}")
                time.sleep(1)  # Brief pause between requests
        
        return list(set(all_emails))  # Return unique emails
        
    except Exception as e:
        print(f"Error checking contact pages: {e}")
        return all_emails

def process_club_websites(input_csv="idpa_clubs.csv", output_csv="idpa_clubs_with_emails.csv"):
    """Process each club's website to find contact emails"""
    # Load the clubs data
    clubs = []
    try:
        with open(input_csv, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            clubs = list(reader)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return
    
    # Count clubs with websites
    websites_count = sum(1 for club in clubs if club.get('website'))
    print(f"Found {len(clubs)} clubs, {websites_count} with websites")
    
    # Process each club with a website
    for i, club in enumerate(clubs):
        if club.get('website'):
            print(f"\nProcessing club {i+1}/{len(clubs)}: {club.get('name', 'Unknown')}")
            try:
                # Get emails from the club's website
                emails = check_contact_pages(club['website'])
                
                if emails:
                    club['emails'] = ', '.join(emails)
                    print(f"Found emails: {club['emails']}")
                else:
                    club['emails'] = ""
                    print("No emails found")
            except Exception as e:
                print(f"Error processing {club['website']}: {e}")
                club['emails'] = ""
            
            # Save progress after each club
            save_to_csv(clubs, output_csv)
            
            # Be nice to the servers
            time.sleep(2)
        else:
            # Skip clubs without websites
            club['emails'] = ""
    
    return clubs

def save_to_csv(clubs, filename):
    """Save club information to CSV file"""
    if not clubs:
        print("No clubs to save.")
        return
        
    fieldnames = clubs[0].keys()  # Use the keys from the first club as fieldnames
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for club in clubs:
            writer.writerow(club)

def main():
    """Main function"""
    print("Starting IDPA club email scraper...")
    
    # Check for command line arguments
    start_index = 0
    if len(sys.argv) > 1:
        try:
            start_index = int(sys.argv[1])
            print(f"Starting from club #{start_index}")
        except ValueError:
            print(f"Invalid start index: {sys.argv[1]}")
            return
    
    process_club_websites()
    print("\nEmail scraping complete!")

if __name__ == "__main__":
    main() 