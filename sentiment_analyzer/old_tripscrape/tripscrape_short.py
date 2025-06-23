import requests
from parsel import Selector
import pandas as pd
import time
from typing import List, Dict
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

class TripScape:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Origin': 'https://www.tripadvisor.com',
            'Referer': 'https://www.tripadvisor.com'
        }
    
    def get_page(self, url: str) -> Selector:
        """Fetch a single page and return its Selector object"""
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return Selector(response.text)
        except requests.RequestException as e:
            print(f"Error fetching page: {e}")
            return None
        
    def get_language_counts(self, selector) -> Dict[str, int]:
        """Get available languages and their review counts from the language tabs"""
        language_counts = {}
        
        # Try multiple possible selectors for language buttons
        selectors = [
            'button.Datwj',
            'button[data-automation="language-filter"]',
            'button[data-filtertype="language"]',
            'div[data-test-target="language-filter"] button',
            'button[aria-label*="reviews"]'  # More generic fallback
        ]
        
        for selector_str in selectors:
            language_buttons = selector.css(selector_str)
            if language_buttons:
                print(f"Found language buttons using selector: {selector_str}")
                break
        
        # Also try to find language info in any spans or divs that might contain it
        if not language_buttons:
            text_elements = selector.css('span::text, div::text').getall()
            for text in text_elements:
                if '(' in text and ')' in text:
                    for lang in [('english', 'en'), ('german', 'de'), ('dutch', 'nl')]:
                        if lang[0] in text.lower():
                            try:
                                count = int(text.split('(')[1].split(')')[0].strip())
                                language_counts[lang[1]] = count
                            except ValueError:
                                continue
        
        for button in language_buttons:
            # Try multiple attributes that might contain language info
            for attr in ['aria-label', 'title', 'data-val']:
                label = button.attrib.get(attr, '')
                if '(' in label and ')' in label:
                    lang_part = label.split('(')[0].strip().lower()
                    count_part = label.split('(')[1].rstrip(')').strip()
                    try:
                        if 'english' in lang_part:
                            language_counts['en'] = int(count_part)
                        elif 'german' in lang_part or 'deutsch' in lang_part:
                            language_counts['de'] = int(count_part)
                        elif 'dutch' in lang_part or 'nederlands' in lang_part:
                            language_counts['nl'] = int(count_part)
                    except ValueError:
                        continue
                    
            # Also check button text content
            button_text = ' '.join(button.css('::text').getall()).lower()
            if '(' in button_text and ')' in button_text:
                for lang, code in [('english', 'en'), ('german', 'de'), ('dutch', 'nl'),
                                 ('deutsch', 'de'), ('nederlands', 'nl')]:
                    if lang in button_text:
                        try:
                            count = int(button_text.split('(')[1].split(')')[0].strip())
                            language_counts[code] = count
                        except ValueError:
                            continue
                    
        print(f"Found language counts: {language_counts}")
        return language_counts

    def extract_single_review(self, review) -> Dict:
        """Extract data from a single review element"""
        try:
            review_data = {}
            
            reviewer = review.css('a.BMQDV ::text').get()
            review_data['reviewer'] = reviewer.strip() if reviewer else ''
            
            location = review.css('div.biGQs._P.pZUbB.osNWb span::text').get()
            review_data['location'] = location.strip() if location else ''
            
            title = review.css('div.fiohW.qWPrE.ncFvv.fOtGX a ::text').get()
            review_data['title'] = title.strip() if title else ''
            
            text = review.css('div.biGQs._P.pZUbB.KxBGd span.JguWG ::text').get()
            review_data['text'] = text.strip() if text else ''
            
            rating = review.css('svg.UctUV title ::text').get()
            if rating:
                try:
                    rating_value = float(rating.split()[0].replace(',', '.'))
                    review_data['rating'] = rating_value
                except:
                    review_data['rating'] = None
            else:
                review_data['rating'] = None
            
            date = review.css('div.RpeCd ::text').get()
            review_data['date'] = date.strip() if date else ''
            
            response = review.css('div.biGQs._P.pZUbB.KxBGd span.JguWG::text').getall()
            review_data['property_response'] = response[1].strip() if len(response) > 1 else ''
            
            return review_data
            
        except Exception as e:
            print(f"Error extracting review: {e}")
            return None

    def get_reviews_by_language(self, url: str, language: str, expected_count: int) -> List[Dict]:
        """Get all reviews for a specific language tab"""
        reviews = []
        page = 0
        reviews_per_page = 10  # TripAdvisor's standard pagination size
        
        while len(reviews) < expected_count:
            # Construct URL with language filter and pagination
            parsed_url = urlparse(url)
            query_params = {'filterLang': language}
            
            # Add pagination if needed
            if page > 0:
                query_params['or'] = str(page * reviews_per_page)
            
            # Handle case where URL already has parameters
            if parsed_url.query:
                existing_params = dict(parse_qsl(parsed_url.query))
                query_params.update(existing_params)
            
            # Reconstruct URL with all parameters
            current_url = urlunparse(
                parsed_url._replace(query=urlencode(query_params))
            )
                
            print(f"Fetching {language} reviews page {page + 1} from: {current_url}")
            selector = self.get_page(current_url)
            
            if not selector:
                break
                
            review_elements = selector.css('div._c[data-automation="reviewCard"]')
            if not review_elements:
                print(f"No more reviews found for {language}")
                break
                
            page_reviews = []
            for review_element in review_elements:
                review_data = self.extract_single_review(review_element)
                if review_data:
                    review_data['original_language'] = language
                    page_reviews.append(review_data)
            
            if not page_reviews:
                print(f"No more reviews found for {language}")
                break
                
            reviews.extend(page_reviews)
            print(f"Retrieved {len(reviews)}/{expected_count} {language} reviews")
            
            if len(page_reviews) < reviews_per_page:
                print(f"Last page reached for {language} (incomplete page)")
                break
                
            page += 1
            time.sleep(2)  # Respectful delay
            
        return reviews

    def get_all_reviews(self, url: str, languages: List[str] = None) -> List[Dict]:
        """Get all reviews in specified languages using language tabs"""
        selector = self.get_page(url)
        if not selector:
            return []
            
        available_languages = self.get_language_counts(selector)
        languages = languages or ['en']
        all_reviews = []
        
        for lang in languages:
            if lang in available_languages:
                expected_count = available_languages[lang]
                print(f"\nFetching {lang} reviews (Expected count: {expected_count})")
                lang_reviews = self.get_reviews_by_language(url, lang, expected_count)
                all_reviews.extend(lang_reviews)
                print(f"Retrieved {len(lang_reviews)} {lang} reviews")
                
        return all_reviews

    def save_to_csv(self, reviews: List[Dict], filename: str = 'reviews.csv'):
        """Save reviews to CSV file"""
        if not reviews:
            print("No reviews to save")
            return
            
        df = pd.DataFrame(reviews)
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"Saved {len(reviews)} reviews to {filename}")

# Example usage
if __name__ == "__main__":
    scraper = TripScape()
    test_url = "https://www.tripadvisor.com/Attraction_Review-g298044-d14977416-Reviews-Kikooko_Africa_Safaris-Entebbe_Central_Region.html"
    
    languages = ['en', 'de', 'nl']
    reviews = scraper.get_all_reviews(test_url, languages=languages)
    scraper.save_to_csv(reviews, 'reviews_by_language_tab_2.csv')