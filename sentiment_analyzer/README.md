# TripAdvisor Sentiment Analysis System

A dynamic system for extracting and analyzing reviews from TripAdvisor URLs, with support for destination comparison and sentiment analysis. Built for tourism research and destination marketing analysis.

## Features

### 🎯 **URL-to-JSON Extraction**
- Paste any TripAdvisor URL (hotels, restaurants, attractions)
- Automatically extract reviews with metadata
- Anonymize reviewer information (initials only)
- Support for pagination (multiple pages of reviews)

### 📊 **Sentiment Analysis**
- Integrates with existing sentiment analysis framework
- Aspect-based sentiment analysis for accommodations, attractions, restaurants
- Rating correlation analysis
- Temporal sentiment tracking

### 🔍 **Destination Comparison**
- Side-by-side comparison of multiple destinations
- Visual charts and graphs
- Statistical significance testing
- Export results in JSON format

### 🌐 **Web Interface**
- Simple, functional web interface
- No mobile optimization needed (desktop-focused)
- Real-time extraction progress
- Download and view results

## Quick Start

### 1. Installation

```bash
# Clone or navigate to the sentiment_analyzer directory
cd sentiment_analyzer

# Run the setup script
python setup.py
```

### 2. Start the Web Interface

```bash
python web_interface/app.py
```

Then open http://localhost:5000 in your browser.

### 3. Extract Reviews

1. Go to "Extract Reviews" page
2. Paste a TripAdvisor URL (e.g., `https://www.tripadvisor.com/Hotel_Review-g...`)
3. Select number of reviews to extract (25-200)
4. Click "Extract Reviews"
5. Download the JSON results

### 4. Compare Destinations

1. Extract reviews for 2+ destinations first
2. Go to "Compare Destinations" page
3. Select the destinations you want to compare
4. Click "Compare Destinations"
5. View results and download comparison report

## Supported URLs

### ✅ **Hotels & Accommodations**
```
https://www.tripadvisor.com/Hotel_Review-g...
https://www.tripadvisor.com/VacationRentalReview-g...
```

### ✅ **Restaurants & Dining**
```
https://www.tripadvisor.com/Restaurant_Review-g...
```

### ✅ **Attractions & Activities**
```
https://www.tripadvisor.com/Attraction_Review-g...
https://www.tripadvisor.com/ShowUserReviews-g...
```

## Command Line Usage

### Extract Reviews Directly

```python
from extractors.tripadvisor_extractor import extract_reviews_from_url

# Extract reviews from a URL
url = "https://www.tripadvisor.com/Hotel_Review-g..."
data = extract_reviews_from_url(url, max_reviews=100)

print(f"Extracted {len(data['reviews'])} reviews")
print(f"Business: {data['business_info']['name']}")
```

### Compare Destinations Programmatically

```python
from comparison.destination_comparator import compare_destinations_from_files

# Compare multiple JSON files
files = {
    "Destination A": "path/to/destination_a.json",
    "Destination B": "path/to/destination_b.json"
}

results = compare_destinations_from_files(files)
```

## Data Format

### Extracted Review JSON Structure

```json
{
  "extraction_metadata": {
    "url": "https://www.tripadvisor.com/...",
    "extraction_date": "2024-01-15T10:30:00",
    "total_reviews_extracted": 50,
    "extractor_version": "1.0.0"
  },
  "business_info": {
    "name": "Hotel Example",
    "category": "accommodation",
    "location": "Gambia"
  },
  "reviews": [
    {
      "id": "review_123456_1705312200",
      "text": "Great hotel with excellent service...",
      "rating": 5,
      "title": "Excellent stay!",
      "date": "March 2024",
      "reviewer": "J.D.",
      "trip_type": "Family"
    }
  ]
}
```

### Comparison Results Structure

```json
{
  "comparison_metadata": {
    "destinations": ["Hotel A", "Hotel B"],
    "comparison_date": "2024-01-15T10:30:00",
    "total_reviews": 100
  },
  "overall_comparison": {
    "Hotel A": {
      "review_count": 50,
      "avg_rating": 4.2,
      "avg_sentiment": 0.65,
      "positive_percentage": 75.0
    }
  },
  "sentiment_comparison": {...},
  "aspect_comparison": {...}
}
```

## Privacy & Ethics

### 🔒 **Privacy Protection**
- Reviewer names anonymized to initials only
- No personal identifying information stored
- Review dates kept for temporal analysis only

### ⚖️ **Ethical Use**
- Designed for tourism research and destination improvement
- Respects TripAdvisor's content (public reviews only)
- Rate-limited extraction to be respectful to servers
- One-time extraction model (not continuous monitoring)

## Integration with Existing Analysis

The system builds on your existing sentiment analysis framework:

### 🏨 **Accommodation Analysis**
- Integrates with `AccommodationAnalyzer`
- Room features analysis (cleanliness, comfort, amenities)
- Property features analysis (location, service, facilities)
- Trip purpose segmentation

### 🎯 **Attraction Analysis**
- Integrates with `AttractionAnalyzer`
- Activity type analysis
- Experience aspect analysis
- Unique feature identification

### 📈 **Regional Comparison**
- Extends `RegionalComparisonAnalyzer`
- Cross-destination benchmarking
- Competitive analysis
- Market positioning insights

## Technical Requirements

### Software Dependencies
- Python 3.8+
- Chrome/Chromium browser
- ChromeDriver (automatically installed on macOS via Homebrew)

### Python Packages
- Flask (web interface)
- Selenium (web scraping)
- BeautifulSoup4 (HTML parsing)
- Pandas (data processing)
- Matplotlib/Seaborn (visualizations)
- TextBlob/NLTK (sentiment analysis)

## Troubleshooting

### Common Issues

**ChromeDriver not found:**
```bash
# macOS
brew install chromedriver

# Linux
sudo apt-get install chromium-chromedriver

# Windows
# Download from https://chromedriver.chromium.org/
```

**TripAdvisor blocking requests:**
- The system includes random delays and user-agent rotation
- If blocked, wait a few minutes before retrying
- Consider using different network/IP if persistent issues

**Sentiment analysis not working:**
- Ensure NLTK data is downloaded: `python -c "import nltk; nltk.download('punkt')"`
- Check TextBlob installation: `pip install textblob`

### Performance Tips

**Faster Extraction:**
- Use smaller review counts (25-50) for testing
- Run in headless mode (default)
- Extract during off-peak hours

**Better Analysis:**
- Extract more reviews for statistical significance
- Use recent reviews for current sentiment
- Compare similar business types for meaningful insights

## Example Use Cases

### 🏛️ **Tourism Board Analysis**
1. Extract reviews for top accommodations in Gambia
2. Compare with competitor destinations (Senegal, Ghana)
3. Identify strengths and improvement areas
4. Generate insights for marketing strategy

### 🏨 **Hotel Market Research**
1. Extract reviews for luxury hotels in target market
2. Analyze aspect-based sentiment (service, location, amenities)
3. Compare with international competitors
4. Identify service gaps and opportunities

### 🎯 **Destination Marketing**
1. Extract reviews for key attractions
2. Identify most praised unique features
3. Compare sentiment with similar destinations
4. Develop targeted marketing messages

## File Structure

```
sentiment_analyzer/
├── extractors/
│   ├── __init__.py
│   └── tripadvisor_extractor.py      # Main extraction logic
├── comparison/
│   ├── __init__.py
│   └── destination_comparator.py     # Comparison analysis
├── web_interface/
│   ├── app.py                        # Flask web app
│   └── templates/                    # HTML templates
├── results/                          # Extracted data
├── requirements.txt                  # Python dependencies
├── setup.py                         # Setup script
└── README.md                        # This file
```

## Contributing

This system is designed for tourism research. If you enhance it:

1. **Keep privacy-first approach** - maintain anonymization
2. **Respect rate limits** - don't overload TripAdvisor's servers
3. **Document changes** - update this README
4. **Test thoroughly** - ensure extraction accuracy

## License

Research and tourism development use only. Please use responsibly and ethically.

---

## Getting Started with Gambia Example

To test the system with Gambia tourism:

1. Find TripAdvisor URLs for key Gambia destinations:
   - **Kunta Kinteh Island**: Search TripAdvisor for historical attractions
   - **Luxury Accommodations**: Top-rated hotels in Banjul/Kololi
   - **Local Restaurants**: Authentic Gambian dining experiences

2. Extract reviews for each destination type

3. Compare Gambia destinations with:
   - Similar West African destinations
   - International cultural heritage sites
   - Regional accommodation standards

4. Generate insights for:
   - Tourism marketing strategy
   - Service improvement priorities
   - Competitive positioning

This creates a comprehensive sentiment analysis baseline for Gambia's tourism sector. 