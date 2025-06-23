import pandas as pd
from typing import Dict, Optional
import json
import os

class LocationMapper:
    def __init__(self, mappings_file='location_mappings.json'):
        self.mappings_file = mappings_file
        self.load_mappings()
    
    def load_mappings(self):
        """Load existing mappings from JSON file"""
        if os.path.exists(self.mappings_file):
            with open(self.mappings_file, 'r') as f:
                mappings = json.load(f)
                self.city_to_country = mappings.get('cities', {})
                self.country_aliases = mappings.get('countries', {})
        else:
            # Initialize with extensive mappings
            self.city_to_country = {
                # Switzerland cities
                'zurich': 'Switzerland',
                'bern': 'Switzerland',
                'kirchdorf': 'Switzerland',
                'oberlunkhofen': 'Switzerland',
                'lucerne': 'Switzerland',
                'hettlingen': 'Switzerland',
                'schwanden': 'Switzerland',
                'trimmis': 'Switzerland',
                'lausanne': 'Switzerland',
                'basel': 'Switzerland',
                'geneva': 'Switzerland',
                
                # UK cities
                'london': 'United Kingdom',
                'harrogate': 'United Kingdom',
                'norwich': 'United Kingdom',
                'fleet': 'United Kingdom',
                'manchester': 'United Kingdom',
                'birmingham': 'United Kingdom',
                'edinburgh': 'United Kingdom',
                'glasgow': 'United Kingdom',
                
                # Netherlands cities
                'amsterdam': 'Netherlands',
                'rotterdam': 'Netherlands',
                'den haag': 'Netherlands',
                'zoetermeer': 'Netherlands',
                'utrecht': 'Netherlands',
                'eindhoven': 'Netherlands',
                
                # Cyprus cities
                'prodromi': 'Cyprus',
                'limassol': 'Cyprus',
                'nicosia': 'Cyprus',
                'larnaca': 'Cyprus',
                
                # Australian cities
                'sydney': 'Australia',
                'melbourne': 'Australia',
                'brisbane': 'Australia',
                'perth': 'Australia',
                
                # Austrian cities
                'vienna': 'Austria',
                'salzburg': 'Austria',
                'graz': 'Austria'
            }
            
            self.country_aliases = {
                # United Kingdom variations
                'uk': 'United Kingdom',
                'u.k.': 'United Kingdom',
                'united kingdom': 'United Kingdom',
                'great britain': 'United Kingdom',
                'gb': 'United Kingdom',
                'britain': 'United Kingdom',
                'england': 'United Kingdom',
                'scotland': 'United Kingdom',
                'wales': 'United Kingdom',
                
                # Switzerland variations
                'schweiz': 'Switzerland',
                'suisse': 'Switzerland',
                'svizzera': 'Switzerland',
                'ch': 'Switzerland',
                'switzerland': 'Switzerland',  # Added this
                'Switzerland': 'Switzerland',  # Added this
                
                # Netherlands variations
                'nl': 'Netherlands',
                'nederland': 'Netherlands',
                'the netherlands': 'Netherlands',
                'holland': 'Netherlands',
                'netherlands': 'Netherlands',  # Added this
                
                # USA variations
                'usa': 'United States',
                'us': 'United States',
                'united states': 'United States',
                'united states of america': 'United States',
                'america': 'United States',
                
                # Australia variations
                'aus': 'Australia',
                'australian': 'Australia',
                'australia': 'Australia',  # Added this
                
                # Austria variations
                'österreich': 'Austria',
                'osterreich': 'Austria',
                'at': 'Austria',
                'austria': 'Austria'  # Added this
            }
            
            self.save_mappings()

    def save_mappings(self):
        """Save current mappings to JSON file"""
        mappings = {
            'cities': self.city_to_country,
            'countries': self.country_aliases
        }
        with open(self.mappings_file, 'w') as f:
            json.dump(mappings, f, indent=2)
    
    def prompt_for_location(self, location: str) -> Dict[str, str]:
        """Interactive prompt for unknown locations"""
        print(f"\nUnknown location found: {location}")
        
        # First check if it's just a country name
        location_lower = location.lower()
        if location_lower in self.country_aliases:
            return {'city': None, 'country': self.country_aliases[location_lower]}
            
        # Check if it's a known format first
        if ',' in location:
            city, country = [part.strip() for part in location.split(',', 1)]
            print(f"Detected city: {city}, country: {country}")
            
            # Verify with user
            verify = input("Is this correct? (y/n): ").lower()
            if verify == 'y':
                # Add to mappings
                self.add_mapping(city.lower(), country)
                return {'city': city, 'country': country}
        
        # If not known format or verification failed, ask user
        print("\nPlease provide location details:")
        city = input("City (or press Enter if none): ").strip()
        country = input("Country: ").strip()
        
        if city:
            self.add_mapping(city.lower(), country)
        
        return {
            'city': city if city else None,
            'country': country
        }
    
    def add_mapping(self, city: str, country: str):
        """Add new mapping and save"""
        self.city_to_country[city.lower()] = country
        self.save_mappings()
    
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process DataFrame and prompt for unknown locations"""
        df = df.copy()
        
        # Add new columns
        df['city'] = None
        df['country'] = None
        
        # Track unique locations to avoid repeated prompts
        processed_locations = {}
        
        for idx, row in df.iterrows():
            location = row['location']
            
            # Skip if no location or just contributions
            if pd.isna(location) or 'contribution' in str(location).lower():
                df.at[idx, 'country'] = 'Undisclosed'
                continue
                
            # Check if we've seen this location before
            if location in processed_locations:
                df.at[idx, 'city'] = processed_locations[location]['city']
                df.at[idx, 'country'] = processed_locations[location]['country']
                continue
            
            # Try to map the location
            location_info = self.map_location(location)
            
            # If unknown, prompt user
            if not location_info['country']:
                location_info = self.prompt_for_location(location)
            
            # Store the result
            processed_locations[location] = location_info
            df.at[idx, 'city'] = location_info['city']
            df.at[idx, 'country'] = location_info['country']
        
        return df
    
    def map_location(self, location: str) -> Dict[str, Optional[str]]:
        """Try to map location using existing mappings"""
        if not location:
            return {'city': None, 'country': 'Undisclosed'}
        
        location = location.strip()
        location_lower = location.lower()
        
        # First check if it's a country or country alias
        if location_lower in self.country_aliases:
            return {'city': None, 'country': self.country_aliases[location_lower]}
            
        # Handle "City, Country" format
        if ',' in location:
            city, country = [part.strip() for part in location.split(',', 1)]
            country_lower = country.lower()
            
            # Check country aliases
            if country_lower in self.country_aliases:
                country = self.country_aliases[country_lower]
            
            # Check if city is mapped
            city_lower = city.lower()
            if city_lower in self.city_to_country:
                country = self.city_to_country[city_lower]
            
            return {'city': city, 'country': country}
        
        # Check if it's a mapped city
        if location_lower in self.city_to_country:
            return {
                'city': location,
                'country': self.city_to_country[location_lower]
            }
        
        return {'city': None, 'country': None}

# Example usage
if __name__ == "__main__":
    # Create mapper instance
    mapper = LocationMapper()
    
    # Read your CSV file
    df = pd.read_csv('reviews.csv')
    
    # Process locations
    df_processed = mapper.process_dataframe(df)
    
    # Save processed data
    df_processed.to_csv('reviews_with_locations.csv', index=False)
    
    # Print some statistics
    print("\nCountry distribution:")
    print(df_processed['country'].value_counts())
    
    print("\nCities by country:")
    for country in df_processed['country'].unique():
        if pd.notna(country):
            cities = df_processed[df_processed['country'] == country]['city'].unique()
            print(f"\n{country}:")
            for city in cities:
                if pd.notna(city):
                    print(f"  - {city}")