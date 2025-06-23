import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from collections import defaultdict
from review_analyzer import TongaReviewAnalyzer

class NationalityFindings:
    """
    Analyzes visitor experiences by nationality, breaking down ratings and preferences
    for accommodations, restaurants, and attractions.
    """
    
    def __init__(self, data_dir='tonga_data', output_dir='outputs/by_nationality'):
        """
        Initialize the nationality findings analyzer.
        
        Parameters:
        - data_dir: Directory containing the review data
        - output_dir: Directory to save findings and visualizations
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # Create output directory structure
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Create subdirectories for different categories
        for category in ['accommodations', 'restaurants', 'attractions', 'visualizations']:
            category_dir = os.path.join(output_dir, category)
            if not os.path.exists(category_dir):
                os.makedirs(category_dir)
                
        # Target countries to focus on
        self.target_countries = ['New Zealand', 'Australia', 'United States', 'United Kingdom', 'Tonga']
        
        # Define color scheme for consistent visualizations
        self.country_colors = {
            'New Zealand': '#1E88E5',   # Blue
            'Australia': '#FFC107',     # Amber
            'United States': '#D81B60', # Pink
            'United Kingdom': '#004D40', # Teal
            'Tonga': '#43A047'          # Green
        }
        
        # Load data
        self.reviews_df = None
        
    def load_data(self):
        """
        Load review data using the TongaReviewAnalyzer class.
        """
        print("Loading review data...")
        
        # Use the existing review analyzer to load data
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        analyzer = TongaReviewAnalyzer(
            data_dir=os.path.join(parent_dir, self.data_dir),
            output_dir=self.output_dir
        )
        analyzer.load_data()
        
        if analyzer.all_reviews_df is None or len(analyzer.all_reviews_df) == 0:
            print("No review data available for analysis!")
            return False
            
        # Store review data
        self.reviews_df = analyzer.all_reviews_df
        
        # Extract country from user_location
        self._extract_country_information()
        
        print(f"Loaded {len(self.reviews_df)} reviews for analysis.")
        return True
        
    def _extract_country_information(self):
        """
        Extract and standardize country information from user_location.
        """
        if 'user_location' not in self.reviews_df.columns:
            print("Warning: No user_location data available for country extraction.")
            self.reviews_df['country'] = 'Unknown'
            return
            
        # Common country mappings for standardization
        country_mappings = {
            'australia': 'Australia',
            'new zealand': 'New Zealand',
            'nz': 'New Zealand',
            'usa': 'United States',
            'united states': 'United States',
            'us': 'United States',
            'america': 'United States',
            'uk': 'United Kingdom',
            'united kingdom': 'United Kingdom',
            'england': 'United Kingdom',
            'great britain': 'United Kingdom',
            'tonga': 'Tonga'
        }
        
        # Extract country from location string
        def extract_country(location):
            if pd.isna(location) or location == '' or location == 'Unknown':
                return 'Unknown'
                
            location = str(location).lower()
            
            # Direct mapping
            for key, value in country_mappings.items():
                if key in location:
                    return value
                    
            # Try from country format (e.g., "City, Country")
            parts = [part.strip() for part in location.split(',')]
            if len(parts) > 1:
                for key, value in country_mappings.items():
                    if key in parts[-1]:
                        return value
                        
            return 'Other'
            
        # Apply extraction
        self.reviews_df['country'] = self.reviews_df['user_location'].apply(extract_country)
        
        # Convert 'nan' and empty strings to 'Unknown'
        self.reviews_df.loc[self.reviews_df['country'].isin(['nan', '', None]), 'country'] = 'Unknown'
        
        # Print summary of countries
        country_counts = self.reviews_df['country'].value_counts()
        print("\nCountry distribution in data:")
        for country, count in country_counts.items():
            if country in self.target_countries or country == 'Unknown' or country == 'Other':
                print(f"- {country}: {count}")
        
    def analyze_by_nationality_and_category(self):
        """
        Generate comprehensive analysis of visitor experiences by nationality,
        broken down by category (accommodations, restaurants, attractions).
        """
        if self.reviews_df is None:
            print("No data loaded. Please run load_data() first.")
            return {}
            
        print("\nAnalyzing experiences by nationality and category...")
        
        # Filter to include only target countries and reviews with ratings
        analysis_df = self.reviews_df[
            (self.reviews_df['country'].isin(self.target_countries)) & 
            (self.reviews_df['rating'].notna())
        ].copy()
        
        # Create a results structure
        results = {
            'overall': self._analyze_overall_by_nationality(analysis_df),
            'by_category': {}
        }
        
        # Analyze each category separately
        for category in ['accommodation', 'restaurant', 'attraction']:
            category_df = analysis_df[analysis_df['category'] == category]
            if len(category_df) > 0:
                results['by_category'][category] = self._analyze_category_by_nationality(category_df, category)
                
        # Save results to JSON
        with open(os.path.join(self.output_dir, 'nationality_findings.json'), 'w') as f:
            json.dump(results, f, indent=4)
            
        # Generate visualizations
        self._generate_visualizations(results)
        
        return results
        
    def _analyze_overall_by_nationality(self, df):
        """
        Analyze overall ratings and review counts by nationality.
        
        Parameters:
        - df: DataFrame containing reviews
        
        Returns:
        - Dictionary with analysis results
        """
        # Group by country
        country_stats = df.groupby('country').agg({
            'rating': ['count', 'mean', 'median', 'std'],
            'id': 'count'
        })
        
        # Flatten MultiIndex
        country_stats.columns = ['review_count', 'avg_rating', 'median_rating', 'rating_std', 'total_count']
        
        # Convert to regular dictionary for JSON serialization
        return country_stats.to_dict(orient='index')
        
    def _analyze_category_by_nationality(self, df, category):
        """
        Analyze ratings and review counts by nationality for a specific category.
        
        Parameters:
        - df: DataFrame containing category-specific reviews
        - category: Category name (accommodation, restaurant, attraction)
        
        Returns:
        - Dictionary with category-specific analysis
        """
        # Group by country
        country_stats = df.groupby('country').agg({
            'rating': ['count', 'mean', 'median', 'std'],
            'id': 'count'
        })
        
        # Flatten MultiIndex
        country_stats.columns = ['review_count', 'avg_rating', 'median_rating', 'rating_std', 'total_count']
        
        # Basic results
        results = {
            'stats': country_stats.to_dict(orient='index'),
            'findings': []
        }
        
        # Generate key findings about this category
        if len(country_stats) >= 2:
            # Find highest and lowest rating countries
            max_rating_country = country_stats['avg_rating'].idxmax()
            min_rating_country = country_stats['avg_rating'].idxmin() 
            
            max_rating = country_stats.loc[max_rating_country, 'avg_rating']
            min_rating = country_stats.loc[min_rating_country, 'avg_rating']
            
            # Add findings
            results['findings'].append({
                'type': 'high_rating',
                'finding': f"Visitors from {max_rating_country} rate {category}s the highest " +
                           f"with an average of {max_rating:.2f}/5"
            })
            
            results['findings'].append({
                'type': 'low_rating',
                'finding': f"Visitors from {min_rating_country} rate {category}s the lowest " +
                           f"with an average of {min_rating:.2f}/5"
            })
            
            # Find rating gaps
            if max_rating - min_rating >= 0.5:
                results['findings'].append({
                    'type': 'rating_gap',
                    'finding': f"There is a significant gap of {max_rating - min_rating:.2f} points " +
                               f"between how visitors from {max_rating_country} and {min_rating_country} " +
                               f"rate {category}s"
                })
                
            # Check for country with most reviews
            most_reviews_country = country_stats['review_count'].idxmax()
            most_reviews_count = country_stats.loc[most_reviews_country, 'review_count']
            
            results['findings'].append({
                'type': 'most_reviews',
                'finding': f"Visitors from {most_reviews_country} leave the most reviews for {category}s " +
                           f"with {most_reviews_count} reviews"
            })
            
        # Save country-level data
        category_dir = os.path.join(self.output_dir, f"{category}s")
        os.makedirs(category_dir, exist_ok=True)
        
        # Save country stats for this category
        country_stats.to_csv(os.path.join(category_dir, f"{category}_by_nationality.csv"))
        
        return results
        
    def _generate_visualizations(self, results):
        """
        Generate visualizations for nationality-based analysis.
        
        Parameters:
        - results: Dictionary with analysis results
        """
        print("\nGenerating visualizations...")
        
        # Create visualization directory
        viz_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Get available countries
        countries = list(results['overall'].keys())
        
        # Prepare colors for plotting
        colors = [self.country_colors.get(country, '#BDBDBD') for country in countries]
        
        # 1. Overall ratings by nationality
        self._create_rating_comparison(results['overall'], 'Overall', viz_dir)
        
        # 2. Category-specific visualizations
        for category, category_data in results['by_category'].items():
            # Create category-specific visualizations
            self._create_rating_comparison(category_data['stats'], category.title(), viz_dir)
            
        # 3. Create side-by-side category comparison for each nationality
        self._create_category_comparison_by_nationality(results, viz_dir)
        
        # 4. Create nationality comparison across all categories
        self._create_nationality_category_matrix(results, viz_dir)
        
        print(f"Visualizations saved to {viz_dir}")
        
    def _create_rating_comparison(self, stats, category, output_dir):
        """
        Create comparison charts for ratings by nationality for a specific category.
        
        Parameters:
        - stats: Dictionary with statistics
        - category: Category name
        - output_dir: Output directory
        """
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame.from_dict(stats, orient='index')
        
        # Sort by avg_rating
        df = df.sort_values('avg_rating', ascending=False)
        
        # Get countries and prepare colors
        countries = df.index.tolist()
        colors = [self.country_colors.get(country, '#BDBDBD') for country in countries]
        
        # 1. Average rating by nationality
        plt.figure(figsize=(10, 6))
        bars = plt.bar(countries, df['avg_rating'], color=colors)
        
        # Add rating values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.05,
                f'{height:.2f}',
                ha='center', va='bottom',
                fontweight='bold'
            )
            
        plt.title(f'Average Rating by Nationality - {category}', fontsize=14, pad=20)
        plt.ylabel('Average Rating', fontsize=12)
        plt.xlabel('Country', fontsize=12)
        plt.ylim(0, 5.5)  # Set y-axis for 5-point rating scale
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'avg_rating_by_nationality_{category.lower()}.png'), dpi=300)
        plt.close()
        
        # 2. Review count by nationality
        plt.figure(figsize=(10, 6))
        bars = plt.bar(countries, df['review_count'], color=colors)
        
        # Add count values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height + (df['review_count'].max() * 0.02),
                f'{int(height)}',
                ha='center', va='bottom',
                fontweight='bold'
            )
            
        plt.title(f'Review Count by Nationality - {category}', fontsize=14, pad=20)
        plt.ylabel('Number of Reviews', fontsize=12)
        plt.xlabel('Country', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'review_count_by_nationality_{category.lower()}.png'), dpi=300)
        plt.close()
        
        # 3. Combined chart (rating and count)
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Set up primary axis for ratings
        color = 'tab:blue'
        ax1.set_xlabel('Country')
        ax1.set_ylabel('Average Rating', color=color)
        bars = ax1.bar(countries, df['avg_rating'], color=colors, alpha=0.7)
        
        # Add rating labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.1,
                f'{height:.2f}',
                ha='center', va='bottom',
                color=color, fontweight='bold'
            )
            
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(0, 5.5)
        
        # Secondary axis for review count
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Number of Reviews', color=color)
        line = ax2.plot(countries, df['review_count'], marker='o', color=color, linewidth=2)
        
        # Add count labels
        for i, val in enumerate(df['review_count']):
            ax2.text(
                i,
                val + (0.03 * df['review_count'].max()),
                f'{int(val)}',
                ha='center', va='bottom',
                color=color, fontweight='bold'
            )
            
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title(f'Nationality Analysis: {category}', fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'nationality_analysis_{category.lower()}.png'), dpi=300)
        plt.close()
        
    def _create_category_comparison_by_nationality(self, results, output_dir):
        """
        Create comparison charts showing how each nationality rates different categories.
        
        Parameters:
        - results: Dictionary with analysis results
        - output_dir: Output directory
        """
        # Extract data for each category by nationality
        category_ratings = defaultdict(dict)
        category_counts = defaultdict(dict)
        
        for nationality in self.target_countries:
            # Add overall rating if available
            if nationality in results['overall']:
                category_ratings[nationality]['Overall'] = results['overall'][nationality]['avg_rating']
                category_counts[nationality]['Overall'] = results['overall'][nationality]['review_count']
                
            # Add category-specific ratings
            for category, category_data in results['by_category'].items():
                if nationality in category_data['stats']:
                    # Use title case for category names
                    category_name = category.title()
                    category_ratings[nationality][category_name] = category_data['stats'][nationality]['avg_rating']
                    category_counts[nationality][category_name] = category_data['stats'][nationality]['review_count']
        
        # Create a chart for each nationality
        for nationality in category_ratings.keys():
            # Skip if not enough data
            if len(category_ratings[nationality]) < 2:
                continue
                
            # Convert to DataFrame
            df = pd.DataFrame({
                'Category': list(category_ratings[nationality].keys()),
                'Rating': list(category_ratings[nationality].values()),
                'Count': list(category_counts[nationality].values())
            })
            
            # Sort by rating
            df = df.sort_values('Rating', ascending=False)
            
            # Create color list
            category_colors = {
                'Overall': '#5D4037',      # Brown
                'Accommodation': '#1976D2', # Blue
                'Restaurant': '#D32F2F',   # Red
                'Attraction': '#388E3C'    # Green
            }
            colors = [category_colors.get(cat, '#BDBDBD') for cat in df['Category']]
            
            # Create visualization
            fig, ax1 = plt.subplots(figsize=(10, 6))
            
            # Primary axis for ratings
            color = 'tab:blue'
            ax1.set_xlabel('Category')
            ax1.set_ylabel('Average Rating', color=color)
            bars = ax1.bar(df['Category'], df['Rating'], color=colors, alpha=0.85)
            
            # Add rating labels
            for bar in bars:
                height = bar.get_height()
                ax1.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + 0.1,
                    f'{height:.2f}',
                    ha='center', va='bottom',
                    color=color, fontweight='bold'
                )
                
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.set_ylim(0, 5.5)
            
            # Secondary axis for review count
            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel('Number of Reviews', color=color)
            line = ax2.plot(df['Category'], df['Count'], marker='o', color=color, linewidth=2)
            
            # Add count labels
            for i, val in enumerate(df['Count']):
                ax2.text(
                    i,
                    val + (0.03 * df['Count'].max()),
                    f'{int(val)}',
                    ha='center', va='bottom',
                    color=color, fontweight='bold'
                )
                
            ax2.tick_params(axis='y', labelcolor=color)
            
            plt.title(f'Category Comparison for {nationality} Visitors', fontsize=14, pad=20)
            plt.tight_layout()
            
            # Save the figure
            filename = f'category_comparison_{nationality.lower().replace(" ", "_")}.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=300)
            plt.close()
            
    def _create_nationality_category_matrix(self, results, output_dir):
        """
        Create a heatmap matrix comparing nationalities across categories.
        
        Parameters:
        - results: Dictionary with analysis results
        - output_dir: Output directory
        """
        # Prepare data for the matrix
        matrix_data = []
        
        # Get all categories
        categories = list(results['by_category'].keys())
        categories = [cat.title() for cat in categories]
        categories.append('Overall')
        
        # Get all nationalities
        nationalities = self.target_countries
        
        # Build matrix data
        for nationality in nationalities:
            row_data = {'Nationality': nationality}
            
            # Add overall rating if available
            if nationality in results['overall']:
                row_data['Overall'] = results['overall'][nationality]['avg_rating']
                
            # Add category-specific ratings
            for category_original in results['by_category'].keys():
                category = category_original.title()
                if nationality in results['by_category'][category_original]['stats']:
                    row_data[category] = results['by_category'][category_original]['stats'][nationality]['avg_rating']
                else:
                    row_data[category] = np.nan
                    
            matrix_data.append(row_data)
            
        # Convert to DataFrame
        matrix_df = pd.DataFrame(matrix_data)
        
        # Set nationality as index
        matrix_df.set_index('Nationality', inplace=True)
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        ax = sns.heatmap(matrix_df, annot=True, cmap='RdYlGn', fmt='.2f', linewidths=.5)
        
        # Set title and adjust layout
        plt.title('Average Ratings by Nationality and Category', fontsize=16, pad=20)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(output_dir, 'nationality_category_heatmap.png'), dpi=300)
        plt.close()
        
        # Create a radar chart comparing nationalities
        self._create_radar_chart(matrix_df, output_dir)
        
    def _create_radar_chart(self, df, output_dir):
        """
        Create a radar chart comparing nationalities across categories.
        
        Parameters:
        - df: DataFrame with ratings by nationality and category
        - output_dir: Output directory
        """
        # Get categories and nationalities
        categories = df.columns.tolist()
        nationalities = df.index.tolist()
        
        # Create radar chart
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, polar=True)
        
        # Set the angles for each category
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        
        # Close the loop
        angles += angles[:1]
        
        # Set radial limits
        ax.set_ylim(0, 5)
        
        # Add category labels
        plt.xticks(angles[:-1], categories, size=12)
        
        # Add radial labels
        ax.set_rlabel_position(0)
        plt.yticks([1, 2, 3, 4, 5], ["1", "2", "3", "4", "5"], color="grey", size=10)
        
        # Plot each nationality
        for i, nationality in enumerate(nationalities):
            values = df.loc[nationality].values.flatten().tolist()
            
            # Close the loop
            values += values[:1]
            
            # Get color for this nationality
            color = self.country_colors.get(nationality, '#BDBDBD')
            
            # Plot the line
            ax.plot(angles, values, linewidth=2, linestyle='solid', color=color, label=nationality)
            ax.fill(angles, values, color=color, alpha=0.1)
            
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Set title
        plt.title('Nationality Comparison Across Categories', size=16, pad=20)
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'nationality_category_radar.png'), dpi=300)
        plt.close()

def run_nationality_findings():
    """
    Run the nationality findings analysis.
    """
    # Set up paths
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(parent_dir, 'tonga_data')
    output_dir = os.path.join(parent_dir, 'outputs', 'nationality_findings')
    
    # Initialize analyzer
    analyzer = NationalityFindings(data_dir=data_dir, output_dir=output_dir)
    
    # Load data
    if analyzer.load_data():
        # Run analysis
        results = analyzer.analyze_by_nationality_and_category()
        
        # Print key findings
        print("\nKey Findings by Nationality and Category:")
        
        # Print overall findings
        print("\nOverall Rating by Nationality:")
        for country, stats in results['overall'].items():
            print(f"- {country}: {stats['avg_rating']:.2f}/5 ({stats['review_count']} reviews)")
            
        # Print category-specific findings
        for category, data in results['by_category'].items():
            print(f"\n{category.title()} Findings:")
            for finding in data['findings']:
                print(f"- {finding['finding']}")
                
        print(f"\nDetailed results and visualizations saved to: {output_dir}")
        
    else:
        print("Analysis failed due to data loading issues.")

if __name__ == "__main__":
    run_nationality_findings()