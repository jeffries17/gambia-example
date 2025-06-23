import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

class VisitorExpectations:
    def __init__(self, reviews_df):
        """
        Initializes the VisitorExpectations object with the reviews DataFrame.
        """
        self.reviews_df = reviews_df

    def identify_expectation_keywords(self):
        """
        Adds a column 'expectation_mentioned' to identify reviews mentioning key expectation words.
        """
        expectation_keywords = ['hoped', 'expected', 'anticipated', 'disappointed', 'surprised']
        pattern = '|'.join(expectation_keywords)
        self.reviews_df['expectation_mentioned'] = self.reviews_df['text'].str.contains(pattern, case=False, na=False)
        return self.reviews_df

    def analyze_expectation_discrepancies(self):
        """
        Analyzes ratings where expectations were explicitly mentioned.
        """
        if 'expectation_mentioned' not in self.reviews_df.columns:
            print("Expectation data not identified. Run identify_expectation_keywords first.")
            return None
        
        expectation_reviews = self.reviews_df[self.reviews_df['expectation_mentioned']]
        return expectation_reviews['rating'].describe()

    def visualize_expectation_discrepancies(self):
        """
        Visualizes the distribution of ratings for reviews with expectation keywords.
        """
        if 'expectation_mentioned' not in self.reviews_df.columns:
            print("Expectation data not identified. Run identify_expectation_keywords first.")
            return

        output_dir = 'outputs/visitor_expectations'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        plt.figure(figsize=(10, 6))
        sns.histplot(self.reviews_df[self.reviews_df['expectation_mentioned']]['rating'], bins=5, kde=False)
        plt.title('Distribution of Ratings with Expectation Keywords')
        plt.xlabel('Rating')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(output_dir, 'ratings_with_expectations.png'))
        plt.close()

# Example usage will be outlined in the test script.
