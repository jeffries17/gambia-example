import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from fpdf import FPDF
import base64
from io import BytesIO

class TourismReportGenerator:
    """
    Generates comprehensive tourism planning reports from analysis results.
    """
    
    def __init__(self, base_results_dir='tourism_analysis_results', output_dir='tourism_reports'):
        """
        Initialize the report generator.
        
        Parameters:
        - base_results_dir: Directory containing all analysis results
        - output_dir: Directory to save generated reports
        """
        self.base_dir = base_results_dir
        self.output_dir = output_dir
        self.data = {}
        self.recommendations = {}
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created reports directory: {output_dir}")
    
    def load_analysis_data(self):
        """
        Load data from all analysis modules.
        """
        # Directories to look for data
        subdirs = [
            '',  # Main results
            'restaurant_insights',
            'accommodation_insights',
            'activity_insights',
            'seasonal_insights',
            'traveler_insights',
            'price_value_insights',
            'expectation_insights',
            'competitive_insights'
        ]
        
        # Load CSV files
        for subdir in subdirs:
            dir_path = os.path.join(self.base_dir, subdir)
            if os.path.exists(dir_path):
                self.data[subdir] = {}
                for file in os.listdir(dir_path):
                    if file.endswith('.csv'):
                        file_path = os.path.join(dir_path, file)
                        try:
                            self.data[subdir][file] = pd.read_csv(file_path)
                            print(f"Loaded {file_path}")
                        except Exception as e:
                            print(f"Error loading {file_path}: {str(e)}")
        
        # Load recommendation JSON files
        for subdir in subdirs:
            dir_path = os.path.join(self.base_dir, subdir)
            if os.path.exists(dir_path):
                for file in os.listdir(dir_path):
                    if file.endswith('recommendations.json'):
                        file_path = os.path.join(dir_path, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                self.recommendations[subdir] = json.load(f)
                            print(f"Loaded recommendations from {file_path}")
                        except Exception as e:
                            print(f"Error loading {file_path}: {str(e)}")
    
    def generate_html_report(self):
        """
        Generate a comprehensive HTML report.
        
        Returns:
        - Path to the generated HTML file
        """
        # Load data if not already loaded
        if not self.data:
            self.load_analysis_data()
        
        # Create report filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_filename = f"tonga_tourism_report_{timestamp}.html"
        report_path = os.path.join(self.output_dir, report_filename)
        
        # Start building HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Tonga Tourism Planning Report</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 0;
                    color: #333;
                }}
                .container {{
                    width: 90%;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .header {{
                    background-color: #3f51b5;
                    color: white;
                    padding: 20px 0;
                    text-align: center;
                }}
                .section {{
                    margin: 40px 0;
                    border: 1px solid #ddd;
                    padding: 20px;
                    border-radius: 5px;
                }}
                .section-title {{
                    border-bottom: 2px solid #3f51b5;
                    padding-bottom: 10px;
                    margin-bottom: 20px;
                    color: #3f51b5;
                }}
                .visualization {{
                    margin: 20px 0;
                    height: 500px;
                }}
                .recommendations {{
                    background-color: #f5f5f5;
                    padding: 15px;
                    border-radius: 5px;
                }}
                .recommendation-item {{
                    margin-bottom: 10px;
                    padding-left: 20px;
                    position: relative;
                }}
                .recommendation-item:before {{
                    content: "•";
                    position: absolute;
                    left: 0;
                    color: #3f51b5;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    padding: 12px 15px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #3f51b5;
                    color: white;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <div class="container">
                    <h1>Tonga Tourism Planning Report</h1>
                    <p>Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}</p>
                </div>
            </div>
            <div class="container">
                <div class="section">
                    <h2 class="section-title">Executive Summary</h2>
                    <p>This report provides comprehensive insights into tourism patterns and opportunities in Tonga, based on analysis of visitor reviews and feedback. Key findings are organized by business type, seasonal patterns, visitor segments, and more.</p>
        """
        
        # Add executive summary of key recommendations
        html_content += """
                    <h3>Key Recommendations</h3>
                    <div class="recommendations">
        """
        
        # Collect all recommendations
        all_recommendations = []
        
        for module, recs in self.recommendations.items():
            module_name = module.replace('_insights', '').title() if module else 'General'
            
            for category, items in recs.items():
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, str):
                            all_recommendations.append({
                                'Module': module_name,
                                'Category': category.replace('_', ' ').title(),
                                'Recommendation': item
                            })
        
        # Select top recommendations for executive summary
        top_recommendations = all_recommendations[:5]  # Just take first 5 for the summary
        
        for rec in top_recommendations:
            html_content += f"""
                        <div class="recommendation-item">
                            <strong>{rec['Category']}:</strong> {rec['Recommendation']}
                        </div>
            """
        
        html_content += """
                    </div>
                </div>
        """
        
        # Add business type analysis section
        html_content += """
                <div class="section">
                    <h2 class="section-title">Business Type Analysis</h2>
                    <p>This section presents insights specific to different types of tourism businesses in Tonga.</p>
        """
        
        # Add accommodation insights
        if 'accommodation_insights' in self.data:
            html_content += """
                    <h3>Accommodation Insights</h3>
                    <p>Analysis of accommodation reviews reveals patterns in guest preferences and satisfaction.</p>
            """
            
            # Add accommodation type visualization if available
            if 'accommodation_type_mentions.png' in os.listdir(os.path.join(self.base_dir, 'accommodation_insights')):
                img_path = os.path.join(self.base_dir, 'accommodation_insights', 'accommodation_type_mentions.png')
                with open(img_path, "rb") as img_file:
                    b64_img = base64.b64encode(img_file.read()).decode('utf-8')
                html_content += f"""
                    <div>
                        <h4>Accommodation Types Mentioned</h4>
                        <img src="data:image/png;base64,{b64_img}" alt="Accommodation Types" style="max-width:100%;">
                    </div>
                """
            
            # Add accommodation recommendations
            if 'accommodation_insights' in self.recommendations:
                html_content += """
                    <h4>Accommodation Recommendations</h4>
                    <div class="recommendations">
                """
                
                for category, items in self.recommendations['accommodation_insights'].items():
                    if isinstance(items, list) and items:
                        html_content += f"""
                        <h5>{category.replace('_', ' ').title()}</h5>
                        """
                        
                        for item in items:
                            if isinstance(item, str):
                                html_content += f"""
                                <div class="recommendation-item">
                                    {item}
                                </div>
                                """
                
                html_content += """
                    </div>
                """
        
        # Add activity insights
        if 'activity_insights' in self.data:
            html_content += """
                    <h3>Activity Insights</h3>
                    <p>Analysis of activity and attraction reviews highlights visitor preferences and experiences.</p>
            """
            
            # Add activity type visualization if available
            if 'activity_type_mentions.png' in os.listdir(os.path.join(self.base_dir, 'activity_insights')):
                img_path = os.path.join(self.base_dir, 'activity_insights', 'activity_type_mentions.png')
                with open(img_path, "rb") as img_file:
                    b64_img = base64.b64encode(img_file.read()).decode('utf-8')
                html_content += f"""
                    <div>
                        <h4>Activity Types Mentioned</h4>
                        <img src="data:image/png;base64,{b64_img}" alt="Activity Types" style="max-width:100%;">
                    </div>
                """
            
            # Add activity recommendations
            if 'activity_insights' in self.recommendations:
                html_content += """
                    <h4>Activity Recommendations</h4>
                    <div class="recommendations">
                """
                
                for category, items in self.recommendations['activity_insights'].items():
                    if isinstance(items, list) and items:
                        html_content += f"""
                        <h5>{category.replace('_', ' ').title()}</h5>
                        """
                        
                        for item in items:
                            if isinstance(item, str):
                                html_content += f"""
                                <div class="recommendation-item">
                                    {item}
                                </div>
                                """
                
                html_content += """
                    </div>
                """
        
        # Add restaurant insights
        if 'restaurant_insights' in self.data:
            html_content += """
                    <h3>Restaurant and Dining Insights</h3>
                    <p>Analysis of dining and restaurant reviews reveals food preferences and dining experiences.</p>
            """
            
            # Add food category visualization if available
            if 'food_category_mentions.png' in os.listdir(os.path.join(self.base_dir, 'restaurant_insights')):
                img_path = os.path.join(self.base_dir, 'restaurant_insights', 'food_category_mentions.png')
                with open(img_path, "rb") as img_file:
                    b64_img = base64.b64encode(img_file.read()).decode('utf-8')
                html_content += f"""
                    <div>
                        <h4>Food Categories Mentioned</h4>
                        <img src="data:image/png;base64,{b64_img}" alt="Food Categories" style="max-width:100%;">
                    </div>
                """
            
            # Add restaurant recommendations
            if 'restaurant_insights' in self.recommendations:
                html_content += """
                    <h4>Restaurant Recommendations</h4>
                    <div class="recommendations">
                """
                
                for category, items in self.recommendations['restaurant_insights'].items():
                    if isinstance(items, list) and items:
                        html_content += f"""
                        <h5>{category.replace('_', ' ').title()}</h5>
                        """
                        
                        for item in items:
                            if isinstance(item, str):
                                html_content += f"""
                                <div class="recommendation-item">
                                    {item}
                                </div>
                                """
                
                html_content += """
                    </div>
                """
        
        html_content += """
                </div>
        """
        
        # Add visitor segment analysis
        html_content += """
                <div class="section">
                    <h2 class="section-title">Visitor Segment Analysis</h2>
                    <p>This section presents insights on different types of visitors and their preferences.</p>
        """
        
        # Add traveler segment insights
        if 'traveler_insights' in self.data:
            html_content += """
                    <h3>Traveler Segment Insights</h3>
                    <p>Analysis of different traveler types (families, couples, solo travelers, etc.) and their experiences.</p>
            """
            
            # Add traveler segment visualization if available
            if 'sentiment_by_trip_type.png' in os.listdir(os.path.join(self.base_dir, 'traveler_insights')):
                img_path = os.path.join(self.base_dir, 'traveler_insights', 'sentiment_by_trip_type.png')
                with open(img_path, "rb") as img_file:
                    b64_img = base64.b64encode(img_file.read()).decode('utf-8')
                html_content += f"""
                    <div>
                        <h4>Sentiment by Traveler Segment</h4>
                        <img src="data:image/png;base64,{b64_img}" alt="Traveler Sentiment" style="max-width:100%;">
                    </div>
                """
            
            # Add traveler segment recommendations
            if 'traveler_insights' in self.recommendations:
                html_content += """
                    <h4>Traveler Segment Recommendations</h4>
                    <div class="recommendations">
                """
                
                for category, items in self.recommendations['traveler_insights'].items():
                    if isinstance(items, list) and items:
                        html_content += f"""
                        <h5>{category.replace('_', ' ').title()}</h5>
                        """
                        
                        for item in items:
                            if isinstance(item, str):
                                html_content += f"""
                                <div class="recommendation-item">
                                    {item}
                                </div>
                                """
                
                html_content += """
                    </div>
                """
        
        # Add price value insights
        if 'price_value_insights' in self.data:
            html_content += """
                    <h3>Price-Value Perception</h3>
                    <p>Analysis of how different visitor segments perceive the value for money.</p>
            """
            
            # Add price value visualization if available
            if 'value_sentiment_by_category.png' in os.listdir(os.path.join(self.base_dir, 'price_value_insights')):
                img_path = os.path.join(self.base_dir, 'price_value_insights', 'value_sentiment_by_category.png')
                with open(img_path, "rb") as img_file:
                    b64_img = base64.b64encode(img_file.read()).decode('utf-8')
                html_content += f"""
                    <div>
                        <h4>Value Sentiment by Category</h4>
                        <img src="data:image/png;base64,{b64_img}" alt="Price Value Sentiment" style="max-width:100%;">
                    </div>
                """
        
        html_content += """
                </div>
        """
        
        # Add seasonal patterns section
        html_content += """
                <div class="section">
                    <h2 class="section-title">Seasonal Patterns</h2>
                    <p>This section presents insights on how tourism experiences vary by season.</p>
        """
        
        # Add seasonal insights
        if 'seasonal_insights' in self.data:
            html_content += """
                    <h3>Seasonal Trend Analysis</h3>
                    <p>Analysis of how visitor experiences and preferences change throughout the year.</p>
            """
            
            # Add seasonal visualization if available
            if 'sentiment_by_season.png' in os.listdir(os.path.join(self.base_dir, 'seasonal_insights')):
                img_path = os.path.join(self.base_dir, 'seasonal_insights', 'sentiment_by_season.png')
                with open(img_path, "rb") as img_file:
                    b64_img = base64.b64encode(img_file.read()).decode('utf-8')
                html_content += f"""
                    <div>
                        <h4>Sentiment by Season</h4>
                        <img src="data:image/png;base64,{b64_img}" alt="Seasonal Sentiment" style="max-width:100%;">
                    </div>
                """
            
            # Add seasonal recommendations
            if 'seasonal_insights' in self.recommendations:
                html_content += """
                    <h4>Seasonal Recommendations</h4>
                    <div class="recommendations">
                """
                
                for category, items in self.recommendations['seasonal_insights'].items():
                    if isinstance(items, list) and items:
                        html_content += f"""
                        <h5>{category.replace('_', ' ').title()}</h5>
                        """
                        
                        for item in items:
                            if isinstance(item, str):
                                html_content += f"""
                                <div class="recommendation-item">
                                    {item}
                                </div>
                                """
                
                html_content += """
                    </div>
                """
        
        html_content += """
                </div>
        """
        
        # Add expectation vs. reality section
        html_content += """
                <div class="section">
                    <h2 class="section-title">Visitor Expectations vs. Reality</h2>
                    <p>This section highlights gaps between visitor expectations and actual experiences.</p>
        """
        
        # Add expectation insights
        if 'expectation_insights' in self.data:
            html_content += """
                    <h3>Expectation Gap Analysis</h3>
                    <p>Analysis of where visitor expectations are exceeded or not met.</p>
            """
            
            # Add expectation visualization if available
            if 'expectation_gaps.png' in os.listdir(os.path.join(self.base_dir, 'expectation_insights')):
                img_path = os.path.join(self.base_dir, 'expectation_insights', 'expectation_gaps.png')
                with open(img_path, "rb") as img_file:
                    b64_img = base64.b64encode(img_file.read()).decode('utf-8')
                html_content += f"""
                    <div>
                        <h4>Expectation-Experience Gaps</h4>
                        <img src="data:image/png;base64,{b64_img}" alt="Expectation Gaps" style="max-width:100%;">
                    </div>
                """
            
            # Add expectation recommendations
            if 'expectation_insights' in self.recommendations:
                html_content += """
                    <h4>Expectation Management Recommendations</h4>
                    <div class="recommendations">
                """
                
                for category, items in self.recommendations['expectation_insights'].items():
                    if isinstance(items, list) and items:
                        html_content += f"""
                        <h5>{category.replace('_', ' ').title()}</h5>
                        """
                        
                        for item in items:
                            if isinstance(item, str):
                                html_content += f"""
                                <div class="recommendation-item">
                                    {item}
                                </div>
                                """
                
                html_content += """
                    </div>
                """
        
        html_content += """
                </div>
        """
        
        # Add competitive intelligence section
        html_content += """
                <div class="section">
                    <h2 class="section-title">Competitive Intelligence</h2>
                    <p>This section presents insights on how Tonga compares to other destinations.</p>
        """
        
        # Add competitive insights
        if 'competitive_insights' in self.data:
            html_content += """
                    <h3>Competitive Analysis</h3>
                    <p>Analysis of how Tonga compares to other destinations based on visitor reviews.</p>
            """
            
            # Add competitor visualization if available
            if 'competitor_mentions.png' in os.listdir(os.path.join(self.base_dir, 'competitive_insights')):
                img_path = os.path.join(self.base_dir, 'competitive_insights', 'competitor_mentions.png')
                with open(img_path, "rb") as img_file:
                    b64_img = base64.b64encode(img_file.read()).decode('utf-8')
                html_content += f"""
                    <div>
                        <h4>Competitor Destinations Mentioned</h4>
                        <img src="data:image/png;base64,{b64_img}" alt="Competitor Mentions" style="max-width:100%;">
                    </div>
                """
            
            # Add uniqueness visualization if available
            if 'unique_advantages.png' in os.listdir(os.path.join(self.base_dir, 'competitive_insights')):
                img_path = os.path.join(self.base_dir, 'competitive_insights', 'unique_advantages.png')
                with open(img_path, "rb") as img_file:
                    b64_img = base64.b64encode(img_file.read()).decode('utf-8')
                html_content += f"""
                    <div>
                        <h4>Tonga's Unique Advantages</h4>
                        <img src="data:image/png;base64,{b64_img}" alt="Unique Advantages" style="max-width:100%;">
                    </div>
                """
            
            # Add competitive recommendations
            if 'competitive_insights' in self.recommendations:
                html_content += """
                    <h4>Competitive Positioning Recommendations</h4>
                    <div class="recommendations">
                """
                
                for category, items in self.recommendations['competitive_insights'].items():
                    if isinstance(items, list) and items:
                        html_content += f"""
                        <h5>{category.replace('_', ' ').title()}</h5>
                        """
                        
                        for item in items:
                            if isinstance(item, str):
                                html_content += f"""
                                <div class="recommendation-item">
                                    {item}
                                </div>
                                """
                
                html_content += """
                    </div>
                """
        
        html_content += """
                </div>
        """
        
        # Add comprehensive recommendations table
        html_content += """
                <div class="section">
                    <h2 class="section-title">All Recommendations</h2>
                    <p>A comprehensive list of all recommendations from the analysis.</p>
                    <table>
                        <tr>
                            <th>Category</th>
                            <th>Area</th>
                            <th>Recommendation</th>
                        </tr>
        """
        
        for rec in all_recommendations:
            html_content += f"""
                        <tr>
                            <td>{rec['Module']}</td>
                            <td>{rec['Category']}</td>
                            <td>{rec['Recommendation']}</td>
                        </tr>
            """
        
        html_content += """
                    </table>
                </div>
        """
        
        # Add conclusion
        html_content += """
                <div class="section">
                    <h2 class="section-title">Conclusion</h2>
                    <p>This report has presented a comprehensive analysis of tourism reviews for Tonga, highlighting key insights across business types, visitor segments, seasons, and competitive positioning.</p>
                    <p>The recommendations provided are based on data-driven analysis of visitor sentiments and experiences. By implementing these recommendations, Tonga can enhance its tourism offerings, better meet visitor expectations, and strengthen its competitive position in the Pacific tourism market.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Write the HTML content to a file
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Generated HTML report at {report_path}")
        return report_path
    
    def generate_pdf_report(self):
        """
        Generate a PDF report.
        
        Returns:
        - Path to the generated PDF file
        """
        # Load data if not already loaded
        if not self.data:
            self.load_analysis_data()
        
        # Create report filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_filename = f"tonga_tourism_report_{timestamp}.pdf"
        report_path = os.path.join(self.output_dir, report_filename)
        
        # Create PDF object
        pdf = FPDF()
        pdf.add_page()
        
        # Add title and introduction
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Tonga Tourism Planning Report', 0, 1, 'C')
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, f'Generated on {datetime.now().strftime("%B %d, %Y")}', 0, 1, 'C')
        pdf.ln(10)
        
        # Add executive summary
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Executive Summary', 0, 1, 'L')
        pdf.set_font('Arial', '', 12)
        pdf.multi_cell(0, 10, 'This report provides comprehensive insights into tourism patterns and opportunities in Tonga, based on analysis of visitor reviews and feedback. Key findings are organized by business type, seasonal patterns, visitor segments, and more.')
        
        # Add key recommendations
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Key Recommendations:', 0, 1, 'L')
        pdf.set_font('Arial', '', 12)
        
        # Collect all recommendations
        all_recommendations = []
        
        for module, recs in self.recommendations.items():
            module_name = module.replace('_insights', '').title() if module else 'General'
            
            for category, items in recs.items():
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, str):
                            all_recommendations.append({
                                'Module': module_name,
                                'Category': category.replace('_', ' ').title(),
                                'Recommendation': item
                            })
        
        # Select top recommendations for executive summary
        top_recommendations = all_recommendations[:5]  # Just take first 5 for the summary
        
        for rec in top_recommendations:
            pdf.multi_cell(0, 10, f"• {rec['Category']}: {rec['Recommendation']}")
        
        pdf.ln(10)
        
        # Add sections for each analysis module
        modules = [
            ('Accommodation', 'accommodation_insights'),
            ('Activities and Attractions', 'activity_insights'),
            ('Restaurant and Dining', 'restaurant_insights'),
            ('Traveler Segments', 'traveler_insights'),
            ('Seasonal Patterns', 'seasonal_insights'),
            ('Price-Value Perception', 'price_value_insights'),
            ('Visitor Expectations', 'expectation_insights'),
            ('Competitive Intelligence', 'competitive_insights')
        ]
        
        for title, module in modules:
            if module in self.recommendations:
                pdf.add_page()
                pdf.set_font('Arial', 'B', 14)
                pdf.cell(0, 10, title, 0, 1, 'L')
                pdf.ln(5)
                
                # Add module recommendations
                for category, items in self.recommendations[module].items():
                    if isinstance(items, list) and items:
                        pdf.set_font('Arial', 'BI', 12)
                        pdf.cell(0, 10, f"{category.replace('_', ' ').title()}", 0, 1, 'L')
                        pdf.set_font('Arial', '', 12)
                        
                        for item in items:
                            if isinstance(item, str):
                                pdf.multi_cell(0, 10, f"• {item}")
                        
                        pdf.ln(5)
        
        # Add a comprehensive recommendations table
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'All Recommendations', 0, 1, 'L')
        pdf.ln(5)
        
        # Add table header
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(40, 10, 'Category', 1, 0, 'C')
        pdf.cell(50, 10, 'Area', 1, 0, 'C')
        pdf.cell(100, 10, 'Recommendation', 1, 1, 'C')
        
        # Add table rows
        pdf.set_font('Arial', '', 10)
        for i, rec in enumerate(all_recommendations):
            # Start a new page if needed
            if i > 0 and i % 20 == 0:
                pdf.add_page()
                # Add table header again
                pdf.set_font('Arial', 'B', 12)
                pdf.cell(50, 10, 'Area', 1, 0, 'C')
                pdf.cell(100, 10, 'Recommendation', 1, 1, 'C')
                pdf.set_font('Arial', '', 10)
            
            # Calculate how much space we need for the recommendation
            rec_text = rec['Recommendation']
            lines_needed = len(rec_text) // 50 + 1  # Rough estimate of lines needed
            cell_height = max(lines_needed * 5, 10)  # At least 10 points high
            
            # Print the cells
            pdf.cell(40, cell_height, rec['Module'], 1, 0, 'L')
            pdf.cell(50, cell_height, rec['Category'], 1, 0, 'L')
            
            # Handle multi-line recommendation text
            current_x = pdf.get_x()
            current_y = pdf.get_y()
            pdf.multi_cell(100, cell_height / lines_needed, rec_text, 1, 'L')
            pdf.set_xy(current_x + 100, current_y + cell_height)
            pdf.ln()
        
        # Save the PDF
        try:
            pdf.output(report_path)
            print(f"Generated PDF report at {report_path}")
            return report_path
        except Exception as e:
            print(f"Error generating PDF: {str(e)}")
            return None
    
    def generate_plotly_report(self):
        """
        Generate an interactive HTML report using Plotly.
        
        Returns:
        - Path to the generated HTML file
        """
        # Load data if not already loaded
        if not self.data:
            self.load_analysis_data()
        
        # Create report filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_filename = f"tonga_tourism_interactive_report_{timestamp}.html"
        report_path = os.path.join(self.output_dir, report_filename)
        
        # Create a list to store all the figures
        figures = []
        
        # Create theme prevalence plot
        try:
            theme_data = {}
            
            # Try to get restaurant theme data
            if 'restaurant_insights' in self.data and 'theme_by_category.csv' in self.data['restaurant_insights']:
                theme_data['Restaurant'] = self.data['restaurant_insights']['theme_by_category.csv']
            
            # Try to get accommodation theme data
            if 'accommodation_insights' in self.data and 'theme_by_type.csv' in self.data['accommodation_insights']:
                theme_data['Accommodation'] = self.data['accommodation_insights']['theme_by_type.csv']
            
            # Try to get activity theme data
            if 'activity_insights' in self.data and 'theme_by_category.csv' in self.data['activity_insights']:
                theme_data['Activity'] = self.data['activity_insights']['theme_by_category.csv']
            
            if theme_data:
                # Create a dataframe for visualization
                theme_plot_data = []
                
                for business_type, df in theme_data.items():
                    # Prepare data for plot
                    # This will need to be adjusted based on the actual data structure
                    try:
                        # Assuming df has themes as columns and categories as rows
                        themes = df.columns[1:]  # Skip the first column which is likely the category name
                        categories = df.iloc[:, 0]
                        
                        for i, category in enumerate(categories):
                            for j, theme in enumerate(themes):
                                theme_plot_data.append({
                                    'Business Type': business_type,
                                    'Category': category,
                                    'Theme': theme,
                                    'Score': df.iloc[i, j+1]
                                })
                    except Exception as e:
                        print(f"Error preparing theme data for {business_type}: {str(e)}")
                
                if theme_plot_data:
                    theme_df = pd.DataFrame(theme_plot_data)
                    fig = px.bar(
                        theme_df, 
                        x='Theme', 
                        y='Score', 
                        color='Business Type',
                        facet_col='Category',
                        title='Theme Prevalence by Business Type and Category',
                        labels={'Score': 'Prevalence Score'},
                        barmode='group'
                    )
                    figures.append(fig)
        except Exception as e:
            print(f"Error creating theme prevalence plot: {str(e)}")
        
        # Create sentiment map
        try:
            # Try to get sentiment data by traveler segment
            segment_sentiment = None
            aspect_sentiment = None
            
            if 'traveler_insights' in self.data and 'sentiment_by_segment.csv' in self.data['traveler_insights']:
                segment_sentiment = self.data['traveler_insights']['sentiment_by_segment.csv']
            
            # Try to get sentiment data by aspect
            if '' in self.data and 'sentiment_by_theme.csv' in self.data['']:
                aspect_sentiment = self.data['']['sentiment_by_theme.csv']
            
            if segment_sentiment is not None or aspect_sentiment is not None:
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=("Sentiment by Visitor Segment", "Sentiment by Experience Aspect")
                )
                
                # Add segment sentiment bar chart
                if segment_sentiment is not None:
                    try:
                        segments = segment_sentiment['trip_type_standard'].tolist()
                        sentiments = segment_sentiment['sentiment_score'].tolist()
                        
                        fig.add_trace(
                            go.Bar(
                                x=segments,
                                y=sentiments,
                                marker=dict(
                                    color=sentiments,
                                    colorscale='RdBu',
                                    cmin=-1,
                                    cmid=0,
                                    cmax=1
                                ),
                                name="Segment Sentiment"
                            ),
                            row=1, col=1
                        )
                    except Exception as e:
                        print(f"Error creating segment sentiment chart: {str(e)}")
                
                # Add aspect sentiment bar chart
                if aspect_sentiment is not None:
                    try:
                        aspects = aspect_sentiment['primary_theme'].tolist()
                        sentiments = aspect_sentiment['sentiment_score'].tolist()
                        
                        fig.add_trace(
                            go.Bar(
                                x=aspects,
                                y=sentiments,
                                marker=dict(
                                    color=sentiments,
                                    colorscale='RdBu',
                                    cmin=-1,
                                    cmid=0,
                                    cmax=1
                                ),
                                name="Aspect Sentiment"
                            ),
                            row=2, col=1
                        )
                    except Exception as e:
                        print(f"Error creating aspect sentiment chart: {str(e)}")
                
                fig.update_layout(
                    height=800,
                    title_text="Sentiment Patterns in Tourism Experiences"
                )
                
                figures.append(fig)
        except Exception as e:
            print(f"Error creating sentiment map: {str(e)}")
        
        # Create expectation gap matrix
        try:
            # Try to get expectation gap data
            if 'expectation_insights' in self.data and 'expectation_gap_analysis.csv' in self.data['expectation_insights']:
                gap_df = self.data['expectation_insights']['expectation_gap_analysis.csv']
                
                # Create a bubble chart
                fig = px.scatter(
                    gap_df, 
                    x="sentiment_gap",  # Assuming these columns exist
                    y="surprise_ratio",
                    size="total_mentions",
                    color="sentiment_gap",
                    hover_name=gap_df.index,
                    color_continuous_scale=px.colors.sequential.Viridis,
                    title="Expectation-Experience Gap Matrix"
                )
                
                fig.update_layout(
                    xaxis_title="Sentiment Gap (Positive - Negative)",
                    yaxis_title="Positive to Negative Surprise Ratio",
                    height=600,
                    coloraxis_colorbar=dict(title="Sentiment Gap")
                )
                
                # Add a reference line at x=0 and y=1
                fig.add_shape(type="line", x0=0, y0=0, x1=0, y1=1, line=dict(color="red", width=2, dash="dash"))
                fig.add_shape(type="line", x0=-1, y0=1, x1=1, y1=1, line=dict(color="red", width=2, dash="dash"))
                
                fig.add_annotation(x=0.5, y=0.95, text="Positive Surprises > Negative",
                                 showarrow=False, yshift=10)
                fig.add_annotation(x=0.5, y=0.05, text="Negative Surprises > Positive",
                                 showarrow=False, yshift=-10)
                
                figures.append(fig)
        except Exception as e:
            print(f"Error creating expectation gap matrix: {str(e)}")
        
        # Create seasonal patterns plot
        try:
            # Try to get seasonal data
            if 'seasonal_insights' in self.data and 'sentiment_by_month.csv' in self.data['seasonal_insights']:
                month_sentiment = self.data['seasonal_insights']['sentiment_by_month.csv']
                
                # Create a line chart
                fig = px.line(
                    month_sentiment,
                    x="month_name",
                    y="sentiment_score",
                    markers=True,
                    title="Seasonal Sentiment Patterns"
                )
                
                fig.update_layout(
                    xaxis_title="Month",
                    yaxis_title="Average Sentiment Score (-1 to 1)",
                    height=500
                )
                
                figures.append(fig)
        except Exception as e:
            print(f"Error creating seasonal patterns plot: {str(e)}")
        
        # Create competitive intelligence plot
        try:
            # Try to get competitor data
            competitor_columns = [col for col in self.data.get('competitive_insights', {}).keys() if 'competitor' in col]
            
            if competitor_columns:
                competitor_data = []
                
                for file in competitor_columns:
                    df = self.data['competitive_insights'][file]
                    competitor_data.append({
                        'Destination': file.replace('competitor_', '').replace('.csv', '').replace('_', ' ').title(),
                        'Mentions': df['count'].sum() if 'count' in df.columns else len(df)
                    })
                
                if competitor_data:
                    competitor_df = pd.DataFrame(competitor_data)
                    competitor_df = competitor_df.sort_values('Mentions', ascending=False)
                    
                    fig = px.bar(
                        competitor_df,
                        x='Destination',
                        y='Mentions',
                        title="Competitor Destinations Mentioned",
                        color='Mentions',
                        color_continuous_scale=px.colors.sequential.Blues
                    )
                    
                    fig.update_layout(
                        xaxis_title="Destination",
                        yaxis_title="Number of Mentions",
                        height=500
                    )
                    
                    figures.append(fig)
        except Exception as e:
            print(f"Error creating competitive intelligence plot: {str(e)}")
        
        # Create recommendations table
        try:
            # Collect all recommendations
            all_recommendations = []
            
            for module, recs in self.recommendations.items():
                module_name = module.replace('_insights', '').title() if module else 'General'
                
                for category, items in recs.items():
                    if isinstance(items, list):
                        for item in items:
                            if isinstance(item, str):
                                all_recommendations.append({
                                    'Module': module_name,
                                    'Category': category.replace('_', ' ').title(),
                                    'Recommendation': item
                                })
            
            if all_recommendations:
                rec_df = pd.DataFrame(all_recommendations)
                
                fig = go.Figure(data=[go.Table(
                    header=dict(
                        values=list(rec_df.columns),
                        fill_color='paleturquoise',
                        align='left'
                    ),
                    cells=dict(
                        values=[rec_df[col] for col in rec_df.columns],
                        fill_color='lavender',
                        align='left',
                        height=30
                    )
                )])
                
                fig.update_layout(
                    title="Tourism Planning Recommendations",
                    height=800
                )
                
                figures.append(fig)
        except Exception as e:
            print(f"Error creating recommendations table: {str(e)}")
        
        # Create HTML with all figures
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Tonga Tourism Interactive Report</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 0;
                    color: #333;
                }}
                .container {{
                    width: 90%;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .header {{
                    background-color: #3f51b5;
                    color: white;
                    padding: 20px 0;
                    text-align: center;
                }}
                .plot-container {{
                    margin: 40px 0;
                    border: 1px solid #ddd;
                    padding: 20px;
                    border-radius: 5px;
                }}
                h2 {{
                    color: #3f51b5;
                    border-bottom: 2px solid #3f51b5;
                    padding-bottom: 10px;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <div class="container">
                    <h1>Tonga Tourism Interactive Report</h1>
                    <p>Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}</p>
                </div>
            </div>
            <div class="container">
                <h2>Introduction</h2>
                <p>This interactive report presents key insights from the analysis of tourism reviews for Tonga. Use the interactive charts below to explore patterns in visitor experiences, preferences, and satisfaction.</p>
        """
        
        # Add each figure to the HTML
        for i, fig in enumerate(figures):
            div_id = f"plot_{i}"
            html_content += f"""
                <div class="plot-container">
                    <div id="{div_id}" style="height: 600px;"></div>
                </div>
                <script>
                    var plotData = {fig.to_json()};
                    Plotly.newPlot('{div_id}', plotData.data, plotData.layout);
                </script>
            """
        
        # Add conclusion
        html_content += """
                <h2>Conclusion</h2>
                <p>This interactive report has presented key visualizations of tourism patterns and opportunities in Tonga. The insights provided can guide tourism planning efforts to enhance visitor experiences and strengthen Tonga's position in the competitive tourism market.</p>
            </div>
        </body>
        </html>
        """
        
        # Write the HTML content to a file
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Generated interactive report at {report_path}")
        return report_path
    
    def generate_all_reports(self):
        """
        Generate all report types.
        
        Returns:
        - Dictionary with paths to all generated reports
        """
        reports = {}
        
        print("Generating HTML report...")
        reports['html'] = self.generate_html_report()
        
        print("Generating interactive Plotly report...")
        reports['plotly'] = self.generate_plotly_report()
        
        try:
            print("Generating PDF report...")
            reports['pdf'] = self.generate_pdf_report()
        except Exception as e:
            print(f"Error generating PDF report: {str(e)}")
            reports['pdf'] = None
        
        return reports


# Function to run the report generator standalone
def generate_tourism_reports(results_dir='tourism_analysis_results', output_dir='tourism_reports'):
    """
    Create tourism planning reports from analysis results.
    
    Parameters:
    - results_dir: Directory containing analysis results
    - output_dir: Directory to save generated reports
    
    Returns:
    - Dictionary with paths to generated reports
    """
    generator = TourismReportGenerator(results_dir, output_dir)
    return generator.generate_all_reports()


# Run report generator if script is executed directly
if __name__ == '__main__':
    reports = generate_tourism_reports()
    print("\nReport generation complete.")
    for report_type, path in reports.items():
        if path:
            print(f"  {report_type.upper()} report: {path}")