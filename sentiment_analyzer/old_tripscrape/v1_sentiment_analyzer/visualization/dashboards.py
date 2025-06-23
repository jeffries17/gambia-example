import os
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np

class TourismDashboard:
    def __init__(self, base_results_dir='tourism_analysis_results'):
        # Handle potential nested directory structure
        nested_dir = os.path.join(base_results_dir, 'tourism_analysis_results')
        if os.path.exists(nested_dir):
            # Use the nested directory for business type insights
            self.business_insights_dir = nested_dir
            # Use parent directory for other insights
            self.other_insights_dir = base_results_dir
        else:
            # All insights are in the base directory
            self.business_insights_dir = base_results_dir
            self.other_insights_dir = base_results_dir
            
        self.data = {}
        self.recommendations = {}
        self.app = None
    
    def load_analysis_data(self):
        """
        Load data from all analysis modules with correct file names,
        handling the nested directory structure.
        """
        # Business-type insights (in potentially nested directory)
        business_insights = {
            'restaurant_insights': {
                'csv_files': ['themes_by_trip_type.csv', 'cuisine_comparison.csv'],
                'json_files': ['restaurant_recommendations.json']
            },
            'accommodation_insights': {
                'csv_files': ['trip_purpose_preferences.csv', 'accommodation_satisfaction_metrics.csv'],
                'json_files': ['accommodation_recommendations.json']
            },
            'activity_insights': {
                'csv_files': ['segment_activity_preferences.csv', 'activity_satisfaction_metrics.csv'],
                'json_files': ['activity_recommendations.json']
            }
        }

        # Other insights (in parent directory)
        other_insights = {
            'seasonal_insights': {
                'csv_files': ['seasonal_activity_patterns.csv', 'seasonal_sentiment.csv'],
                'json_files': ['seasonal_recommendations.json']
            },
            'traveler_insights': {
                'csv_files': ['theme_seasonality.csv', 'cross_segment_sentiment.csv'],
                'json_files': ['traveler_segment_recommendations.json']
            },
            'price_value_insights': {
                'csv_files': ['expectation_gap_analysis.csv'],
                'json_files': ['price_value_recommendations.json']
            },
            'expectation_insights': {
                'csv_files': ['expectation_gap_analysis.csv'],
                'json_files': ['expectation_recommendations.json']
            },
            'competitive_insights': {
                'csv_files': ['competitor_analysis.csv'],
                'json_files': ['competitive_recommendations.json']
            }
        }
        # Known analysis directories and their expected files
        analysis_structure = {
            'restaurant_insights': {
                'csv_files': [
                    'themes_by_trip_type.csv',
                    'cuisine_comparison.csv',
                ],
                'json_files': ['restaurant_recommendations.json']
            },
            'accommodation_insights': {
                'csv_files': [
                    'trip_purpose_preferences.csv',
                    'accommodation_satisfaction_metrics.csv',
                ],
                'json_files': ['accommodation_recommendations.json']
            },
            'activity_insights': {
                'csv_files': [
                    'segment_activity_preferences.csv',
                    'activity_satisfaction_metrics.csv',
                ],
                'json_files': ['activity_recommendations.json']
            },
            'seasonal_insights': {
                'csv_files': [
                    'seasonal_activity_patterns.csv',
                    'seasonal_sentiment.csv',
                ],
                'json_files': ['seasonal_recommendations.json']
            },
            'traveler_insights': {
                'csv_files': [
                    'theme_seasonality.csv',
                    'cross_segment_sentiment.csv',
                ],
                'json_files': ['traveler_segment_recommendations.json']
            },
            'price_value_insights': {
                'csv_files': [
                    'expectation_gap_analysis.csv',
                ],
                'json_files': ['price_value_recommendations.json']
            },
            'expectation_insights': {
                'csv_files': [
                    'expectation_gap_analysis.csv',
                ],
                'json_files': ['expectation_recommendations.json']
            },
            'competitive_insights': {
                'csv_files': [
                    'competitor_analysis.csv',
                ],
                'json_files': ['competitive_recommendations.json']
            }
        }

        print("Starting data load...")
        
        # Load business insights from potentially nested directory
        print("\nLoading business insights...")
        for subdir, expected_files in business_insights.items():
            dir_path = os.path.join(self.business_insights_dir, subdir)
            self._load_directory_data(dir_path, subdir, expected_files)

        # Load other insights from parent directory
        print("\nLoading other insights...")
        for subdir, expected_files in other_insights.items():
            dir_path = os.path.join(self.other_insights_dir, subdir)
            self._load_directory_data(dir_path, subdir, expected_files)

    def _load_directory_data(self, dir_path, subdir, expected_files):
            if os.path.exists(dir_path):
                print(f"\nChecking directory: {dir_path}")
                self.data[subdir] = {}
                
                # Load CSV files
                for csv_file in expected_files['csv_files']:
                    file_path = os.path.join(dir_path, csv_file)
                    if os.path.exists(file_path):
                        try:
                            self.data[subdir][csv_file] = pd.read_csv(file_path)
                            print(f"Loaded CSV: {csv_file}")
                        except Exception as e:
                            print(f"Error loading {csv_file}: {str(e)}")
                    else:
                        print(f"Missing CSV file: {csv_file}")
                
                # Load JSON files
                for json_file in expected_files['json_files']:
                    file_path = os.path.join(dir_path, json_file)
                    if os.path.exists(file_path):
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                self.recommendations[subdir] = json.load(f)
                            print(f"Loaded JSON: {json_file}")
                        except Exception as e:
                            print(f"Error loading {json_file}: {str(e)}")
                    else:
                        print(f"Missing JSON file: {json_file}")
            else:
                print(f"\nDirectory not found: {dir_path}")

    def create_theme_prevalence_dashboard(self):
        """
        Create a dashboard showing theme prevalence by business type with correct file names.
        """
        # Updated file paths for theme data
        theme_data = {}
        
        # Restaurant themes
        if ('restaurant_insights' in self.data and 
            'themes_by_trip_type.csv' in self.data['restaurant_insights']):
            theme_data['Restaurant'] = self.data['restaurant_insights']['themes_by_trip_type.csv']
        
        # Accommodation themes
        if ('accommodation_insights' in self.data and 
            'trip_purpose_preferences.csv' in self.data['accommodation_insights']):
            theme_data['Accommodation'] = self.data['accommodation_insights']['trip_purpose_preferences.csv']
        
        # Activity themes
        if ('activity_insights' in self.data and 
            'segment_activity_preferences.csv' in self.data['activity_insights']):
            theme_data['Activity'] = self.data['activity_insights']['segment_activity_preferences.csv']

        # Create visualization
        fig = go.Figure()
        
        for business_type, df in theme_data.items():
            try:
                # Extract theme columns (assuming they don't include 'trip_type' or similar metadata columns)
                theme_cols = [col for col in df.columns if not any(x in col.lower() 
                    for x in ['trip', 'type', 'standard', 'segment', 'purpose'])]
                
                for theme in theme_cols:
                    fig.add_trace(go.Bar(
                        name=f"{business_type}: {theme}",
                        x=[business_type],
                        y=[df[theme].mean()],
                        text=[f"{df[theme].mean():.2f}"],
                        textposition='auto'
                    ))
            except Exception as e:
                print(f"Error creating theme prevalence for {business_type}: {str(e)}")
        
        fig.update_layout(
            title="Theme Prevalence by Business Type",
            xaxis_title="Business Type",
            yaxis_title="Prevalence Score",
            barmode='group',
            height=600
        )
        
        return fig
    
    def create_expectation_gap_matrix(self):
        """
        Create an opportunity matrix highlighting gaps between expectations and experiences.
        """
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
            
            return fig
        else:
            # Create empty figure with instructions
            fig = go.Figure()
            fig.add_annotation(
                text="No expectation gap data available. Run visitor expectations analysis first.",
                showarrow=False,
                font=dict(size=20)
            )
            fig.update_layout(height=600)
            return fig
    
    def create_sentiment_map(self):
        """
        Create a sentiment map showing patterns of satisfaction.
        """
        # This would ideally use geographic data, but since we don't have that,
        # we'll create a heatmap of sentiment by visitor segment and aspect
        
        # Try to get sentiment data by traveler segment
        segment_sentiment = None
        aspect_sentiment = None
        
        if 'traveler_insights' in self.data and 'sentiment_by_segment.csv' in self.data['traveler_insights']:
            segment_sentiment = self.data['traveler_insights']['sentiment_by_segment.csv']
        
        # Try to get sentiment data by aspect
        if '' in self.data and 'sentiment_by_theme.csv' in self.data['']:
            aspect_sentiment = self.data['']['sentiment_by_theme.csv']
        
        # Create subplot with two heatmaps
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Sentiment by Visitor Segment", "Sentiment by Experience Aspect")
        )
        
        # Add segment sentiment heatmap
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
                print(f"Error creating segment sentiment heatmap: {str(e)}")
        
        # Add aspect sentiment heatmap
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
                print(f"Error creating aspect sentiment heatmap: {str(e)}")
        
        fig.update_layout(
            height=800,
            title_text="Sentiment Patterns in Tourism Experiences"
        )
        
        return fig
    
    def create_trend_analysis(self):
        """
        Create trend analysis showing emerging visitor interests.
        """
        # Try to get trend data
        trend_data = None
        
        if '' in self.data and 'theme_evolution.csv' in self.data['']:
            trend_data = self.data['']['theme_evolution.csv']
        
        if trend_data is not None:
            # Convert to long format if needed
            if 'publication_year' in trend_data.columns:
                # Data is already in the right format
                pass
            else:
                # Need to melt the data to long format
                id_vars = [col for col in trend_data.columns if not col.startswith('theme_')]
                value_vars = [col for col in trend_data.columns if col.startswith('theme_')]
                trend_data = pd.melt(trend_data, id_vars=id_vars, value_vars=value_vars, 
                                    var_name='theme', value_name='prevalence')
                trend_data['theme'] = trend_data['theme'].str.replace('theme_', '')
            
            # Create line chart
            fig = px.line(
                trend_data,
                x='publication_year',
                y='prevalence',
                color='theme',
                title="Evolution of Visitor Interests Over Time"
            )
            
            fig.update_layout(
                xaxis_title="Year",
                yaxis_title="Theme Prevalence",
                height=600
            )
            
            return fig
        else:
            # Create empty figure with instructions
            fig = go.Figure()
            fig.add_annotation(
                text="No trend data available. Run temporal analysis first.",
                showarrow=False,
                font=dict(size=20)
            )
            fig.update_layout(height=600)
            return fig
    
    def create_recommendation_dashboard(self):
        """
        Create a dashboard displaying key recommendations.
        """
        # Collect recommendations from all modules
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
        
        # Convert to DataFrame
        if all_recommendations:
            rec_df = pd.DataFrame(all_recommendations)
            
            # Create a table
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
            
            return fig
        else:
            # Create empty figure with instructions
            fig = go.Figure()
            fig.add_annotation(
                text="No recommendations available. Run analyses with recommendation generation.",
                showarrow=False,
                font=dict(size=20)
            )
            fig.update_layout(height=600)
            return fig
    
    def create_dashboard_app(self):
        """
        Create a Dash app with all dashboards.
        """
        # Load data if not already loaded
        if not self.data:
            self.load_analysis_data()
        
        # Initialize Dash app
        app = dash.Dash(__name__, suppress_callback_exceptions=True)
        self.app = app
        
        # Define app layout
        app.layout = html.Div([
            html.H1("Tonga Tourism Planning Dashboard", style={'textAlign': 'center'}),
            
            html.Div([
                html.Label("Select Dashboard:"),
                dcc.Dropdown(
                    id='dashboard-selector',
                    options=[
                        {'label': 'Theme Prevalence by Business Type', 'value': 'theme'},
                        {'label': 'Expectation-Experience Gap Matrix', 'value': 'expectation'},
                        {'label': 'Sentiment Patterns', 'value': 'sentiment'},
                        {'label': 'Visitor Interest Trends', 'value': 'trends'},
                        {'label': 'Key Recommendations', 'value': 'recommendations'}
                    ],
                    value='theme'
                )
            ], style={'width': '30%', 'margin': '0 auto', 'padding': '20px'}),
            
            dcc.Graph(id='main-dashboard', style={'height': '80vh'})
        ])
        
        # Define callback to update dashboard
        @app.callback(
            Output('main-dashboard', 'figure'),
            [Input('dashboard-selector', 'value')]
        )
        def update_dashboard(selected_dashboard):
            if selected_dashboard == 'theme':
                return self.create_theme_prevalence_dashboard()
            elif selected_dashboard == 'expectation':
                return self.create_expectation_gap_matrix()
            elif selected_dashboard == 'sentiment':
                return self.create_sentiment_map()
            elif selected_dashboard == 'trends':
                return self.create_trend_analysis()
            elif selected_dashboard == 'recommendations':
                return self.create_recommendation_dashboard()
            else:
                # Default
                return self.create_theme_prevalence_dashboard()
        
        return app
    
    def run_dashboard(self, debug=True, port=8050):
        """
        Run the dashboard app.
        
        Parameters:
        - debug: Whether to run in debug mode
        - port: Port to run the server on
        """
        if self.app is None:
            self.create_dashboard_app()
        
        print(f"Dashboard running at http://localhost:{port}/")
        self.app.run_server(debug=debug, port=port)


# Function to run the dashboard standalone
def run_tourism_dashboard(results_dir='tourism_analysis_results', port=8050):
    """
    Create and run the tourism planning dashboard.
    
    Parameters:
    - results_dir: Directory containing analysis results
    - port: Port to run the dashboard on
    """
    dashboard = TourismDashboard(results_dir)
    dashboard.run_dashboard(port=port)


# Run dashboard if script is executed directly
if __name__ == '__main__':
    run_tourism_dashboard()