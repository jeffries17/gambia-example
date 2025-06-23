#!/usr/bin/env python3
"""
Tourism Analysis API

Enhanced API for tourism board analysis with AI-powered insights generation.
"""

import os
import sys
import json
import logging
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Add parent directories to path for imports
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)

from analysis.tourism_insights_analyzer import analyze_tourism_insights
from reporting.tourism_report_generator import generate_tourism_report
from workflows.tourism_workflow import analyze_destination_from_url
from ai_insights_generator import generate_tourism_ai_insights

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
app.config['RESULTS_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

@app.route('/')
def index():
    """Main API documentation page."""
    return jsonify({
        'service': 'Tourism Analysis API',
        'version': '2.0',
        'endpoints': {
            '/analyze-reviews': 'POST - Analyze tourism reviews from JSON data',
            '/analyze-url': 'POST - Analyze TripAdvisor URL',
            '/generate-ai-insights': 'POST - Generate AI-powered actionable insights',
            '/insights': 'GET - List available analysis insights',
            '/reports': 'GET - List available reports'
        },
        'ai_features': {
            'powered_by': 'Google AI (Gemini)',
            'capabilities': [
                'Tourism management recommendations',
                'Digital reputation management',
                'Digital visibility strategies'
            ]
        }
    })

@app.route('/analyze-reviews', methods=['POST'])
def analyze_reviews():
    """Analyze tourism reviews from uploaded JSON data."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Handle both raw review arrays and structured data
        if isinstance(data, list):
            # Raw review array
            reviews_data = data
            destination_name = 'Unknown Destination'
        elif 'reviews' in data:
            # Structured data with reviews array
            reviews_data = data['reviews']
            destination_name = data.get('destination', 'Unknown Destination')
        else:
            return jsonify({'error': 'Invalid data format. Expected reviews array or structured data with reviews field.'}), 400
        
        if not reviews_data or len(reviews_data) == 0:
            return jsonify({'error': 'No reviews found in provided data'}), 400
        
        logger.info(f"Analyzing {len(reviews_data)} reviews for {destination_name}")
        
        # Create TripAdvisor-compatible format
        tripadvisor_data = {
            'business_info': {
                'name': destination_name,
                'category': 'Tourism',
                'location': data.get('location', 'Unknown'),
                'rating': None,
                'num_reviews': len(reviews_data)
            },
            'reviews': reviews_data,
            'extraction_metadata': {
                'url': 'uploaded_data',
                'extraction_date': datetime.now().isoformat(),
                'total_reviews': len(reviews_data)
            }
        }
        
        # Run tourism insights analysis
        insights = analyze_tourism_insights(tripadvisor_data, output_dir=f'{app.config["RESULTS_FOLDER"]}/insights')
        
        # Generate basic report
        report_path = generate_tourism_report(insights, report_type='executive', output_dir=f'{app.config["RESULTS_FOLDER"]}/reports')
        
        # Prepare response
        response = {
            'status': 'success',
            'analysis_metadata': {
                'destination': destination_name,
                'total_reviews': len(reviews_data),
                'analysis_date': datetime.now().isoformat()
            },
            'overall_sentiment': insights.get('overall_sentiment', {}),
            'aspect_performance': insights.get('aspect_sentiment', {}),
            'key_themes': insights.get('recurring_themes', {}),
            'executive_summary': insights.get('executive_summary', {}),
            'files_generated': [report_path] if report_path else []
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in analyze_reviews: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze-url', methods=['POST'])
def analyze_url():
    """Analyze TripAdvisor URL with tourism insights."""
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        destination_name = data.get('destination_name', '').strip()
        max_reviews = int(data.get('max_reviews', 50))
        include_reports = data.get('include_reports', True)
        
        if not url:
            return jsonify({'error': 'URL is required'}), 400
        
        if 'tripadvisor.com' not in url:
            return jsonify({'error': 'Please provide a valid TripAdvisor URL'}), 400
        
        logger.info(f"Starting tourism analysis for: {url}")
        
        # Run complete tourism analysis workflow
        workflow_results = analyze_destination_from_url(
            url=url,
            destination_name=destination_name,
            max_reviews=max_reviews
        )
        
        if 'error' in workflow_results:
            return jsonify({'error': workflow_results['error']}), 500
        
        # Extract key information for response
        insights = workflow_results.get('insights', {})
        analysis_summary = workflow_results.get('analysis_summary', {})
        
        response = {
            'status': 'success',
            'extraction_metadata': workflow_results.get('workflow_metadata', {}),
            'analysis_summary': analysis_summary,
            'overall_sentiment': insights.get('overall_sentiment', {}),
            'aspect_performance': insights.get('aspect_sentiment', {}),
            'language_diversity': insights.get('language_analysis', {}),
            'executive_summary': insights.get('executive_summary', {}),
            'files_created': workflow_results.get('files_created', [])
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in analyze_url: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/generate-ai-insights', methods=['POST'])
def generate_ai_insights():
    """Generate AI-powered actionable insights for tourism management."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Handle different input formats
        if 'analysis_data' in data:
            # Direct analysis data provided
            analysis_data = data['analysis_data']
            destination_name = data.get('destination_name', 'Tourism Destination')
        elif 'reviews' in data:
            # Raw reviews provided - run analysis first
            reviews_data = data['reviews']
            destination_name = data.get('destination_name', 'Tourism Destination')
            
            # Create TripAdvisor-compatible format and analyze
            tripadvisor_data = {
                'business_info': {
                    'name': destination_name,
                    'category': 'Tourism',
                    'location': data.get('location', 'Unknown'),
                    'rating': None,
                    'num_reviews': len(reviews_data)
                },
                'reviews': reviews_data,
                'extraction_metadata': {
                    'url': 'api_request',
                    'extraction_date': datetime.now().isoformat(),
                    'total_reviews': len(reviews_data)
                }
            }
            
            # Run analysis
            analysis_data = analyze_tourism_insights(tripadvisor_data, output_dir=f'{app.config["RESULTS_FOLDER"]}/temp_insights')
        else:
            return jsonify({'error': 'Either analysis_data or reviews must be provided'}), 400
        
        # Extract API key if provided, otherwise use default
        api_key = data.get('api_key') or os.getenv('GOOGLE_API_KEY')
        if not api_key:
            return jsonify({
                'error': 'Google API key not provided. Please include api_key in request or set GOOGLE_API_KEY environment variable.'
            }), 400
        
        logger.info(f"Generating AI insights for {destination_name}")
        
        # Generate AI insights
        ai_insights = generate_tourism_ai_insights(
            analysis_data=analysis_data,
            destination_name=destination_name,
            api_key=api_key,
            output_dir=f'{app.config["RESULTS_FOLDER"]}/ai_insights'
        )
        
        # Prepare response
        response = {
            'status': 'success',
            'destination': destination_name,
            'ai_insights': ai_insights,
            'generation_metadata': ai_insights.get('generation_metadata', {}),
            'summary': ai_insights.get('summary', ''),
            'actionable_insights': ai_insights.get('actionable_insights', [])
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in generate_ai_insights: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/insights')
def list_insights():
    """List available tourism insights files."""
    try:
        insights_files = []
        insights_dir = os.path.join(app.config['RESULTS_FOLDER'], 'insights')
        
        if os.path.exists(insights_dir):
            for filename in os.listdir(insights_dir):
                if filename.endswith('.json') and 'tourism_insights' in filename:
                    file_path = os.path.join(insights_dir, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            metadata = data.get('analysis_metadata', {})
                            
                            insights_files.append({
                                'filename': filename,
                                'destination': metadata.get('destination', 'Unknown'),
                                'category': metadata.get('category', 'Unknown'),
                                'total_reviews': metadata.get('total_reviews', 0),
                                'analysis_date': metadata.get('analysis_date', 'Unknown')[:10],
                                'overall_sentiment': data.get('overall_sentiment', {}).get('overall_score', 0)
                            })
                    except:
                        continue
        
        return jsonify({
            'status': 'success',
            'insights_count': len(insights_files),
            'insights_files': insights_files
        })
        
    except Exception as e:
        logger.error(f"Error listing insights: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/reports')
def list_reports():
    """List available tourism reports."""
    try:
        reports = []
        reports_dir = os.path.join(app.config['RESULTS_FOLDER'], 'reports')
        
        if os.path.exists(reports_dir):
            for filename in os.listdir(reports_dir):
                if filename.endswith('.md'):
                    file_path = os.path.join(reports_dir, filename)
                    file_stats = os.stat(file_path)
                    
                    reports.append({
                        'filename': filename,
                        'size_kb': round(file_stats.st_size / 1024, 1),
                        'created_date': datetime.fromtimestamp(file_stats.st_mtime).isoformat()[:10]
                    })
        
        return jsonify({
            'status': 'success',
            'reports_count': len(reports),
            'reports': reports
        })
        
    except Exception as e:
        logger.error(f"Error listing reports: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """API health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'Tourism Analysis API',
        'version': '2.0',
        'ai_available': True  # Always true since we have fallback
    })

if __name__ == '__main__':
    # Create output directories
    for directory in ['insights', 'reports', 'ai_insights', 'temp_insights']:
        os.makedirs(os.path.join(app.config['RESULTS_FOLDER'], directory), exist_ok=True)
    
    print("🚀 Tourism Analysis API v2.0 Starting...")
    print("📊 Endpoints available:")
    print("   POST /analyze-reviews - Analyze tourism review data")
    print("   POST /analyze-url - Analyze TripAdvisor URLs")
    print("   POST /generate-ai-insights - Generate AI-powered insights")
    print("   GET /insights - List available insights")
    print("   GET /reports - List available reports")
    print("   GET /health - Health check")
    print("🤖 AI Features: Google AI (Gemini) integration for actionable insights")
    
    app.run(debug=True, host='0.0.0.0', port=5000) 