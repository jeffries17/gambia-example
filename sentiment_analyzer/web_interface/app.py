#!/usr/bin/env python3
"""
Tourism-Focused Web Interface for TripAdvisor Sentiment Analysis

Enhanced web interface that provides tourism board insights, professional reports,
and comprehensive destination analysis.
"""

import os
import json
import uuid
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from werkzeug.utils import secure_filename
import logging

# Add parent directories to path for imports
import sys
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)

from workflows.tourism_workflow import analyze_destination_from_url, compare_destinations_from_urls
from analysis.tourism_insights_analyzer import analyze_tourism_insights
from reporting.tourism_report_generator import generate_tourism_report

app = Flask(__name__)
app.config['SECRET_KEY'] = 'tourism-analysis-secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'outputs'

# Create necessary directories
for folder in [app.config['UPLOAD_FOLDER'], app.config['RESULTS_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/')
def index():
    """Tourism-focused main page."""
    return render_template('tourism_index.html')

@app.route('/extract')
def extract_page():
    """Enhanced extraction page with tourism focus."""
    return render_template('tourism_extract.html')

@app.route('/insights')
def insights_page():
    """Tourism insights dashboard."""
    # Get available analysis files
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
    
    return render_template('tourism_insights.html', insights_files=insights_files)

@app.route('/reports')
def reports_page():
    """Tourism reports page."""
    report_files = []
    reports_dir = os.path.join(app.config['RESULTS_FOLDER'], 'reports')
    
    if os.path.exists(reports_dir):
        for filename in os.listdir(reports_dir):
            if filename.endswith('.md'):
                file_path = os.path.join(reports_dir, filename)
                
                # Parse report type and destination from filename
                report_type = 'Unknown'
                if 'executive_report' in filename:
                    report_type = 'Executive Summary'
                elif 'detailed_analysis' in filename:
                    report_type = 'Detailed Analysis'
                elif 'comparison_report' in filename:
                    report_type = 'Destination Comparison'
                
                # Extract date from filename
                date_str = 'Unknown'
                try:
                    date_parts = filename.split('_')[-1].replace('.md', '')
                    if len(date_parts) == 8:  # YYYYMMDD format
                        date_str = f"{date_parts[:4]}-{date_parts[4:6]}-{date_parts[6:8]}"
                except:
                    pass
                
                report_files.append({
                    'filename': filename,
                    'report_type': report_type,
                    'date': date_str,
                    'file_size': os.path.getsize(file_path)
                })
    
    return render_template('tourism_reports.html', report_files=report_files)

@app.route('/api/tourism-extract', methods=['POST'])
def api_tourism_extract():
    """Enhanced API endpoint with tourism insights analysis."""
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
        
        return jsonify({
            'success': True,
            'destination': insights.get('analysis_metadata', {}).get('destination', 'Unknown'),
            'category': insights.get('analysis_metadata', {}).get('category', 'Unknown'),
            'location': insights.get('analysis_metadata', {}).get('location', 'Unknown'),
            'total_reviews': analysis_summary.get('total_reviews', 0),
            'overall_sentiment': analysis_summary.get('overall_sentiment', 0),
            'confidence_level': insights.get('overall_sentiment', {}).get('confidence_level', 'Unknown'),
            'key_insights': workflow_results.get('summary', {}).get('top_insights', []),
            'files_created': len(workflow_results.get('files_created', [])),
            'analysis_quality': workflow_results.get('summary', {}).get('analysis_quality', 'Unknown'),
            'next_steps': workflow_results.get('summary', {}).get('next_steps', [])[:3]
        })
        
    except Exception as e:
        logger.error(f"Tourism extraction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/tourism-compare', methods=['POST'])
def api_tourism_compare():
    """Enhanced comparison API with tourism insights."""
    try:
        data = request.get_json()
        destinations = data.get('destinations', [])
        
        if len(destinations) < 2:
            return jsonify({'error': 'Please provide at least 2 destinations for comparison'}), 400
        
        logger.info(f"Comparing {len(destinations)} destinations for tourism insights")
        
        # Prepare destinations list
        dest_configs = []
        for dest in destinations:
            if 'url' not in dest:
                return jsonify({'error': 'Each destination must have a URL'}), 400
            
            dest_configs.append((
                dest['url'], 
                dest.get('name', f"Destination_{len(dest_configs)+1}")
            ))
        
        # Run comparison workflow
        comparison_results = compare_destinations_from_urls(
            dest_configs, 
            max_reviews_each=data.get('max_reviews_each', 50)
        )
        
        if 'error' in comparison_results:
            return jsonify({'error': comparison_results['error']}), 500
        
        # Extract summary information
        comparison_summary = comparison_results.get('comparison_summary', {})
        
        return jsonify({
            'success': True,
            'total_destinations': comparison_summary.get('total_destinations', 0),
            'total_reviews_analyzed': comparison_summary.get('total_reviews_analyzed', 0),
            'best_performer': comparison_summary.get('best_performer', ['Unknown', {}])[0] if comparison_summary.get('best_performer') else 'Unknown',
            'average_sentiment': comparison_summary.get('average_sentiment', 0),
            'key_differentiators': comparison_summary.get('key_differentiators', []),
            'competitive_insights': comparison_summary.get('competitive_insights', []),
            'files_created': len(comparison_results.get('files_created', []))
        })
        
    except Exception as e:
        logger.error(f"Tourism comparison error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/insights/<filename>')
def view_insights(filename):
    """View tourism insights file."""
    safe_filename = secure_filename(filename)
    filepath = os.path.join(app.config['RESULTS_FOLDER'], 'insights', safe_filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'Insights file not found'}), 404
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            insights_data = json.load(f)
        
        # Format for web display
        formatted_insights = {
            'metadata': insights_data.get('analysis_metadata', {}),
            'overall_sentiment': insights_data.get('overall_sentiment', {}),
            'top_aspects': {
                k: v for k, v in insights_data.get('aspect_sentiment', {}).items()
                if v.get('mention_percentage', 0) > 5  # Only show aspects mentioned in >5% of reviews
            },
            'key_themes': {
                k: v for k, v in insights_data.get('recurring_themes', {}).items()
                if v.get('percentage', 0) > 10  # Only show themes mentioned in >10% of reviews
            },
            'language_diversity': insights_data.get('language_analysis', {}),
            'executive_summary': insights_data.get('executive_summary', {}),
            'responsiveness': insights_data.get('responsiveness_analysis', {})
        }
        
        return jsonify(formatted_insights)
        
    except Exception as e:
        return jsonify({'error': f'Error reading insights file: {str(e)}'}), 500

@app.route('/api/report/<filename>')
def view_report(filename):
    """View or download a report file."""
    safe_filename = secure_filename(filename)
    filepath = os.path.join(app.config['RESULTS_FOLDER'], 'reports', safe_filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'Report file not found'}), 404
    
    # Determine if this is a download or view request
    view_mode = request.args.get('view', 'false').lower() == 'true'
    
    if view_mode and filename.endswith('.md'):
        # Return markdown content for viewing
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            return jsonify({'content': content, 'type': 'markdown'})
        except Exception as e:
            return jsonify({'error': f'Error reading report: {str(e)}'}), 500
    else:
        # Download the file
        return send_file(filepath, as_attachment=True)

@app.route('/api/download/<path:filepath>')
def download_analysis_file(filepath):
    """Download any analysis file."""
    # Security check - only allow files in results directory
    safe_path = os.path.join(app.config['RESULTS_FOLDER'], filepath)
    safe_path = os.path.abspath(safe_path)
    results_path = os.path.abspath(app.config['RESULTS_FOLDER'])
    
    if not safe_path.startswith(results_path):
        return jsonify({'error': 'Invalid file path'}), 400
    
    if not os.path.exists(safe_path):
        return jsonify({'error': 'File not found'}), 404
    
    return send_file(safe_path, as_attachment=True)

@app.route('/api/gambia-demo', methods=['POST'])
def gambia_demo():
    """Special endpoint for Gambia tourism demonstration."""
    try:
        data = request.get_json()
        demo_urls = data.get('urls', [])
        
        if not demo_urls:
            return jsonify({'error': 'Please provide at least one Gambia TripAdvisor URL'}), 400
        
        logger.info(f"Running Gambia tourism demo with {len(demo_urls)} URLs")
        
        # Quick analysis for demonstration
        if len(demo_urls) == 1:
            # Single destination analysis
            result = analyze_destination_from_url(demo_urls[0], "Gambia_Demo", max_reviews=25)
        else:
            # Multi-destination comparison
            dest_configs = [(url, f"Gambia_Destination_{i+1}") for i, url in enumerate(demo_urls)]
            result = compare_destinations_from_urls(dest_configs, max_reviews_each=20)
        
        return jsonify({
            'success': True,
            'demo_type': 'single' if len(demo_urls) == 1 else 'comparison',
            'files_created': len(result.get('files_created', [])),
            'summary': result.get('summary', {}) if len(demo_urls) == 1 else result.get('comparison_summary', {}),
            'message': 'Gambia tourism analysis demo completed successfully'
        })
        
    except Exception as e:
        logger.error(f"Gambia demo error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create enhanced templates for tourism focus
    create_tourism_templates()
    
    print("Tourism-Focused TripAdvisor Sentiment Analysis System")
    print("===================================================")
    print("Features:")
    print("- Tourism board insights analysis")
    print("- Professional report generation") 
    print("- Destination comparison")
    print("- Language diversity analysis")
    print("- Service responsiveness metrics")
    print("")
    print("Navigate to http://localhost:5000 to start analysis")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

def create_tourism_templates():
    """Create tourism-focused HTML templates."""
    templates_dir = 'templates'
    os.makedirs(templates_dir, exist_ok=True)
    
    # Enhanced tourism index template
    tourism_index = '''{% extends "base.html" %}

{% block title %}Tourism Board Sentiment Analysis{% endblock %}

{% block content %}
<div class="row mb-4">
    <div class="col-12">
        <div class="jumbotron bg-primary text-white p-4 rounded">
            <h1 class="display-4">Tourism Sentiment Analysis</h1>
            <p class="lead">Professional destination analysis for tourism boards and destination marketing organizations</p>
            <p>Extract insights from TripAdvisor reviews to improve visitor experience and competitive positioning</p>
        </div>
    </div>
</div>

<div class="row">
    <div class="col-md-4">
        <div class="card h-100">
            <div class="card-body">
                <h5 class="card-title">🎯 Extract & Analyze</h5>
                <p class="card-text">Paste TripAdvisor URLs to get comprehensive tourism insights including sentiment analysis, aspect-based reviews, and language diversity.</p>
                <ul class="small">
                    <li>Overall sentiment scoring</li>
                    <li>Aspect-based analysis</li>
                    <li>Keyword extraction</li>
                    <li>Theme identification</li>
                </ul>
                <a href="{{ url_for('extract_page') }}" class="btn btn-primary">Start Analysis</a>
            </div>
        </div>
    </div>
    
    <div class="col-md-4">
        <div class="card h-100">
            <div class="card-body">
                <h5 class="card-title">📊 Tourism Insights</h5>
                <p class="card-text">View detailed tourism board insights including international market analysis, service responsiveness, and strategic recommendations.</p>
                <ul class="small">
                    <li>Language diversity metrics</li>
                    <li>Service responsiveness scores</li>
                    <li>Recurring theme analysis</li>
                    <li>Executive summaries</li>
                </ul>
                <a href="{{ url_for('insights_page') }}" class="btn btn-success">View Insights</a>
            </div>
        </div>
    </div>
    
    <div class="col-md-4">
        <div class="card h-100">
            <div class="card-body">
                <h5 class="card-title">📋 Professional Reports</h5>
                <p class="card-text">Download professional reports formatted for tourism boards, including executive summaries and detailed analysis documents.</p>
                <ul class="small">
                    <li>Executive summary reports</li>
                    <li>Detailed analysis documents</li>
                    <li>Destination comparison reports</li>
                    <li>Strategic recommendations</li>
                </ul>
                <a href="{{ url_for('reports_page') }}" class="btn btn-info">Download Reports</a>
            </div>
        </div>
    </div>
</div>

<div class="row mt-5">
    <div class="col-12">
        <h3>Key Features for Tourism Boards</h3>
        <div class="row">
            <div class="col-md-6">
                <h5>📈 Sentiment Analysis</h5>
                <ul>
                    <li><strong>Overall Sentiment Score:</strong> Single metric for visitor satisfaction</li>
                    <li><strong>Aspect-Based Scores:</strong> Detailed breakdown by accommodation, restaurants, attractions</li>
                    <li><strong>Confidence Levels:</strong> Statistical reliability indicators</li>
                </ul>
            </div>
            <div class="col-md-6">
                <h5>🌍 International Market Analysis</h5>
                <ul>
                    <li><strong>Language Diversity:</strong> Assessment of international visitor base</li>
                    <li><strong>Market Diversification:</strong> Insights into visitor nationality spread</li>
                    <li><strong>Growth Opportunities:</strong> Identification of underrepresented markets</li>
                </ul>
            </div>
        </div>
        
        <div class="row mt-3">
            <div class="col-md-6">
                <h5>🏆 Competitive Intelligence</h5>
                <ul>
                    <li><strong>Benchmarking:</strong> Compare with similar destinations</li>
                    <li><strong>Positioning:</strong> Identify competitive advantages</li>
                    <li><strong>Best Practices:</strong> Learn from top performers</li>
                </ul>
            </div>
            <div class="col-md-6">
                <h5>💼 Strategic Recommendations</h5>
                <ul>
                    <li><strong>Actionable Insights:</strong> Specific improvement recommendations</li>
                    <li><strong>Priority Areas:</strong> Focus areas for maximum impact</li>
                    <li><strong>Marketing Strategies:</strong> Leverage strengths in promotion</li>
                </ul>
            </div>
        </div>
    </div>
</div>

<div class="row mt-5">
    <div class="col-12">
        <div class="alert alert-info" role="alert">
            <h4 class="alert-heading">🇬🇲 Gambia Tourism Demo</h4>
            <p>Ready to analyze Gambia's tourism sentiment? Use the analysis tools above with Gambia TripAdvisor URLs to:</p>
            <ul class="mb-0">
                <li>Assess visitor satisfaction for key destinations</li>
                <li>Identify strengths in cultural tourism and hospitality</li>
                <li>Compare with regional competitors (Senegal, Ghana)</li>
                <li>Generate insights for tourism marketing strategy</li>
            </ul>
        </div>
    </div>
</div>
{% endblock %}'''
    
    # Enhanced extraction template
    tourism_extract = '''{% extends "base.html" %}

{% block title %}Tourism Analysis - Extract Reviews{% endblock %}

{% block content %}
<div class="row">
    <div class="col-12">
        <h2>Tourism Destination Analysis</h2>
        <p class="lead">Extract and analyze TripAdvisor reviews for comprehensive tourism insights</p>
    </div>
</div>

<div class="row">
    <div class="col-md-8">
        <div class="card">
            <div class="card-body">
                <form id="tourismExtractForm">
                    <div class="mb-3">
                        <label for="url" class="form-label">TripAdvisor URL</label>
                        <input type="url" class="form-control" id="url" required 
                               placeholder="https://www.tripadvisor.com/...">
                        <div class="form-text">Paste URL for hotels, restaurants, or attractions</div>
                    </div>
                    
                    <div class="mb-3">
                        <label for="destinationName" class="form-label">Destination Name (Optional)</label>
                        <input type="text" class="form-control" id="destinationName" 
                               placeholder="e.g., Kunta Kinteh Island, Banjul Hotel">
                        <div class="form-text">Custom name for reports and analysis</div>
                    </div>
                    
                    <div class="mb-3">
                        <label for="maxReviews" class="form-label">Analysis Depth</label>
                        <select class="form-control" id="maxReviews">
                            <option value="25">Quick Analysis (25 reviews)</option>
                            <option value="50" selected>Standard Analysis (50 reviews)</option>
                            <option value="100">Comprehensive Analysis (100 reviews)</option>
                            <option value="200">Deep Analysis (200 reviews)</option>
                        </select>
                        <div class="form-text">More reviews = higher confidence, longer processing time</div>
                    </div>
                    
                    <div class="mb-3">
                        <div class="form-check">
                            <input class="form-check-input" type="checkbox" value="" id="includeReports" checked>
                            <label class="form-check-label" for="includeReports">
                                Generate professional reports
                            </label>
                        </div>
                    </div>
                    
                    <button type="submit" class="btn btn-primary btn-lg">
                        <span id="submitText">🚀 Start Tourism Analysis</span>
                        <span id="loadingSpinner" class="spinner-border spinner-border-sm d-none" role="status"></span>
                    </button>
                </form>
            </div>
        </div>
    </div>
    
    <div class="col-md-4">
        <div class="card">
            <div class="card-header">
                <h6 class="card-title mb-0">Analysis Includes</h6>
            </div>
            <div class="card-body">
                <ul class="small mb-3">
                    <li><strong>Overall Sentiment Score</strong> - Visitor satisfaction metric</li>
                    <li><strong>Aspect Analysis</strong> - Accommodation, dining, attractions</li>
                    <li><strong>Keyword Extraction</strong> - Most mentioned positive/negative terms</li>
                    <li><strong>Theme Identification</strong> - Recurring visitor narratives</li>
                    <li><strong>Language Diversity</strong> - International market assessment</li>
                    <li><strong>Service Responsiveness</strong> - Staff and management engagement</li>
                </ul>
                
                <h6>Report Outputs</h6>
                <ul class="small">
                    <li>Executive Summary (Tourism Board ready)</li>
                    <li>Detailed Analysis Report</li>
                    <li>Raw data (JSON format)</li>
                    <li>Strategic recommendations</li>
                </ul>
            </div>
        </div>
        
        <div class="card mt-3">
            <div class="card-header">
                <h6 class="card-title mb-0">Privacy & Ethics</h6>
            </div>
            <div class="card-body small">
                <p>✓ Reviewer names anonymized to initials<br>
                ✓ Public reviews only<br>
                ✓ Tourism research purpose<br>
                ✓ Rate-limited extraction</p>
            </div>
        </div>
    </div>
</div>

<div class="row mt-4 d-none" id="resultsSection">
    <div class="col-12">
        <div class="alert alert-success" role="alert" id="successAlert"></div>
        <div class="alert alert-danger" role="alert" id="errorAlert"></div>
    </div>
</div>

<script>
$(document).ready(function() {
    $('#tourismExtractForm').on('submit', function(e) {
        e.preventDefault();
        
        const url = $('#url').val();
        const destinationName = $('#destinationName').val();
        const maxReviews = $('#maxReviews').val();
        const includeReports = $('#includeReports').is(':checked');
        
        // Show loading state
        $('#submitText').html('<span class="spinner-border spinner-border-sm" role="status"></span> Analyzing...');
        $('button[type="submit"]').prop('disabled', true);
        
        // Make API call
        $.ajax({
            url: '/api/tourism-extract',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                url: url,
                destination_name: destinationName,
                max_reviews: parseInt(maxReviews),
                include_reports: includeReports
            }),
            success: function(response) {
                if (response.success) {
                    let resultsHtml = `
                        <h4>🎉 Tourism Analysis Complete!</h4>
                        <div class="row">
                            <div class="col-md-6">
                                <h6>Destination Overview</h6>
                                <ul>
                                    <li><strong>Name:</strong> ${response.destination}</li>
                                    <li><strong>Category:</strong> ${response.category}</li>
                                    <li><strong>Location:</strong> ${response.location}</li>
                                    <li><strong>Reviews Analyzed:</strong> ${response.total_reviews}</li>
                                </ul>
                            </div>
                            <div class="col-md-6">
                                <h6>Key Metrics</h6>
                                <ul>
                                    <li><strong>Overall Sentiment:</strong> ${response.overall_sentiment.toFixed(2)} (${response.overall_sentiment > 0.3 ? 'Positive' : response.overall_sentiment < 0 ? 'Negative' : 'Neutral'})</li>
                                    <li><strong>Confidence:</strong> ${response.confidence_level}</li>
                                    <li><strong>Analysis Quality:</strong> ${response.analysis_quality}</li>
                                </ul>
                            </div>
                        </div>
                        
                        <h6>Key Insights</h6>
                        <ul>
                    `;
                    
                    response.key_insights.forEach(insight => {
                        resultsHtml += `<li>${insight}</li>`;
                    });
                    
                    resultsHtml += `
                        </ul>
                        
                        <h6>Recommended Next Steps</h6>
                        <ol>
                    `;
                    
                    response.next_steps.forEach(step => {
                        resultsHtml += `<li>${step}</li>`;
                    });
                    
                    resultsHtml += `
                        </ol>
                        
                        <div class="mt-3">
                            <a href="/insights" class="btn btn-success">📊 View Detailed Insights</a>
                            <a href="/reports" class="btn btn-info">📋 Download Reports</a>
                        </div>
                    `;
                    
                    $('#successAlert').html(resultsHtml).show();
                    $('#errorAlert').hide();
                    $('#resultsSection').removeClass('d-none');
                }
            },
            error: function(xhr) {
                const error = xhr.responseJSON ? xhr.responseJSON.error : 'Unknown error occurred';
                $('#errorAlert').text('Analysis Error: ' + error).show();
                $('#successAlert').hide();
                $('#resultsSection').removeClass('d-none');
            },
            complete: function() {
                // Reset loading state
                $('#submitText').text('🚀 Start Tourism Analysis');
                $('button[type="submit"]').prop('disabled', false);
            }
        });
    });
});
</script>
{% endblock %}'''
    
    # Write templates
    template_files = {
        'tourism_index.html': tourism_index,
        'tourism_extract.html': tourism_extract
    }
    
    for filename, content in template_files.items():
        filepath = os.path.join(templates_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ Created tourism template: {filename}") 