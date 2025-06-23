#!/usr/bin/env python3
"""
Simple Test Endpoint for AI Insights Generation

Test script to demonstrate the AI insights generation endpoint functionality
with modern visualizations.
"""

import json
import google.generativeai as genai
from datetime import datetime
from flask import Flask, request, jsonify
try:
    from flask_cors import CORS
    HAS_CORS = True
except ImportError:
    HAS_CORS = False
    print("Warning: flask_cors not available, CORS not enabled")

app = Flask(__name__)
if HAS_CORS:
    CORS(app)

@app.route('/')
def index():
    """API information."""
    return jsonify({
        'service': 'Gambia Tourism AI Insights API',
        'version': '2.0',
        'endpoints': {
            '/generate-insights': 'POST - Generate AI insights from analysis data',
            '/test-gambia': 'GET - Test with pre-loaded Gambia data',
            '/modern-visuals': 'GET - Generate modern visualizations'
        },
        'ai_powered_by': 'Google AI (Gemini-1.5-Flash)',
        'features': {
            'modern_visualizations': True,
            'interactive_dashboards': True,
            'professional_styling': True,
            'high_dpi_graphics': True
        }
    })

@app.route('/generate-insights', methods=['POST'])
def generate_insights():
    """Generate AI insights from tourism analysis data."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Get parameters
        analysis_data = data.get('analysis_data')
        destination_name = data.get('destination_name', 'Tourism Destination')
        api_key = data.get('api_key', 'AIzaSyB5lhObzZf6WPaTUnIw5aQiHAsJ0shyC-o')
        include_visuals = data.get('include_visuals', True)
        
        if not analysis_data:
            return jsonify({'error': 'analysis_data is required'}), 400
        
        # Generate insights using Google AI
        insights = generate_ai_insights(analysis_data, destination_name, api_key)
        
        # Generate modern visualizations if requested
        visual_paths = {}
        if include_visuals:
            try:
                import sys
                sys.path.append('.')
                from utils.modern_visualizer import create_modern_tourism_visuals
                
                visual_paths = create_modern_tourism_visuals(
                    analysis_data, destination_name, 
                    'outputs/modern_visualizations'
                )
            except Exception as e:
                visual_paths = {'error': f'Visualization generation failed: {str(e)}'}
        
        return jsonify({
            'status': 'success',
            'destination': destination_name,
            'insights': insights,
            'visualizations': visual_paths,
            'generated_at': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test-gambia', methods=['GET'])
def test_gambia():
    """Test endpoint with pre-loaded Gambia data."""
    try:
        # Load the Gambia analysis data
        with open('outputs/gambia_insights/tourism_insights_Gambia Tourism Destinations_20250623_153533.json', 'r') as f:
            analysis_data = json.load(f)
        
        # Generate insights
        insights = generate_ai_insights(analysis_data, 'Gambia (Kunta Kinteh Island)')
        
        return jsonify({
            'status': 'success',
            'destination': 'Gambia (Kunta Kinteh Island)',
            'insights': insights,
            'visualizations': {
                'modern_dashboard': 'outputs/gambia_insights/modern_visualizations/modern_dashboard_gambia.png',
                'interactive_dashboard': 'outputs/gambia_insights/modern_visualizations/interactive_dashboard_gambia.html',
                'wordclouds': {
                    'cultural_heritage': 'outputs/gambia_insights/modern_visualizations/modern_wordcloud_cultural_heritage.png',
                    'infrastructure': 'outputs/gambia_insights/modern_visualizations/modern_wordcloud_infrastructure.png',
                    'service_tourism': 'outputs/gambia_insights/modern_visualizations/modern_wordcloud_service_tourism.png'
                },
                'comparison': 'outputs/gambia_insights/modern_visualizations/before_after_comparison.png'
            },
            'source': 'Pre-loaded Gambia tourism analysis with modern visualizations',
            'generated_at': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/modern-visuals', methods=['GET'])
def generate_modern_visuals():
    """Generate modern visualizations for existing analysis."""
    try:
        # Load the Gambia analysis data
        with open('outputs/gambia_insights/tourism_insights_Gambia Tourism Destinations_20250623_153533.json', 'r') as f:
            analysis_data = json.load(f)
        
        # Generate modern visualizations
        import sys
        sys.path.append('.')
        from utils.modern_visualizer import ModernTourismVisualizer
        
        visualizer = ModernTourismVisualizer()
        
        # Create dashboard
        dashboard_path = visualizer.create_modern_dashboard(
            analysis_data,
            'Gambia (Kunta Kinteh Island)',
            'outputs/gambia_insights/modern_visualizations/api_generated_dashboard.png'
        )
        
        return jsonify({
            'status': 'success',
            'destination': 'Gambia (Kunta Kinteh Island)',
            'visualizations_generated': {
                'modern_dashboard': dashboard_path,
                'features': [
                    'Professional typography (SF Pro Display, Helvetica Neue)',
                    'Modern color palette with semantic meaning',
                    'High DPI (300 DPI) crisp output',
                    'Clean, minimal design',
                    'Proper white space usage',
                    'Consistent visual hierarchy'
                ]
            },
            'style_improvements': {
                'typography': 'Professional font families',
                'colors': 'Carefully selected modern palette',
                'layout': 'Clean, minimal design with proper spacing',
                'quality': '300 DPI for professional output',
                'accessibility': 'Color-blind friendly palette'
            },
            'generated_at': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def generate_ai_insights(analysis_data, destination_name, api_key='AIzaSyB5lhObzZf6WPaTUnIw5aQiHAsJ0shyC-o'):
    """Generate AI insights using Google AI."""
    try:
        # Configure Google AI
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Extract key data
        overall = analysis_data.get('overall_sentiment', {})
        aspects = analysis_data.get('aspect_sentiment', {})
        executive_summary = analysis_data.get('executive_summary', {})
        language_analysis = analysis_data.get('language_analysis', {})
        
        # Create prompt
        prompt = f"""You are a tourism management and digital reputation expert analyzing review data for {destination_name}.

ANALYSIS DATA:
• Total Reviews: {overall.get('total_reviews', 0)}
• Average Rating: {overall.get('average_rating', 0):.2f}/5
• Overall Sentiment: {overall.get('overall_score', 0):.3f}
• Positive Reviews: {overall.get('sentiment_distribution', {}).get('positive_percentage', 0):.1f}%
• International Appeal: {language_analysis.get('international_appeal', 0):.1f}% non-English reviews

CURRENT ISSUES:
{chr(10).join([f'• {item}' for item in executive_summary.get('areas_for_improvement', [])])}

STRENGTHS:
{chr(10).join([f'• {item}' for item in executive_summary.get('strengths', [])])}

Generate exactly 3 actionable insights:

INSIGHT 1: [TOURISM MANAGEMENT]
Title: [Action-oriented title]
Issue: [Specific problem from data]
Action: [Concrete, feasible steps]
Expected Impact: [Measurable outcomes]
Timeline: [Realistic timeframe]

INSIGHT 2: [DIGITAL REPUTATION MANAGEMENT]
Title: [Action-oriented title]
Issue: [Digital reputation problem]
Action: [Specific digital steps]
Expected Impact: [Reputation outcomes]
Timeline: [Implementation timeframe]

INSIGHT 3: [DIGITAL VISIBILITY & ENGAGEMENT]
Title: [Action-oriented title]
Issue: [Digital opportunity]
Action: [Concrete marketing steps]
Expected Impact: [Visibility outcomes]
Timeline: [Implementation timeframe]

Focus on government accountability, digital engagement, and feasible next steps."""
        
        # Generate content
        response = model.generate_content(prompt)
        
        # Parse response into structured format
        insights_text = response.text
        insights = parse_ai_insights(insights_text)
        
        return {
            'ai_response': insights_text,
            'structured_insights': insights,
            'model_used': 'gemini-1.5-flash',
            'prompt_summary': f'Generated tourism management insights for {destination_name}',
            'modern_features': {
                'professional_visuals': True,
                'interactive_charts': True,
                'high_dpi_output': True,
                'modern_typography': True
            }
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'fallback_insights': generate_fallback_insights(destination_name)
        }

def parse_ai_insights(text):
    """Parse AI response into structured insights."""
    insights = []
    
    # Split by INSIGHT markers
    sections = text.split('INSIGHT')
    
    for i, section in enumerate(sections[1:], 1):  # Skip first empty section
        lines = section.strip().split('\n')
        insight = {
            'insight_number': i,
            'category': '',
            'title': '',
            'issue': '',
            'action': '',
            'expected_impact': '',
            'timeline': ''
        }
        
        for line in lines:
            line = line.strip()
            if line.startswith('Title:'):
                insight['title'] = line.replace('Title:', '').strip()
            elif line.startswith('Issue:'):
                insight['issue'] = line.replace('Issue:', '').strip()
            elif line.startswith('Action:'):
                insight['action'] = line.replace('Action:', '').strip()
            elif line.startswith('Expected Impact:'):
                insight['expected_impact'] = line.replace('Expected Impact:', '').strip()
            elif line.startswith('Timeline:'):
                insight['timeline'] = line.replace('Timeline:', '').strip()
            elif '[' in line and ']' in line:
                insight['category'] = line.split('[')[1].split(']')[0]
        
        if insight['title'] and insight['action']:
            insights.append(insight)
    
    return insights

def generate_fallback_insights(destination_name):
    """Generate fallback insights when AI fails."""
    return [
        {
            'insight_number': 1,
            'category': 'Tourism Management',
            'title': 'Address Infrastructure Concerns with Modern Visual Monitoring',
            'issue': 'Reviews indicate infrastructure and maintenance issues',
            'action': 'Establish maintenance protocol with modern dashboard monitoring and secure funding for improvements',
            'expected_impact': 'Improved visitor satisfaction and site preservation with real-time tracking',
            'timeline': '3-6 months for immediate measures'
        },
        {
            'insight_number': 2,
            'category': 'Digital Reputation Management',
            'title': 'Implement Professional Review Response Strategy',
            'issue': 'Lack of management engagement with visitor feedback',
            'action': 'Create response team with modern analytics dashboard and commit to timely review responses',
            'expected_impact': 'Improved online reputation and visitor confidence with measurable metrics',
            'timeline': '2-4 weeks to implement'
        },
        {
            'insight_number': 3,
            'category': 'Digital Visibility & Engagement',
            'title': 'Launch Modern Digital Marketing Campaign',
            'issue': 'Opportunity to leverage international visitor interest',
            'action': 'Develop multilingual content with modern interactive visualizations and international marketing campaign',
            'expected_impact': 'Increased international visibility and visitor numbers with trackable ROI',
            'timeline': '6-12 months for full implementation'
        }
    ]

if __name__ == '__main__':
    print("🚀 Starting Modern Tourism AI Insights Server...")
    print("📊 Endpoints:")
    print("   GET  / - API information")
    print("   POST /generate-insights - Generate AI insights with modern visuals")
    print("   GET  /test-gambia - Test with Gambia data")
    print("   GET  /modern-visuals - Generate modern visualizations")
    print("🤖 Powered by Google AI (Gemini-1.5-Flash)")
    print("🎨 Features modern visualizations with:")
    print("   • Professional typography")
    print("   • Clean, modern design")
    print("   • High DPI (300 DPI) output")
    print("   • Interactive charts")
    print("   • Consistent branding")
    
    app.run(debug=True, host='0.0.0.0', port=5001) 