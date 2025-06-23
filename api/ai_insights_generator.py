#!/usr/bin/env python3
"""
AI Insights Generator for Tourism Management

This module uses Google AI to generate actionable insights for tourism management
and digital reputation management based on sentiment analysis and review data.
"""

import os
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime

try:
    import google.generativeai as genai
    HAS_GOOGLE_AI = True
except ImportError:
    HAS_GOOGLE_AI = False

logger = logging.getLogger(__name__)

class TourismAIInsightsGenerator:
    """
    Generates AI-powered actionable insights for tourism management using Google AI.
    """
    
    def __init__(self, api_key: str = None):
        if api_key is None:
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("Google API key not found. Please set GOOGLE_API_KEY environment variable.")
        """
        Initialize the AI insights generator.
        
        Args:
            api_key (str): Google AI API key
        """
        self.api_key = api_key
        self.model = None
        
        if HAS_GOOGLE_AI and api_key:
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-pro')
                logger.info("Google AI model initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Google AI: {str(e)}")
                self.model = None
        else:
            logger.warning("Google AI not available - install google-generativeai package")
    
    def generate_actionable_insights(self, analysis_data: Dict, destination_name: str = "Tourism Destination") -> Dict:
        """
        Generate 3 actionable insights for tourism management and digital reputation.
        
        Args:
            analysis_data (Dict): Complete analysis results from tourism insights analyzer
            destination_name (str): Name of the destination
            
        Returns:
            Dict: AI-generated actionable insights
        """
        if not self.model:
            return self._generate_fallback_insights(analysis_data, destination_name)
        
        try:
            # Prepare analysis summary for AI
            prompt = self._create_insights_prompt(analysis_data, destination_name)
            
            # Generate insights using Google AI
            response = self.model.generate_content(prompt)
            
            # Parse and structure the response
            insights = self._parse_ai_response(response.text)
            
            # Add metadata
            insights['generation_metadata'] = {
                'destination': destination_name,
                'generated_at': datetime.now().isoformat(),
                'model_used': 'gemini-pro',
                'insights_count': len(insights.get('actionable_insights', []))
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating AI insights: {str(e)}")
            return self._generate_fallback_insights(analysis_data, destination_name)
    
    def _create_insights_prompt(self, analysis_data: Dict, destination_name: str) -> str:
        """Create a comprehensive prompt for AI insights generation."""
        
        # Extract key metrics
        overall_sentiment = analysis_data.get('overall_sentiment', {})
        sentiment_score = overall_sentiment.get('overall_score', 0)
        positive_pct = overall_sentiment.get('sentiment_distribution', {}).get('positive_percentage', 0)
        negative_pct = overall_sentiment.get('sentiment_distribution', {}).get('negative_percentage', 0)
        total_reviews = overall_sentiment.get('total_reviews', 0)
        avg_rating = overall_sentiment.get('average_rating', 0)
        
        # Extract aspect performance
        aspects = analysis_data.get('aspect_sentiment', {})
        
        # Extract themes and issues
        themes = analysis_data.get('recurring_themes', {})
        executive_summary = analysis_data.get('executive_summary', {})
        
        # Extract language/international data
        language_analysis = analysis_data.get('language_analysis', {})
        international_appeal = language_analysis.get('international_appeal', 0)
        
        # Extract responsiveness data
        responsiveness = analysis_data.get('responsiveness_analysis', {})
        
        prompt = f"""
You are a tourism management and digital reputation expert. Analyze this TripAdvisor review data for {destination_name} and provide exactly 3 actionable insights focused on:

1. Tourism management improvements
2. Digital reputation management
3. Digital visibility and engagement strategies

ANALYSIS DATA:
==============

DESTINATION: {destination_name}
TOTAL REVIEWS: {total_reviews}
AVERAGE RATING: {avg_rating:.2f}/5
OVERALL SENTIMENT: {sentiment_score:.3f} ({positive_pct:.1f}% positive, {negative_pct:.1f}% negative)
INTERNATIONAL APPEAL: {international_appeal:.1f}% non-English reviews

ASPECT PERFORMANCE:
"""
        
        # Add aspect performance details
        for aspect, data in aspects.items():
            sentiment = data.get('average_sentiment', 0)
            mention_pct = data.get('mention_percentage', 0)
            prompt += f"- {aspect.title()}: {sentiment:.3f} sentiment, {mention_pct:.1f}% mention rate\n"
        
        prompt += f"\nRECURRING THEMES:\n"
        for theme, data in themes.items():
            percentage = data.get('percentage', 0)
            avg_sentiment = data.get('average_sentiment', 0)
            description = data.get('description', '')
            prompt += f"- {theme.replace('_', ' ').title()}: {percentage:.1f}% of reviews, {avg_sentiment:.3f} sentiment - {description}\n"
        
        prompt += f"\nEXECUTIVE SUMMARY INSIGHTS:\n"
        strengths = executive_summary.get('strengths', [])
        improvements = executive_summary.get('areas_for_improvement', [])
        recommendations = executive_summary.get('strategic_recommendations', [])
        
        if strengths:
            prompt += "STRENGTHS:\n"
            for strength in strengths[:3]:
                prompt += f"- {strength}\n"
        
        if improvements:
            prompt += "IMPROVEMENT AREAS:\n"
            for improvement in improvements[:3]:
                prompt += f"- {improvement}\n"
        
        if recommendations:
            prompt += "EXISTING RECOMMENDATIONS:\n"
            for rec in recommendations[:3]:
                prompt += f"- {rec}\n"
        
        prompt += f"""

MANAGEMENT RESPONSIVENESS:
{json.dumps(responsiveness, indent=2) if responsiveness else "No responsiveness data available"}

TASK:
Generate exactly 3 actionable insights in this format:

INSIGHT 1: [TOURISM MANAGEMENT]
Title: [Clear action-oriented title]
Issue: [Specific problem identified from the data]
Action: [Concrete, feasible steps to take]
Expected Impact: [What this will achieve]
Timeline: [Suggested implementation timeframe]

INSIGHT 2: [DIGITAL REPUTATION MANAGEMENT] 
Title: [Clear action-oriented title]
Issue: [Specific digital reputation problem]
Action: [Concrete steps for reputation management]
Expected Impact: [What this will achieve]
Timeline: [Suggested implementation timeframe]

INSIGHT 3: [DIGITAL VISIBILITY & ENGAGEMENT]
Title: [Clear action-oriented title]
Issue: [Specific visibility/engagement opportunity]
Action: [Concrete steps to improve digital presence]
Expected Impact: [What this will achieve]
Timeline: [Suggested implementation timeframe]

Focus on:
- Specific, implementable actions
- Government/management accountability
- Digital reputation improvement
- Review response strategies
- Infrastructure and service improvements
- International market development
- Feasible next steps based on actual visitor feedback

Provide insights that address both the positive opportunities and negative concerns found in the reviews.
"""
        
        return prompt
    
    def _parse_ai_response(self, response_text: str) -> Dict:
        """Parse the AI response into structured insights."""
        insights = {
            'actionable_insights': [],
            'summary': '',
            'focus_areas': ['tourism_management', 'digital_reputation', 'digital_visibility']
        }
        
        try:
            # Split response into sections
            sections = response_text.split('INSIGHT')
            
            for i, section in enumerate(sections[1:], 1):  # Skip first empty section
                if section.strip():
                    insight = self._parse_single_insight(f"INSIGHT{section}", i)
                    if insight:
                        insights['actionable_insights'].append(insight)
            
            # Generate summary
            insights['summary'] = f"Generated {len(insights['actionable_insights'])} actionable insights for tourism management and digital reputation improvement."
            
        except Exception as e:
            logger.error(f"Error parsing AI response: {str(e)}")
            insights['error'] = str(e)
            insights['raw_response'] = response_text
        
        return insights
    
    def _parse_single_insight(self, section_text: str, insight_number: int) -> Optional[Dict]:
        """Parse a single insight from the AI response."""
        try:
            lines = section_text.strip().split('\n')
            insight = {
                'insight_number': insight_number,
                'category': '',
                'title': '',
                'issue': '',
                'action': '',
                'expected_impact': '',
                'timeline': ''
            }
            
            current_field = None
            
            for line in lines:
                line = line.strip()
                
                if ':' in line:
                    field_part, content_part = line.split(':', 1)
                    field_part = field_part.strip().lower()
                    
                    if 'tourism management' in field_part or 'digital reputation' in field_part or 'digital visibility' in field_part:
                        if 'tourism' in field_part:
                            insight['category'] = 'Tourism Management'
                        elif 'reputation' in field_part:
                            insight['category'] = 'Digital Reputation Management'
                        else:
                            insight['category'] = 'Digital Visibility & Engagement'
                    elif 'title' in field_part:
                        insight['title'] = content_part.strip()
                        current_field = 'title'
                    elif 'issue' in field_part:
                        insight['issue'] = content_part.strip()
                        current_field = 'issue'
                    elif 'action' in field_part:
                        insight['action'] = content_part.strip()
                        current_field = 'action'
                    elif 'expected impact' in field_part or 'impact' in field_part:
                        insight['expected_impact'] = content_part.strip()
                        current_field = 'expected_impact'
                    elif 'timeline' in field_part:
                        insight['timeline'] = content_part.strip()
                        current_field = 'timeline'
                elif current_field and line:
                    # Continue previous field if line doesn't start a new field
                    insight[current_field] += ' ' + line
            
            # Clean up fields
            for key in insight:
                if isinstance(insight[key], str):
                    insight[key] = insight[key].strip()
            
            # Validate insight has required fields
            if insight['title'] and insight['action']:
                return insight
            
        except Exception as e:
            logger.error(f"Error parsing insight section: {str(e)}")
        
        return None
    
    def _generate_fallback_insights(self, analysis_data: Dict, destination_name: str) -> Dict:
        """Generate fallback insights when AI is not available."""
        
        overall_sentiment = analysis_data.get('overall_sentiment', {})
        aspects = analysis_data.get('aspect_sentiment', {})
        executive_summary = analysis_data.get('executive_summary', {})
        
        fallback_insights = {
            'actionable_insights': [
                {
                    'insight_number': 1,
                    'category': 'Tourism Management',
                    'title': 'Address Infrastructure and Maintenance Issues',
                    'issue': 'Analysis shows concerns about site maintenance and infrastructure decay affecting visitor experience.',
                    'action': 'Establish immediate maintenance protocol and secure government funding for infrastructure improvements.',
                    'expected_impact': 'Improved visitor satisfaction and prevention of further deterioration.',
                    'timeline': '3-6 months for immediate fixes, 12 months for major improvements'
                },
                {
                    'insight_number': 2,
                    'category': 'Digital Reputation Management',
                    'title': 'Implement Review Response Strategy',
                    'issue': 'Zero management responses to visitor reviews, including negative feedback about site conditions.',
                    'action': 'Designate response team, create response templates, and commit to responding to all reviews within 48 hours.',
                    'expected_impact': 'Improved online reputation and visitor confidence in management care.',
                    'timeline': '2-4 weeks to implement'
                },
                {
                    'insight_number': 3,
                    'category': 'Digital Visibility & Engagement',
                    'title': 'Leverage International Visitor Base',
                    'issue': 'Strong international appeal but potential for broader digital reach.',
                    'action': 'Create multilingual content, engage with international travel blogs, and optimize for international SEO.',
                    'expected_impact': 'Increased international visibility and visitor numbers.',
                    'timeline': '6-12 months for full implementation'
                }
            ],
            'summary': 'Generated fallback insights based on analysis patterns (AI service unavailable).',
            'focus_areas': ['tourism_management', 'digital_reputation', 'digital_visibility'],
            'generation_metadata': {
                'destination': destination_name,
                'generated_at': datetime.now().isoformat(),
                'model_used': 'fallback',
                'insights_count': 3
            }
        }
        
        return fallback_insights
    
    def save_insights(self, insights: Dict, output_dir: str = 'outputs', filename: str = None) -> str:
        """Save AI insights to JSON file."""
        os.makedirs(output_dir, exist_ok=True)
        
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            destination = insights.get('generation_metadata', {}).get('destination', 'unknown')
            filename = f"ai_insights_{destination}_{timestamp}.json"
        
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(insights, f, indent=2, ensure_ascii=False)
        
        logger.info(f"AI insights saved to: {filepath}")
        return filepath

# Convenience function
def generate_tourism_ai_insights(analysis_data: Dict, destination_name: str = "Tourism Destination", 
                               api_key: str = None, 
                               output_dir: str = 'outputs') -> Dict:
    """
    Quick function to generate AI insights from tourism analysis data.
    
    Args:
        analysis_data (Dict): Tourism analysis results
        destination_name (str): Name of the destination
        api_key (str): Google AI API key
        output_dir (str): Output directory for results
        
    Returns:
        Dict: AI-generated actionable insights
    """
    generator = TourismAIInsightsGenerator(api_key=api_key)
    insights = generator.generate_actionable_insights(analysis_data, destination_name)
    
    # Save the insights
    if insights:
        generator.save_insights(insights, output_dir)
    
    return insights 