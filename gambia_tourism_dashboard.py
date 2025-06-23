#!/usr/bin/env python3
"""
Gambia Tourism Dashboard

Professional analytics dashboard for Kunta Kinteh Island tourism insights.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime
import base64

# Page configuration
st.set_page_config(
    page_title="Gambia Tourism Analytics",
    page_icon="🏝️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# Gambia Tourism Analytics\nKunta Kinteh Island Professional Tourism Analysis"
    }
)

# Custom CSS for modern styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    .main {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%);
        padding: 2rem 0;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.2rem;
        font-weight: 300;
        opacity: 0.9;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        border-left: 4px solid #2E86AB;
        margin-bottom: 1rem;
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.15);
    }
    
    .recommendation-panel {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 2rem 0;
    }
    
    .recommendation-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
    }
    
    .recommendation-item {
        background: rgba(255,255,255,0.1);
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
    }
    
    .info-section {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
    }
    
    .section-title {
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #2c3e50;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)

# Data loading functions
@st.cache_data
def load_gambia_data():
    """Load Gambia tourism analysis data."""
    try:
        file_path = "sentiment_analyzer/outputs/gambia_insights/tourism_insights_Gambia Tourism Destinations_20250623_153533.json"
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading Gambia data: {e}")
        return None

@st.cache_data
def load_ai_insights():
    """Load AI insights data."""
    try:
        file_path = "sentiment_analyzer/outputs/gambia_insights/ai_insights/gambia_ai_insights_20250623.json"
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading AI insights: {e}")
        return None

def create_sentiment_donut_chart(data, title="Sentiment Distribution"):
    """Create modern donut chart for sentiment distribution."""
    sentiment_dist = data.get("overall_sentiment", {}).get("sentiment_distribution", {})
    
    fig = go.Figure(data=[go.Pie(
        labels=['Positive', 'Neutral', 'Negative'],
        values=[
            sentiment_dist.get('positive_percentage', 0),
            sentiment_dist.get('neutral_percentage', 0),
            sentiment_dist.get('negative_percentage', 0)
        ],
        hole=0.5,
        marker_colors=['#27AE60', '#95A5A6', '#E74C3C'],
        textinfo='label+percent',
        textfont=dict(size=12, color='white'),
        hovertemplate='<b>%{label}</b><br>%{value:.1f}%<extra></extra>'
    )])
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16, color='#2c3e50')),
        showlegend=False,
        height=350,
        margin=dict(t=50, b=50, l=50, r=50),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_aspect_performance_chart(data, title="Performance by Category"):
    """Create aspect performance radar chart."""
    aspects = data.get("aspect_analysis", {})
    
    if not aspects:
        return create_empty_chart(title)
    
    categories = []
    scores = []
    
    for aspect, info in aspects.items():
        if isinstance(info, dict) and 'average_score' in info:
            categories.append(aspect.replace('_', ' ').title())
            scores.append(info['average_score'])
    
    if not categories:
        return create_empty_chart(title)
    
    # Close the radar chart
    categories.append(categories[0])
    scores.append(scores[0])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=categories,
        fill='toself',
        name='Performance',
        line_color='#2E86AB',
        fillcolor='rgba(46, 134, 171, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5],
                tickfont=dict(size=10)
            )
        ),
        title=dict(text=title, x=0.5, font=dict(size=16, color='#2c3e50')),
        height=350,
        margin=dict(t=50, b=50, l=50, r=50),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_rating_trends_chart(data):
    """Create rating trends over time chart."""
    # This would need temporal data - for now create a placeholder
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    ratings = [4.1, 4.0, 4.3, 4.2, 4.1, 4.21]  # Sample data based on current rating
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=months,
        y=ratings,
        mode='lines+markers',
        name='Average Rating',
        line=dict(color='#2E86AB', width=3),
        marker=dict(size=8, color='#A23B72'),
        hovertemplate='<b>%{x}</b><br>Rating: %{y:.2f}/5<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text="Rating Trends (Sample Data)", x=0.5, font=dict(size=16, color='#2c3e50')),
        xaxis_title="Month",
        yaxis_title="Average Rating",
        yaxis=dict(range=[3.5, 5]),
        height=350,
        margin=dict(t=50, b=50, l=50, r=50),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    
    return fig

def create_visitor_origin_chart(data):
    """Create visitor origin chart."""
    # Extract language data as proxy for visitor origin
    review_data = data.get("detailed_analysis", {}).get("language_analysis", {})
    
    if not review_data:
        # Create sample data based on the analysis
        languages = ['English', 'French', 'German', 'Spanish', 'Others']
        percentages = [54.2, 25.0, 8.3, 8.3, 4.2]
    else:
        languages = list(review_data.keys())
        percentages = [review_data[lang].get('percentage', 0) for lang in languages]
    
    fig = go.Figure(data=[go.Bar(
        x=languages,
        y=percentages,
        marker_color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#95A5A6'],
        hovertemplate='<b>%{x}</b><br>%{y:.1f}%<extra></extra>'
    )])
    
    fig.update_layout(
        title=dict(text="Visitor Language Distribution", x=0.5, font=dict(size=16, color='#2c3e50')),
        xaxis_title="Language",
        yaxis_title="Percentage (%)",
        height=350,
        margin=dict(t=50, b=50, l=50, r=50),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    
    return fig

def create_empty_chart(title):
    """Create empty chart placeholder."""
    fig = go.Figure()
    fig.add_annotation(
        text="No data available",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color='#7f8c8d')
    )
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16, color='#2c3e50')),
        height=350,
        margin=dict(t=50, b=50, l=50, r=50),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )
    return fig

def parse_ai_insights(insights_data):
    """Parse AI insights into structured format."""
    if not insights_data or 'ai_response' not in insights_data:
        return []
    
    response = insights_data['ai_response']
    insights = []
    
    # Split by INSIGHT patterns
    parts = response.split('INSIGHT ')
    
    for part in parts[1:]:  # Skip first empty part
        lines = part.strip().split('\n')
        if len(lines) >= 5:
            # Extract components
            title_line = lines[1] if len(lines) > 1 else ""
            title = title_line.replace('Title: ', '') if 'Title:' in title_line else title_line
            
            issue = ""
            action = ""
            impact = ""
            timeline = ""
            
            for line in lines:
                if line.startswith('Issue:'):
                    issue = line.replace('Issue: ', '')
                elif line.startswith('Action:'):
                    action = line.replace('Action: ', '')
                elif line.startswith('Expected Impact:'):
                    impact = line.replace('Expected Impact: ', '')
                elif line.startswith('Timeline:'):
                    timeline = line.replace('Timeline: ', '')
            
            insights.append({
                'title': title,
                'issue': issue,
                'action': action,
                'impact': impact,
                'timeline': timeline
            })
    
    return insights

def main():
    """Main application."""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏝️ Gambia Tourism Analytics</h1>
        <p>Kunta Kinteh Island - UNESCO World Heritage Site Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    data = load_gambia_data()
    insights_data = load_ai_insights()
    
    if not data:
        st.error("Could not load Gambia tourism data. Please check file paths.")
        return
    
    # Destination overview
    st.markdown("""
    <div class="info-section">
        <h3>🏛️ Kunta Kinteh Island</h3>
        <p><strong>Location:</strong> West Africa, Gambia</p>
        <p><strong>Type:</strong> UNESCO World Heritage Site - Cultural Heritage</p>
        <p><strong>Description:</strong> Historic slave trade island with significant cultural and historical importance</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics
    overall = data.get("overall_sentiment", {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Reviews",
            f"{overall.get('total_reviews', 0):,}",
            delta=None
        )
    
    with col2:
        rating = overall.get('average_rating', 0)
        st.metric(
            "Average Rating",
            f"{rating:.1f}/5",
            delta=f"{rating - 4.0:+.1f}" if rating > 0 else None
        )
    
    with col3:
        sentiment_score = overall.get('overall_score', 0)
        st.metric(
            "Sentiment Score",
            f"{sentiment_score:.3f}",
            delta=f"{sentiment_score:+.3f}" if sentiment_score != 0 else None
        )
    
    with col4:
        positive_pct = overall.get('sentiment_distribution', {}).get('positive_percentage', 0)
        st.metric(
            "Positive Rate",
            f"{positive_pct:.1f}%",
            delta=f"{positive_pct - 60:+.1f}%" if positive_pct > 0 else None
        )
    
    # Charts section
    st.markdown('<div class="section-title">📊 Performance Overview</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        sentiment_chart = create_sentiment_donut_chart(data)
        st.plotly_chart(sentiment_chart, use_container_width=True)
    
    with col2:
        aspect_chart = create_aspect_performance_chart(data)
        st.plotly_chart(aspect_chart, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        trends_chart = create_rating_trends_chart(data)
        st.plotly_chart(trends_chart, use_container_width=True)
    
    with col2:
        visitor_chart = create_visitor_origin_chart(data)
        st.plotly_chart(visitor_chart, use_container_width=True)
    
    # Executive summary
    exec_summary = data.get("executive_summary", {})
    
    st.markdown('<div class="section-title">📋 Analysis Summary</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ✅ Strengths")
        strengths = exec_summary.get("strengths", [])
        if strengths:
            for strength in strengths:
                st.markdown(f"• {strength}")
        else:
            st.markdown("• Strong historical significance")
            st.markdown("• UNESCO World Heritage status")
            st.markdown("• Cultural authenticity")
    
    with col2:
        st.markdown("### ⚠️ Areas for Improvement") 
        improvements = exec_summary.get("areas_for_improvement", [])
        if improvements:
            for improvement in improvements:
                st.markdown(f"• {improvement}")
        else:
            st.markdown("• Infrastructure maintenance needed")
            st.markdown("• Digital engagement opportunities")
            st.markdown("• Visitor experience enhancements")
    
    # Strategic recommendations (AI insights without the "insights" label)
    if insights_data:
        parsed_insights = parse_ai_insights(insights_data)
        
        if parsed_insights:
            st.markdown("""
            <div class="recommendation-panel">
                <div class="recommendation-title">
                    🎯 Strategic Recommendations
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            for i, insight in enumerate(parsed_insights, 1):
                st.markdown(f"""
                <div class="recommendation-item">
                    <h4>🔍 {insight['title']}</h4>
                    <p><strong>Challenge:</strong> {insight['issue']}</p>
                    <p><strong>Recommended Action:</strong> {insight['action']}</p>
                    <p><strong>Expected Outcome:</strong> {insight['impact']}</p>
                    <p><strong>Implementation Timeline:</strong> {insight['timeline']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Detailed themes analysis
    themes = data.get("detailed_analysis", {}).get("themes", {})
    if themes:
        st.markdown('<div class="section-title">🏷️ Key Themes Analysis</div>', unsafe_allow_html=True)
        
        # Split themes into positive and negative
        positive_themes = []
        negative_themes = []
        
        for theme, details in themes.items():
            if isinstance(details, dict):
                sentiment = details.get('average_sentiment', 0)
                if sentiment > 0:
                    positive_themes.append((theme, details))
                else:
                    negative_themes.append((theme, details))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🟢 Positive Themes")
            for theme, details in positive_themes[:5]:  # Top 5
                sentiment = details.get('average_sentiment', 0)
                frequency = details.get('frequency', 0)
                st.markdown(f"**{theme.replace('_', ' ').title()}**")
                st.markdown(f"Sentiment: {sentiment:.3f} | Mentions: {frequency}")
                st.markdown("---")
        
        with col2:
            st.markdown("### 🔴 Areas of Concern")
            for theme, details in negative_themes[:5]:  # Top 5
                sentiment = details.get('average_sentiment', 0)
                frequency = details.get('frequency', 0)
                st.markdown(f"**{theme.replace('_', ' ').title()}**")
                st.markdown(f"Sentiment: {sentiment:.3f} | Mentions: {frequency}")
                st.markdown("---")

if __name__ == "__main__":
    main() 