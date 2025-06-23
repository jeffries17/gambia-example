#!/usr/bin/env python3
"""
Tourism Insights Web Application

Modern, interactive web application for tourism analysis with competitor comparisons,
AI-powered insights, and professional visualizations.
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
from streamlit_option_menu import option_menu
import base64

# Add local modules to path
sys.path.append('.')
sys.path.append('./sentiment_analyzer')

# Page configuration
st.set_page_config(
    page_title="Tourism Analytics Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/tourism-analytics',
        'Report a bug': None,
        'About': "# Tourism Analytics Intelligence\nProfessional tourism insights and competitor analysis platform."
    }
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global styling */
    .main {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.15);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #7f8c8d;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Insights panel */
    .insights-panel {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 2rem 0;
    }
    
    .insights-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .insight-item {
        background: rgba(255,255,255,0.1);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
    }
    
    /* Comparison section */
    .comparison-section {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
    }
    
    /* Side panel styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)

# Tourism destinations data
DESTINATIONS = {
    "Gambia (Kunta Kinteh Island)": {
        "file": "sentiment_analyzer/outputs/gambia_insights/tourism_insights_Gambia Tourism Destinations_20250623_153533.json",
        "description": "UNESCO World Heritage Site - Historic slave trade island",
        "location": "West Africa",
        "type": "Cultural Heritage"
    },
    "Senegal (Goree Island)": {
        "file": "goree_analysis_results.json",
        "description": "UNESCO World Heritage Site - Historic slave trade island",
        "location": "West Africa",
        "type": "Cultural Heritage"
    }
    # Additional destinations will be added here when competitor data is provided
}

def add_destination(name, file_path, description, location, destination_type):
    """Helper function to add a new destination to the app."""
    DESTINATIONS[name] = {
        "file": file_path,
        "description": description,
        "location": location,
        "type": destination_type
    }

def load_destination_data(destination_name):
    """Load destination data from file."""
    dest_info = DESTINATIONS.get(destination_name, {})
    
    if "file" in dest_info and os.path.exists(dest_info["file"]):
        try:
            with open(dest_info["file"], 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading data for {destination_name}: {e}")
            return None
    
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
        textfont=dict(size=14, family='Inter'),
        hovertemplate='<b>%{label}</b><br>%{percent}<br>Count: %{value:.0f}<extra></extra>'
    )])
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18, family='Inter', color='#2c3e50')),
        font=dict(family='Inter'),
        showlegend=True,
        height=400,
        margin=dict(t=60, b=40, l=40, r=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_aspect_performance_chart(data, title="Aspect Performance"):
    """Create modern bar chart for aspect performance."""
    aspects = data.get("aspect_sentiment", {})
    
    if not aspects:
        return go.Figure()
    
    aspect_names = []
    sentiment_scores = []
    mention_rates = []
    
    colors = ['#3498DB', '#E67E22', '#9B59B6', '#1ABC9C', '#E74C3C', '#F39C12']
    
    for i, (aspect, aspect_data) in enumerate(aspects.items()):
        aspect_names.append(aspect.replace('_', ' ').title())
        sentiment_scores.append(aspect_data.get('average_sentiment', 0))
        mention_rates.append(aspect_data.get('mention_percentage', 0))
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=aspect_names,
        y=sentiment_scores,
        marker_color=colors[:len(aspect_names)],
        text=[f'{score:.3f}' for score in sentiment_scores],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Sentiment: %{y:.3f}<br>Mentions: %{customdata:.1f}%<extra></extra>',
        customdata=mention_rates,
        name='Sentiment Score'
    ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18, family='Inter', color='#2c3e50')),
        xaxis_title="Tourism Aspects",
        yaxis_title="Sentiment Score",
        font=dict(family='Inter'),
        height=400,
        margin=dict(t=60, b=40, l=40, r=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    )
    
    return fig

def create_comparison_chart(dest1_data, dest2_data, dest1_name, dest2_name):
    """Create comparison chart between two destinations."""
    
    # Extract overall metrics
    dest1_overall = dest1_data.get("overall_sentiment", {})
    dest2_overall = dest2_data.get("overall_sentiment", {})
    
    metrics = ['Average Rating', 'Overall Sentiment', 'Positive %', 'Total Reviews']
    dest1_values = [
        dest1_overall.get('average_rating', 0),
        dest1_overall.get('overall_score', 0) * 10,  # Scale for visualization
        dest1_overall.get('sentiment_distribution', {}).get('positive_percentage', 0),
        dest1_overall.get('total_reviews', 0) / 100  # Scale for visualization
    ]
    dest2_values = [
        dest2_overall.get('average_rating', 0),
        dest2_overall.get('overall_score', 0) * 10,  # Scale for visualization
        dest2_overall.get('sentiment_distribution', {}).get('positive_percentage', 0),
        dest2_overall.get('total_reviews', 0) / 100  # Scale for visualization
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=dest1_values,
        theta=metrics,
        fill='toself',
        name=dest1_name,
        marker_color='#3498DB',
        line_color='#3498DB'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=dest2_values,
        theta=metrics,
        fill='toself',
        name=dest2_name,
        marker_color='#E74C3C',
        line_color='#E74C3C'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max(max(dest1_values), max(dest2_values)) * 1.2])
        ),
        title=dict(text="Destination Comparison", x=0.5, font=dict(size=18, family='Inter', color='#2c3e50')),
        font=dict(family='Inter'),
        height=500,
        showlegend=True
    )
    
    return fig



def main():
    """Main application function."""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌍 Tourism Analytics Intelligence</h1>
        <p>Professional Destination Analysis & Competitor Intelligence Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation menu
    selected = option_menu(
        menu_title=None,
        options=["🔍 Analysis", "⚖️ Competitor Comparison"],
        icons=["search", "bar-chart"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "#667eea", "font-size": "18px"},
            "nav-link": {"font-size": "16px", "text-align": "center", "margin": "0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#667eea"},
        }
    )
    
    if selected == "🔍 Analysis":
        single_analysis_page()
    elif selected == "⚖️ Competitor Comparison":
        comparison_page()

def single_analysis_page():
    """Single destination analysis page."""
    st.markdown("## 🔍 Destination Analysis")
    
    # Check if we have any destinations
    if not DESTINATIONS:
        st.warning("No destination data available. Please add destination data files.")
        return
    
    # Destination selector
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_destination = st.selectbox(
            "Select Destination for Analysis",
            list(DESTINATIONS.keys()),
            index=0
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Refresh Analysis", type="primary"):
            st.rerun()
    
    # Load destination data
    data = load_destination_data(selected_destination)
    
    if data:
        dest_info = DESTINATIONS[selected_destination]
        
        # Destination info
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%); 
                    padding: 1.5rem; border-radius: 12px; color: white; margin-bottom: 2rem;">
            <h3>📍 {selected_destination}</h3>
            <p><strong>Location:</strong> {dest_info.get('location', 'Unknown')}</p>
            <p><strong>Type:</strong> {dest_info.get('type', 'Unknown')}</p>
            <p><strong>Description:</strong> {dest_info.get('description', 'No description available')}</p>
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
            st.metric(
                "Average Rating",
                f"{overall.get('average_rating', 0):.1f}/5",
                delta=f"{overall.get('average_rating', 0) - 4.0:+.1f}" if overall.get('average_rating', 0) > 0 else None
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
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            sentiment_chart = create_sentiment_donut_chart(data)
            st.plotly_chart(sentiment_chart, use_container_width=True)
        
        with col2:
            aspect_chart = create_aspect_performance_chart(data)
            st.plotly_chart(aspect_chart, use_container_width=True)
        
        # Executive summary
        exec_summary = data.get("executive_summary", {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ✅ Strengths")
            for strength in exec_summary.get("strengths", []):
                st.markdown(f"• {strength}")
        
        with col2:
            st.markdown("### ⚠️ Areas for Improvement")
            for improvement in exec_summary.get("areas_for_improvement", []):
                st.markdown(f"• {improvement}")
    
    else:
        st.error("Could not load data for the selected destination.")

def comparison_page():
    """Competitor comparison page."""
    st.markdown("## ⚖️ Destination Comparison Analysis")
    
    # Destination selectors
    col1, col2, col3 = st.columns([1, 1, 1])
    
    # Check if we have enough destinations for comparison
    if len(DESTINATIONS) < 2:
        st.warning("You need at least 2 destinations to perform comparisons. Currently have: " + str(len(DESTINATIONS)))
        st.markdown("### Available Destinations:")
        for dest in DESTINATIONS.keys():
            st.markdown(f"• {dest}")
        st.markdown("**Please add competitor destination data to enable comparisons.**")
        return
    
    with col1:
        dest1 = st.selectbox(
            "Primary Destination",
            list(DESTINATIONS.keys()),
            index=0
        )
    
    with col2:
        available_competitors = [d for d in DESTINATIONS.keys() if d != dest1]
        if available_competitors:
            dest2 = st.selectbox(
                "Competitor Destination",
                available_competitors,
                index=0
            )
        else:
            st.warning("No competitor destinations available.")
            return
    
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📊 Compare", type="primary"):
            st.rerun()
    
    if dest1 != dest2:
        # Load data for both destinations
        data1 = load_destination_data(dest1)
        data2 = load_destination_data(dest2)
        
        if data1 and data2:
            st.markdown(f"""
            <div class="comparison-section">
                <h3>🔥 Head-to-Head Comparison</h3>
                <p>Comparing <strong>{dest1}</strong> vs <strong>{dest2}</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Comparison metrics
            overall1 = data1.get("overall_sentiment", {})
            overall2 = data2.get("overall_sentiment", {})
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                rating1 = overall1.get('average_rating', 0)
                rating2 = overall2.get('average_rating', 0)
                winner = dest1 if rating1 > rating2 else dest2
                st.markdown(f"""
                **Average Rating**
                - {dest1}: {rating1:.1f}/5
                - {dest2}: {rating2:.1f}/5
                
                🏆 **Winner:** {winner}
                """)
            
            with col2:
                sentiment1 = overall1.get('overall_score', 0)
                sentiment2 = overall2.get('overall_score', 0)
                winner = dest1 if sentiment1 > sentiment2 else dest2
                st.markdown(f"""
                **Sentiment Score**
                - {dest1}: {sentiment1:.3f}
                - {dest2}: {sentiment2:.3f}
                
                🏆 **Winner:** {winner}
                """)
            
            with col3:
                reviews1 = overall1.get('total_reviews', 0)
                reviews2 = overall2.get('total_reviews', 0)
                winner = dest1 if reviews1 > reviews2 else dest2
                st.markdown(f"""
                **Total Reviews**
                - {dest1}: {reviews1:,}
                - {dest2}: {reviews2:,}
                
                🏆 **Winner:** {winner}
                """)
            
            with col4:
                pos1 = overall1.get('sentiment_distribution', {}).get('positive_percentage', 0)
                pos2 = overall2.get('sentiment_distribution', {}).get('positive_percentage', 0)
                winner = dest1 if pos1 > pos2 else dest2
                st.markdown(f"""
                **Positive Rate**
                - {dest1}: {pos1:.1f}%
                - {dest2}: {pos2:.1f}%
                
                🏆 **Winner:** {winner}
                """)
            
            # Comparison chart
            comparison_chart = create_comparison_chart(data1, data2, dest1, dest2)
            st.plotly_chart(comparison_chart, use_container_width=True)
            
            # Side by side sentiment distribution
            col1, col2 = st.columns(2)
            
            with col1:
                chart1 = create_sentiment_donut_chart(data1, f"{dest1} Sentiment")
                st.plotly_chart(chart1, use_container_width=True)
            
            with col2:
                chart2 = create_sentiment_donut_chart(data2, f"{dest2} Sentiment")
                st.plotly_chart(chart2, use_container_width=True)
        
        else:
            st.error("Could not load data for one or both destinations.")
    else:
        st.warning("Please select different destinations for comparison.")

if __name__ == "__main__":
    main()