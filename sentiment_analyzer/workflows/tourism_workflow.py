#!/usr/bin/env python3
"""
Integrated Tourism Analysis Workflow

This module provides end-to-end workflow for tourism boards:
1. Extract reviews from TripAdvisor URLs
2. Analyze for tourism-specific insights
3. Generate professional reports
4. Support competitor comparison
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

# Add parent directories to path for imports
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)

from extractors.tripadvisor_extractor import extract_reviews_from_url
from analysis.tourism_insights_analyzer import analyze_tourism_insights
from reporting.tourism_report_generator import generate_tourism_report
from comparison.destination_comparator import compare_destinations_from_files

logger = logging.getLogger(__name__)

class TourismAnalysisWorkflow:
    """
    Complete workflow for tourism board analysis.
    """
    
    def __init__(self, output_base_dir='outputs'):
        """Initialize the workflow."""
        self.output_base_dir = output_base_dir
        
        # Create output directories
        self.extraction_dir = os.path.join(output_base_dir, 'extractions')
        self.insights_dir = os.path.join(output_base_dir, 'insights')
        self.reports_dir = os.path.join(output_base_dir, 'reports')
        self.comparisons_dir = os.path.join(output_base_dir, 'comparisons')
        
        for directory in [self.extraction_dir, self.insights_dir, self.reports_dir, self.comparisons_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def analyze_single_destination(self, url: str, destination_name: str = None, 
                                 max_reviews: int = 100, generate_reports: bool = True) -> Dict:
        """
        Complete analysis workflow for a single destination.
        
        Args:
            url (str): TripAdvisor URL
            destination_name (str): Custom name for the destination
            max_reviews (int): Maximum reviews to extract
            generate_reports (bool): Whether to generate reports
            
        Returns:
            Dict: Complete analysis results with file paths
        """
        logger.info(f"Starting complete analysis for: {url}")
        
        results = {
            'workflow_metadata': {
                'url': url,
                'destination_name': destination_name,
                'max_reviews': max_reviews,
                'analysis_date': datetime.now().isoformat()
            },
            'files_created': [],
            'analysis_summary': {}
        }
        
        try:
            # Step 1: Extract reviews
            logger.info("Step 1: Extracting reviews from TripAdvisor...")
            extraction_data = extract_reviews_from_url(url, max_reviews=max_reviews, save_file=False)
            
            # Save extraction data
            business_name = extraction_data['business_info']['name']
            safe_name = destination_name or business_name.replace(' ', '_').replace('/', '_')
            
            extraction_filename = f"extraction_{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            extraction_path = os.path.join(self.extraction_dir, extraction_filename)
            
            with open(extraction_path, 'w', encoding='utf-8') as f:
                json.dump(extraction_data, f, indent=2, ensure_ascii=False)
            
            results['files_created'].append(extraction_path)
            results['extraction_data'] = extraction_data
            
            logger.info(f"✓ Extracted {len(extraction_data['reviews'])} reviews")
            
            # Step 2: Analyze tourism insights
            logger.info("Step 2: Analyzing tourism insights...")
            insights = analyze_tourism_insights(extraction_data, output_dir=self.insights_dir)
            
            results['insights'] = insights
            results['analysis_summary'] = {
                'overall_sentiment': insights.get('overall_sentiment', {}).get('overall_score', 0),
                'total_reviews': insights.get('analysis_metadata', {}).get('total_reviews', 0),
                'key_strengths': len(insights.get('executive_summary', {}).get('strengths', [])),
                'improvement_areas': len(insights.get('executive_summary', {}).get('areas_for_improvement', [])),
                'language_diversity': insights.get('language_analysis', {}).get('diversity_score', 0)
            }
            
            logger.info("✓ Tourism insights analysis complete")
            
            # Step 3: Generate reports
            if generate_reports:
                logger.info("Step 3: Generating reports...")
                
                # Executive report
                exec_report_path = generate_tourism_report(insights, report_type='executive', output_dir=self.reports_dir)
                results['files_created'].append(exec_report_path)
                
                # Detailed report
                detailed_report_path = generate_tourism_report(insights, report_type='detailed', output_dir=self.reports_dir)
                results['files_created'].append(detailed_report_path)
                
                logger.info("✓ Reports generated successfully")
            
            # Step 4: Create summary
            results['summary'] = self._create_workflow_summary(results)
            
            logger.info("✓ Complete analysis workflow finished successfully")
            return results
            
        except Exception as e:
            logger.error(f"Workflow failed: {str(e)}")
            results['error'] = str(e)
            return results
    
    def compare_multiple_destinations(self, destinations: List[Dict], 
                                    generate_comparison_report: bool = True) -> Dict:
        """
        Compare multiple destinations.
        
        Args:
            destinations (List[Dict]): List of destination configs with 'url' and 'name'
            generate_comparison_report (bool): Whether to generate comparison report
            
        Returns:
            Dict: Comparison results
        """
        logger.info(f"Starting comparison analysis for {len(destinations)} destinations")
        
        results = {
            'comparison_metadata': {
                'destinations': [d.get('name', 'Unknown') for d in destinations],
                'comparison_date': datetime.now().isoformat(),
                'total_destinations': len(destinations)
            },
            'individual_analyses': {},
            'comparison_results': {},
            'files_created': []
        }
        
        extraction_files = {}
        
        try:
            # Step 1: Analyze each destination individually
            for i, dest_config in enumerate(destinations):
                url = dest_config['url']
                name = dest_config.get('name', f'Destination_{i+1}')
                max_reviews = dest_config.get('max_reviews', 100)
                
                logger.info(f"Analyzing destination {i+1}/{len(destinations)}: {name}")
                
                # Run individual analysis
                individual_result = self.analyze_single_destination(
                    url=url, 
                    destination_name=name, 
                    max_reviews=max_reviews,
                    generate_reports=False  # Skip individual reports for comparison
                )
                
                results['individual_analyses'][name] = individual_result
                results['files_created'].extend(individual_result.get('files_created', []))
                
                # Prepare for comparison
                extraction_files[name] = individual_result.get('files_created', [None])[0]
            
            # Step 2: Generate comparison
            logger.info("Generating destination comparison...")
            comparison_data = compare_destinations_from_files(extraction_files, output_dir=self.comparisons_dir)
            
            results['comparison_results'] = comparison_data
            
            # Step 3: Generate comparison report
            if generate_comparison_report:
                logger.info("Generating comparison report...")
                comparison_report_path = generate_tourism_report(
                    comparison_data, 
                    report_type='comparison', 
                    output_dir=self.reports_dir
                )
                results['files_created'].append(comparison_report_path)
            
            # Step 4: Create comparison summary
            results['comparison_summary'] = self._create_comparison_summary(results)
            
            logger.info("✓ Multi-destination comparison complete")
            return results
            
        except Exception as e:
            logger.error(f"Comparison workflow failed: {str(e)}")
            results['error'] = str(e)
            return results
    
    def quick_destination_analysis(self, url: str, destination_name: str = None) -> Dict:
        """
        Quick analysis with minimal reviews for rapid insights.
        
        Args:
            url (str): TripAdvisor URL
            destination_name (str): Optional destination name
            
        Returns:
            Dict: Quick analysis results
        """
        logger.info(f"Starting quick analysis for: {url}")
        
        return self.analyze_single_destination(
            url=url,
            destination_name=destination_name,
            max_reviews=25,  # Reduced for speed
            generate_reports=True
        )
    
    def batch_url_analysis(self, urls: List[str], destination_names: List[str] = None) -> Dict:
        """
        Analyze multiple URLs in batch.
        
        Args:
            urls (List[str]): List of TripAdvisor URLs
            destination_names (List[str]): Optional list of destination names
            
        Returns:
            Dict: Batch analysis results
        """
        if destination_names and len(destination_names) != len(urls):
            raise ValueError("Number of destination names must match number of URLs")
        
        destinations = []
        for i, url in enumerate(urls):
            dest_config = {
                'url': url,
                'name': destination_names[i] if destination_names else f'Destination_{i+1}',
                'max_reviews': 50  # Moderate number for batch processing
            }
            destinations.append(dest_config)
        
        return self.compare_multiple_destinations(destinations, generate_comparison_report=True)
    
    def _create_workflow_summary(self, results: Dict) -> Dict:
        """Create a summary of the workflow results."""
        insights = results.get('insights', {})
        
        summary = {
            'destination': results['workflow_metadata'].get('destination_name', 'Unknown'),
            'analysis_quality': self._assess_analysis_quality(insights),
            'key_metrics': {
                'overall_sentiment': insights.get('overall_sentiment', {}).get('overall_score', 0),
                'total_reviews': insights.get('analysis_metadata', {}).get('total_reviews', 0),
                'confidence_level': insights.get('overall_sentiment', {}).get('confidence_level', 'Unknown')
            },
            'top_insights': self._extract_top_insights(insights),
            'files_generated': len(results.get('files_created', [])),
            'next_steps': self._suggest_next_steps(insights)
        }
        
        return summary
    
    def _create_comparison_summary(self, results: Dict) -> Dict:
        """Create a summary of the comparison results."""
        comparison_data = results.get('comparison_results', {})
        overall_comparison = comparison_data.get('overall_comparison', {})
        
        if not overall_comparison:
            return {'error': 'No comparison data available'}
        
        # Find best and worst performers
        by_sentiment = sorted(overall_comparison.items(), 
                            key=lambda x: x[1].get('avg_sentiment', 0), reverse=True)
        
        summary = {
            'total_destinations': len(overall_comparison),
            'best_performer': by_sentiment[0] if by_sentiment else None,
            'worst_performer': by_sentiment[-1] if by_sentiment else None,
            'average_sentiment': sum(data.get('avg_sentiment', 0) for data in overall_comparison.values()) / len(overall_comparison),
            'total_reviews_analyzed': comparison_data.get('comparison_metadata', {}).get('total_reviews', 0),
            'key_differentiators': self._identify_differentiators(comparison_data),
            'competitive_insights': self._generate_competitive_insights(comparison_data)
        }
        
        return summary
    
    def _assess_analysis_quality(self, insights: Dict) -> str:
        """Assess the quality/reliability of the analysis."""
        total_reviews = insights.get('analysis_metadata', {}).get('total_reviews', 0)
        
        if total_reviews >= 100:
            return "High - Large sample size provides reliable insights"
        elif total_reviews >= 50:
            return "Good - Adequate sample size for meaningful analysis"
        elif total_reviews >= 25:
            return "Fair - Limited sample size, insights are indicative"
        else:
            return "Low - Very limited data, insights should be treated cautiously"
    
    def _extract_top_insights(self, insights: Dict) -> List[str]:
        """Extract the most important insights."""
        top_insights = []
        
        # Overall sentiment insight
        overall_sentiment = insights.get('overall_sentiment', {}).get('overall_score', 0)
        if overall_sentiment > 0.4:
            top_insights.append(f"Strong positive sentiment ({overall_sentiment:.2f}) indicates high visitor satisfaction")
        elif overall_sentiment < 0:
            top_insights.append(f"Negative sentiment ({overall_sentiment:.2f}) suggests significant visitor concerns")
        
        # Top performing aspects
        aspects = insights.get('aspect_sentiment', {})
        strong_aspects = [aspect for aspect, data in aspects.items() 
                         if data.get('average_sentiment', 0) > 0.4 and data.get('mention_percentage', 0) > 10]
        if strong_aspects:
            top_insights.append(f"Strongest performance in: {', '.join(strong_aspects)}")
        
        # Language diversity
        lang_data = insights.get('language_analysis', {})
        international_appeal = lang_data.get('international_appeal', 0)
        if international_appeal > 30:
            top_insights.append(f"Strong international appeal ({international_appeal:.1f}% non-English reviews)")
        
        return top_insights[:3]  # Return top 3 insights
    
    def _suggest_next_steps(self, insights: Dict) -> List[str]:
        """Suggest next steps based on analysis."""
        next_steps = []
        
        # Based on sentiment
        overall_sentiment = insights.get('overall_sentiment', {}).get('overall_score', 0)
        if overall_sentiment > 0.4:
            next_steps.append("Leverage positive sentiment in marketing campaigns")
        elif overall_sentiment < 0:
            next_steps.append("Prioritize addressing negative feedback before marketing push")
        
        # Based on responsiveness
        responsiveness = insights.get('responsiveness_analysis', {})
        resp_score = responsiveness.get('overall_responsiveness_score', 0)
        if resp_score < 10:
            next_steps.append("Implement guest feedback response training")
        
        # Based on international appeal
        lang_data = insights.get('language_analysis', {})
        international_appeal = lang_data.get('international_appeal', 0)
        if international_appeal < 15:
            next_steps.append("Develop international marketing strategy")
        
        next_steps.append("Conduct competitor comparison analysis")
        next_steps.append("Monitor sentiment trends over time")
        
        return next_steps[:4]  # Return top 4 suggestions
    
    def _identify_differentiators(self, comparison_data: Dict) -> List[str]:
        """Identify key differentiators between destinations."""
        differentiators = []
        
        overall = comparison_data.get('overall_comparison', {})
        if len(overall) >= 2:
            # Sentiment range
            sentiments = [data.get('avg_sentiment', 0) for data in overall.values()]
            sentiment_range = max(sentiments) - min(sentiments)
            
            if sentiment_range > 0.5:
                differentiators.append("Significant sentiment differences between destinations")
            
            # Review volume differences
            review_counts = [data.get('review_count', 0) for data in overall.values()]
            if max(review_counts) > min(review_counts) * 3:
                differentiators.append("Large differences in review volume/visibility")
        
        return differentiators
    
    def _generate_competitive_insights(self, comparison_data: Dict) -> List[str]:
        """Generate competitive insights from comparison."""
        insights = []
        
        overall = comparison_data.get('overall_comparison', {})
        if overall:
            # Best performer insight
            by_sentiment = sorted(overall.items(), 
                                key=lambda x: x[1].get('avg_sentiment', 0), reverse=True)
            
            if len(by_sentiment) >= 2:
                best = by_sentiment[0]
                second = by_sentiment[1]
                
                sentiment_gap = best[1].get('avg_sentiment', 0) - second[1].get('avg_sentiment', 0)
                
                if sentiment_gap > 0.3:
                    insights.append(f"{best[0]} has significant sentiment advantage over competitors")
                elif sentiment_gap < 0.1:
                    insights.append("Very close competitive performance - small improvements could change rankings")
        
        insights.append("Focus on differentiation through unique cultural or geographic advantages")
        
        return insights

# Convenience functions for common workflows

def analyze_destination_from_url(url: str, destination_name: str = None, 
                               max_reviews: int = 100) -> Dict:
    """
    Complete analysis workflow for a single TripAdvisor URL.
    
    Args:
        url (str): TripAdvisor URL
        destination_name (str): Optional custom name
        max_reviews (int): Maximum reviews to extract
        
    Returns:
        Dict: Complete analysis results
    """
    workflow = TourismAnalysisWorkflow()
    return workflow.analyze_single_destination(url, destination_name, max_reviews)

def compare_destinations_from_urls(urls_and_names: List[tuple], 
                                 max_reviews_each: int = 50) -> Dict:
    """
    Compare multiple destinations from TripAdvisor URLs.
    
    Args:
        urls_and_names (List[tuple]): List of (url, name) tuples
        max_reviews_each (int): Max reviews per destination
        
    Returns:
        Dict: Comparison results
    """
    destinations = [
        {'url': url, 'name': name, 'max_reviews': max_reviews_each}
        for url, name in urls_and_names
    ]
    
    workflow = TourismAnalysisWorkflow()
    return workflow.compare_multiple_destinations(destinations)

def quick_gambia_analysis(gambia_urls: List[str]) -> Dict:
    """
    Quick analysis setup for Gambia destinations.
    
    Args:
        gambia_urls (List[str]): List of Gambia TripAdvisor URLs
        
    Returns:
        Dict: Analysis results
    """
    destinations = [
        {'url': url, 'name': f'Gambia_Destination_{i+1}', 'max_reviews': 30}
        for i, url in enumerate(gambia_urls)
    ]
    
    workflow = TourismAnalysisWorkflow()
    return workflow.compare_multiple_destinations(destinations)

if __name__ == "__main__":
    # Example usage
    print("Tourism Analysis Workflow")
    print("========================")
    
    # Get URL from user
    url = input("Enter TripAdvisor URL for analysis: ").strip()
    
    if url:
        try:
            # Run quick analysis
            results = analyze_destination_from_url(url, max_reviews=25)
            
            print(f"\n✓ Analysis Complete!")
            print(f"Overall Sentiment: {results['analysis_summary']['overall_sentiment']:.2f}")
            print(f"Total Reviews: {results['analysis_summary']['total_reviews']}")
            print(f"Files Created: {len(results['files_created'])}")
            
            print(f"\nGenerated Files:")
            for file_path in results['files_created']:
                print(f"  - {file_path}")
                
        except Exception as e:
            print(f"Error: {str(e)}")
    else:
        print("No URL provided.") 