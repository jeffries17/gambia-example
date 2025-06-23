#!/usr/bin/env python3
"""
Master script to run all analyses with the new directory structure.
This script coordinates the execution of all analysis scripts to 
generate a complete, consistently organized set of outputs.
"""

import os
import subprocess
import sys
import time
from datetime import datetime

def run_command(command, description):
    """Run a command with proper output handling."""
    print(f"\n{'='*80}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {command}")
    print(f"{'='*80}\n")
    
    # Setup environment for Python imports
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__))
    
    start_time = time.time()
    process = subprocess.run(command, shell=True, text=True, env=env)
    end_time = time.time()
    
    if process.returncode == 0:
        print(f"\n✅ {description} completed successfully in {end_time - start_time:.1f} seconds")
        return True
    else:
        print(f"\n❌ {description} failed with exit code {process.returncode}")
        return False

def main():
    """Run all analysis scripts in the correct order with the new directory structure."""
    # Get the project root directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    tonga_analysis_dir = os.path.join(project_dir, "tonga_analysis")
    
    # Print start time
    start_time = datetime.now()
    print(f"\nStarting comprehensive analysis at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output will be organized in the new directory structure")
    
    # Create the old_outputs directory if it doesn't exist
    old_outputs_dir = os.path.join(project_dir, "old_outputs")
    if not os.path.exists(old_outputs_dir):
        os.makedirs(old_outputs_dir)
    
    # Move existing outputs to old_outputs if they exist and aren't already moved
    outputs_dir = os.path.join(project_dir, "outputs")
    if os.path.exists(outputs_dir) and os.listdir(outputs_dir):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(old_outputs_dir, f"outputs_backup_{timestamp}")
        print(f"Moving existing outputs to {backup_dir}")
        run_command(f"mkdir -p {backup_dir} && cp -r {outputs_dir}/* {backup_dir}/", "Backup existing outputs")
        run_command(f"rm -rf {outputs_dir}/*", "Clear old outputs")
    
    # Make sure the new output directory structure exists
    run_command(
        f"mkdir -p {outputs_dir}/by_island/{{tongatapu,vavau,haapai,eua}} "
        f"{outputs_dir}/by_sector/{{accommodations,attractions,restaurants}} "
        f"{outputs_dir}/by_nationality {outputs_dir}/by_traveler_segment "
        f"{outputs_dir}/regional_comparison/{{fiji,samoa,tahiti,tonga_summary,overall}} "
        f"{outputs_dir}/consolidated_reports",
        "Create new output directory structure"
    )
    
    # Find virtual environment activation path
    venv_activate = os.path.join(project_dir, "sentiment_env", "bin", "activate")
    
    # Set Python path to include project directory 
    python_path = f"PYTHONPATH={project_dir}"
    
    # Run analysis scripts in order with virtual environment activated
    scripts = [
        # Main comprehensive analysis
        (f"source {venv_activate} && {python_path} python3 tonga_analysis/run_analysis.py", "Basic analysis"),
        
        # Island-based analyses
        (f"source {venv_activate} && {python_path} python3 tonga_analysis/run_island_analysis.py", "Island analysis"),
        (f"source {venv_activate} && {python_path} python3 tonga_analysis/run_island_accommodation_analysis.py", "Island-specific accommodation analysis"),
        (f"source {venv_activate} && {python_path} python3 tonga_analysis/run_island_restaurant_analysis.py", "Island-specific restaurant analysis"),
        (f"source {venv_activate} && {python_path} python3 tonga_analysis/run_island_attraction_analysis.py", "Island-specific attraction analysis"),
        
        # Sector-specific analyses
        (f"source {venv_activate} && {python_path} python3 tonga_analysis/accommodation_analyzer.py", "Accommodation analysis"),
        (f"source {venv_activate} && {python_path} python3 tonga_analysis/attractions_analyzer.py", "Attractions analysis"),
        (f"source {venv_activate} && {python_path} python3 tonga_analysis/restaurant_analyzer.py", "Restaurant analysis"),
        
        # Advanced and comparative analyses
        (f"source {venv_activate} && {python_path} python3 tonga_analysis/run_traveler_segments.py", "Traveler segment analysis"),
        (f"source {venv_activate} && {python_path} python3 tonga_analysis/run_regional_comparison.py --use-tonga-data", "Regional comparison analysis"),
        
        # Thematic analyses
        (f"source {venv_activate} && {python_path} python3 tonga_analysis/accommodation_theme_sentiment.py", "Accommodation theme sentiment analysis"),
        (f"source {venv_activate} && {python_path} python3 tonga_analysis/thematic_nationality_analysis.py", "Thematic nationality analysis"),
        (f"source {venv_activate} && {python_path} python3 tonga_analysis/theme_sentiment_drivers.py", "Theme sentiment drivers analysis"),
        
        # Visualizations and specific insights
        (f"source {venv_activate} && {python_path} python3 tonga_analysis/segment_sentiment_visualization.py", "Segment sentiment visualization"),
        (f"source {venv_activate} && {python_path} python3 tonga_analysis/island_sentiment_comparison.py", "Island sentiment comparison"),
        (f"source {venv_activate} && {python_path} python3 tonga_analysis/seasonal_patterns.py", "Seasonal patterns analysis"),
        (f"source {venv_activate} && {python_path} python3 word_cloud_generator.py", "Word cloud generation")
    ]
    
    success_count = 0
    failure_count = 0
    
    for cmd, desc in scripts:
        if run_command(cmd, desc):
            success_count += 1
        else:
            failure_count += 1
            print(f"Continuing with remaining analyses despite failure in {desc}")
    
    # Calculate total run time
    end_time = datetime.now()
    total_seconds = (end_time - start_time).total_seconds()
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total runtime: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Scripts run successfully: {success_count}/{len(scripts)}")
    if failure_count > 0:
        print(f"Scripts failed: {failure_count}/{len(scripts)}")
    print(f"{'='*80}\n")
    
    print("Outputs are now organized in the following structure:")
    print(f"- {outputs_dir}/by_island/ - Analysis by island")
    print(f"- {outputs_dir}/by_sector/ - Analysis by sector (accommodations, attractions, restaurants)")
    print(f"- {outputs_dir}/by_nationality/ - Analysis by visitor nationality")
    print(f"- {outputs_dir}/by_traveler_segment/ - Analysis by traveler segment")
    print(f"- {outputs_dir}/regional_comparison/ - Regional competitive analysis")
    print(f"- {outputs_dir}/consolidated_reports/ - Overview and summary reports")
    print("\nYou can now review the visualizations and findings in these directories.")

if __name__ == "__main__":
    main()