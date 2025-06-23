#!/usr/bin/env python3
"""
Tourism Analytics App Launcher

Launches the Streamlit web application with proper configuration
and error handling.
"""

import os
import sys
import subprocess
import webbrowser
from pathlib import Path
import time

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'streamlit',
        'plotly',
        'pandas',
        'streamlit_option_menu'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        
        for package in missing_packages:
            subprocess.run([
                sys.executable, "-m", "pip", "install", package
            ], check=True)
        
        print("✅ All packages installed successfully!")
    else:
        print("✅ All dependencies are installed.")

def launch_app():
    """Launch the Streamlit application."""
    app_file = "tourism_insights_app.py"
    config_file = "streamlit_config.toml"
    
    # Check if app file exists
    if not os.path.exists(app_file):
        print(f"❌ Application file '{app_file}' not found.")
        print("Please ensure you're in the correct directory.")
        return False
    
    # Check if config file exists
    config_args = []
    if os.path.exists(config_file):
        config_args = ["--config", config_file]
        print(f"✅ Using configuration file: {config_file}")
    
    print("🚀 Launching Tourism Analytics Intelligence App...")
    print("📍 The app will open in your default web browser")
    print("🌐 URL: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the application")
    print("-" * 50)
    
    try:
        # Launch Streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run", app_file,
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false",
            "--server.port", "8501"
        ] + config_args
        
        # Wait a moment then open browser
        def open_browser():
            time.sleep(3)
            webbrowser.open("http://localhost:8501")
        
        import threading
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Run the app
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
        return True
    except Exception as e:
        print(f"❌ Error launching application: {e}")
        return False

def main():
    """Main launcher function."""
    print("🌍 Tourism Analytics Intelligence - App Launcher")
    print("=" * 50)
    
    # Check dependencies
    print("📦 Checking dependencies...")
    try:
        check_dependencies()
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        print("Please install manually: pip install streamlit plotly pandas streamlit-option-menu")
        return
    
    # Launch the app
    print("\n🔄 Starting application...")
    success = launch_app()
    
    if success:
        print("\n✅ Application launched successfully!")
    else:
        print("\n❌ Failed to launch application")
        print("Please check the error messages above and try again.")

if __name__ == "__main__":
    main() 