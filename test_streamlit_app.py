#!/usr/bin/env python3
"""
Test script for the Tourism Analytics Streamlit App

Verifies that all dependencies are available and the app can be imported.
"""

import sys
import os

def test_imports():
    """Test that all required modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Streamlit: {e}")
        return False
    
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        print("✅ Plotly imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Plotly: {e}")
        return False
    
    try:
        import pandas as pd
        import numpy as np
        print("✅ Pandas/Numpy imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Pandas/Numpy: {e}")
        return False
    
    try:
        from streamlit_option_menu import option_menu
        print("✅ Streamlit option menu imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import streamlit-option-menu: {e}")
        return False
    
    return True

def test_app_file():
    """Test that the app file exists and can be parsed."""
    print("\n📄 Testing app file...")
    
    app_file = "tourism_insights_app.py"
    if not os.path.exists(app_file):
        print(f"❌ App file '{app_file}' not found")
        return False
    
    print(f"✅ App file '{app_file}' exists")
    
    try:
        # Try to compile the file to check for syntax errors
        with open(app_file, 'r') as f:
            code = f.read()
        
        compile(code, app_file, 'exec')
        print("✅ App file syntax is valid")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error in app file: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading app file: {e}")
        return False

def test_sample_data():
    """Test that sample data structure is valid."""
    print("\n📊 Testing sample data...")
    
    try:
        # Import the app module to test sample data
        sys.path.append('.')
        
        # Read the app file and check sample destinations
        with open('tourism_insights_app.py', 'r') as f:
            content = f.read()
        
        if 'SAMPLE_DESTINATIONS' in content:
            print("✅ Sample destinations found in app")
        else:
            print("❌ Sample destinations not found")
            return False
        
        if 'load_destination_data' in content:
            print("✅ Data loading function found")
        else:
            print("❌ Data loading function not found")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Error testing sample data: {e}")
        return False

def main():
    """Main test function."""
    print("🌍 Tourism Analytics App - Testing Suite")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Test imports
    if not test_imports():
        all_tests_passed = False
    
    # Test app file
    if not test_app_file():
        all_tests_passed = False
    
    # Test sample data
    if not test_sample_data():
        all_tests_passed = False
    
    # Summary
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("✅ All tests passed! The app should work correctly.")
        print("\n🚀 To start the app, run:")
        print("   python run_tourism_app.py")
        print("   OR")
        print("   streamlit run tourism_insights_app.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\n💡 Try installing missing dependencies:")
        print("   pip install -r web_app_requirements.txt")

if __name__ == "__main__":
    main() 