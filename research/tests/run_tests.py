#!/usr/bin/env python3
"""
Test runner script for the sentiment analyzer project.
Runs all test modules in the tests directory.
"""

import os
import sys
import importlib.util
import traceback

def discover_test_modules():
    """Discover all test modules in the tests directory."""
    test_modules = []
    
    for filename in os.listdir(os.path.dirname(__file__)):
        if filename.startswith('test_') and filename.endswith('.py') and filename != 'run_tests.py':
            module_name = filename[:-3]  # Remove .py extension
            test_modules.append(module_name)
    
    return test_modules

def import_test_module(module_name):
    """Import a test module by name."""
    module_path = os.path.join(os.path.dirname(__file__), f"{module_name}.py")
    
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    return module

def run_test_module(module):
    """Run a test module's main function."""
    if hasattr(module, 'main'):
        try:
            print(f"Running {module.__name__}...")
            module.main()
            print(f"✅ {module.__name__} completed successfully")
            return True
        except Exception as e:
            print(f"❌ {module.__name__} failed with error: {str(e)}")
            traceback.print_exc()
            return False
    else:
        print(f"⚠️ {module.__name__} has no main function to run")
        return False

def run_all_tests():
    """Run all discovered test modules."""
    print("=" * 60)
    print("Sentiment Analyzer Test Runner")
    print("=" * 60)
    
    test_modules = discover_test_modules()
    print(f"Discovered {len(test_modules)} test modules to run")
    
    successful_tests = 0
    failed_tests = 0
    
    for module_name in test_modules:
        print("\n" + "-" * 60)
        module = import_test_module(module_name)
        
        if run_test_module(module):
            successful_tests += 1
        else:
            failed_tests += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {successful_tests} passed, {failed_tests} failed")
    print("=" * 60)
    
    return failed_tests == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)