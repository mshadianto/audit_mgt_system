#!/usr/bin/env python3
"""
System test for RAG Agentic AI Internal Audit System
"""

import requests
import time
import sys

def test_health_check():
    """Test application health"""
    try:
        response = requests.get("http://localhost:8501/_stcore/health", timeout=10)
        if response.status_code == 200:
            print("âœ… Health check passed")
            return True
        else:
            print(f"âŒ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"âŒ Health check error: {e}")
        return False

def test_main_page():
    """Test main application page"""
    try:
        response = requests.get("http://localhost:8501", timeout=10)
        if response.status_code == 200 and "streamlit" in response.text.lower():
            print("âœ… Main application loads")
            return True
        else:
            print("âŒ Main application not loading")
            return False
    except Exception as e:
        print(f"âŒ Main page error: {e}")
        return False

def main():
    """Run all tests"""
    print("í·ª Testing RAG Agentic AI Internal Audit System...")
    
    tests = [test_health_check, test_main_page]
    passed = 0
    
    for test in tests:
        if test():
            passed += 1
        time.sleep(1)
    
    print(f"\ní³Š Test Results: {passed}/{len(tests)} passed")
    
    if passed == len(tests):
        print("í¾‰ All tests passed!")
        return 0
    else:
        print("âŒ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
