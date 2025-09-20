#!/usr/bin/env python3
"""
Test script to verify that the Flask routes are properly defined
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_routes():
    """Test that all required routes are defined"""
    try:
        # Import the app
        import app
        
        print("✅ App imported successfully")
        
        # Get the Flask app instance
        flask_app = app.app
        
        print("✅ Flask app instance found")
        
        # Get all routes
        routes = []
        for rule in flask_app.url_map.iter_rules():
            routes.append(rule.endpoint)
        
        print(f"📋 Found {len(routes)} routes:")
        for route in sorted(routes):
            print(f"  - {route}")
        
        # Check for specific routes that were causing issues
        required_routes = [
            'attendance',
            'attendance_today', 
            'export_data',
            'student_photo'
        ]
        
        print("\n🔍 Checking required routes:")
        missing_routes = []
        for route in required_routes:
            if route in routes:
                print(f"  ✅ {route}")
            else:
                print(f"  ❌ {route} - MISSING")
                missing_routes.append(route)
        
        if missing_routes:
            print(f"\n❌ Missing routes: {missing_routes}")
            return False
        else:
            print(f"\n✅ All required routes are present!")
            return True
            
    except Exception as e:
        print(f"❌ Error testing routes: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing Flask routes...")
    success = test_routes()
    if success:
        print("\n🎉 Route test PASSED!")
        sys.exit(0)
    else:
        print("\n💥 Route test FAILED!")
        sys.exit(1)
