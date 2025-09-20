#!/usr/bin/env python3
"""
Test script for Face Recognition System
Tests attendance logging, face recognition, and results display
"""

import os
import sys
import requests
import json
import time
from datetime import datetime

# Add the app directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import log_attendance, face_system

def test_attendance_logging():
    """Test the attendance logging functionality"""
    print("=" * 50)
    print("TESTING ATTENDANCE LOGGING")
    print("=" * 50)
    
    # Test logging attendance for a student
    test_student_id = "TEST001"
    test_student_name = "Test Student"
    
    print(f"Testing attendance logging for {test_student_name} ({test_student_id})")
    
    # First attempt should succeed
    result1 = log_attendance(test_student_id, test_student_name)
    print(f"First attempt: {'SUCCESS' if result1 else 'FAILED'}")
    
    # Second attempt should fail (already logged today)
    result2 = log_attendance(test_student_id, test_student_name)
    print(f"Second attempt (should fail): {'FAILED as expected' if not result2 else 'UNEXPECTED SUCCESS'}")
    
    # Check if attendance.csv was created and has the entry
    if os.path.exists('attendance.csv'):
        with open('attendance.csv', 'r') as f:
            content = f.read()
            print(f"Attendance CSV content:\n{content}")
    else:
        print("ERROR: attendance.csv was not created!")
    
    return result1 and not result2

def test_student_loading():
    """Test student data loading"""
    print("\n" + "=" * 50)
    print("TESTING STUDENT DATA LOADING")
    print("=" * 50)
    
    # Load students
    face_system.load_students()
    
    print(f"Number of students loaded: {len(face_system.students_data)}")
    
    if face_system.students_data:
        print("Student data:")
        for student_id, data in face_system.students_data.items():
            print(f"  {student_id}: {data['name']} ({data.get('email', 'No email')})")
            
            # Check if embedding file exists
            embedding_path = data.get('embedding_path', '')
            if embedding_path and os.path.exists(embedding_path):
                print(f"    ✓ Embedding file exists: {embedding_path}")
            else:
                print(f"    ✗ Embedding file missing: {embedding_path}")
    else:
        print("No students found in database")
    
    return len(face_system.students_data) > 0

def test_api_endpoints():
    """Test API endpoints"""
    print("\n" + "=" * 50)
    print("TESTING API ENDPOINTS")
    print("=" * 50)
    
    base_url = "http://127.0.0.1:5000"
    
    # Test login endpoint
    print("Testing login...")
    login_data = {"username": "admin", "password": "admin"}
    
    session = requests.Session()
    
    try:
        response = session.post(f"{base_url}/login", data=login_data)
        if response.status_code == 200 or response.status_code == 302:
            print("✓ Login successful")
        else:
            print(f"✗ Login failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Login error: {e}")
        return False
    
    # Test attendance marking endpoint
    print("Testing attendance marking endpoint...")
    attendance_data = {
        "student_id": "TEST002",
        "name": "API Test Student"
    }
    
    try:
        response = session.post(
            f"{base_url}/mark_attendance",
            json=attendance_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Attendance marking API: {result}")
        else:
            print(f"✗ Attendance marking failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Attendance marking error: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("Face Recognition System Test Suite")
    print("=" * 50)
    print(f"Test started at: {datetime.now()}")
    
    results = []
    
    # Test 1: Attendance Logging
    try:
        result = test_attendance_logging()
        results.append(("Attendance Logging", result))
    except Exception as e:
        print(f"ERROR in attendance logging test: {e}")
        results.append(("Attendance Logging", False))
    
    # Test 2: Student Loading
    try:
        result = test_student_loading()
        results.append(("Student Loading", result))
    except Exception as e:
        print(f"ERROR in student loading test: {e}")
        results.append(("Student Loading", False))
    
    # Test 3: API Endpoints
    try:
        result = test_api_endpoints()
        results.append(("API Endpoints", result))
    except Exception as e:
        print(f"ERROR in API endpoints test: {e}")
        results.append(("API Endpoints", False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
