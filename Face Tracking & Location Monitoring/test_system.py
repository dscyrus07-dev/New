#!/usr/bin/env python3
"""
Test Script for Face Tracking & Location Monitoring System

This script tests the core functionality of both modules to ensure
everything is working correctly before running the main application.
"""

import sys
import os
import numpy as np
import cv2
import logging

# Add modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all required modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        # Core libraries
        import cv2
        import numpy as np
        import yaml
        print("✅ Core libraries imported successfully")
        
        # Face tracking modules
        from modules.face_tracking import FaceTracker, ZoneMonitor, TrackingDatabase, TrackingVisualizer
        print("✅ Face tracking modules imported successfully")
        
        # Behavior analysis modules  
        from modules.behavior_analysis import BehaviorDetector, DrowsinessAnalyzer, BehaviorVisualizer
        print("✅ Behavior analysis modules imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_face_tracker():
    """Test face tracker initialization and basic functionality"""
    print("\n🔍 Testing Face Tracker...")
    
    try:
        from modules.face_tracking import FaceTracker
        
        # Initialize tracker
        tracker = FaceTracker(
            max_age=30,
            min_hits=3,
            face_model="hog",  # Use HOG for CPU testing
            recognition_threshold=0.6
        )
        
        # Create test frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Test detection (should return empty list for blank frame)
        detections = tracker.detect_faces(test_frame)
        tracks = tracker.update(test_frame, detections)
        
        fps = tracker.get_fps()
        active_count = tracker.get_active_tracks_count()
        
        print(f"✅ Face tracker initialized - FPS: {fps:.1f}, Active tracks: {active_count}")
        return True
        
    except Exception as e:
        print(f"❌ Face tracker test failed: {e}")
        return False

def test_zone_monitor():
    """Test zone monitor functionality"""
    print("\n📍 Testing Zone Monitor...")
    
    try:
        from modules.face_tracking import ZoneMonitor
        
        # Test zone configuration
        zones_config = {
            'test_zone': {
                'name': 'Test Zone',
                'color': [0, 255, 0],
                'points': [[100, 100], [200, 100], [200, 200], [100, 200]]
            }
        }
        
        # Initialize monitor
        monitor = ZoneMonitor(zones_config)
        
        # Test with empty tracks
        transitions = monitor.update([])
        occupancy = monitor.get_all_occupancy()
        stats = monitor.get_zone_stats()
        
        print(f"✅ Zone monitor initialized - Zones: {len(monitor.zones)}, Transitions: {len(transitions)}")
        return True
        
    except Exception as e:
        print(f"❌ Zone monitor test failed: {e}")
        return False

def test_behavior_detector():
    """Test behavior detector functionality"""
    print("\n😴 Testing Behavior Detector...")
    
    try:
        from modules.behavior_analysis import BehaviorDetector
        
        # Initialize detector
        detector = BehaviorDetector(
            ear_threshold=0.25,
            yawn_threshold=20.0
        )
        
        # Create test frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Test detection (should return empty list for blank frame)
        behaviors = detector.detect_behaviors(test_frame)
        fps = detector.get_fps()
        
        print(f"✅ Behavior detector initialized - FPS: {fps:.1f}, Behaviors detected: {len(behaviors)}")
        return True
        
    except Exception as e:
        print(f"❌ Behavior detector test failed: {e}")
        return False

def test_drowsiness_analyzer():
    """Test drowsiness analyzer functionality"""
    print("\n💤 Testing Drowsiness Analyzer...")
    
    try:
        from modules.behavior_analysis import DrowsinessAnalyzer
        
        # Initialize analyzer
        analyzer = DrowsinessAnalyzer(
            time_window=30,
            drowsiness_threshold=60.0
        )
        
        # Test basic functionality
        stats = analyzer.get_alert_statistics(1)  # Non-existent face
        
        print("✅ Drowsiness analyzer initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Drowsiness analyzer test failed: {e}")
        return False

def test_database():
    """Test database functionality"""
    print("\n🗄️ Testing Database...")
    
    try:
        from modules.face_tracking import TrackingDatabase
        
        # Initialize database
        db = TrackingDatabase("test_tracking.db")
        
        # Test basic operations
        stats = db.get_database_stats()
        
        # Cleanup test database
        if os.path.exists("test_tracking.db"):
            os.remove("test_tracking.db")
        
        print("✅ Database operations successful")
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def test_visualizers():
    """Test visualization components"""
    print("\n🎨 Testing Visualizers...")
    
    try:
        from modules.face_tracking import TrackingVisualizer
        from modules.behavior_analysis import BehaviorVisualizer
        
        # Initialize visualizers
        tracking_viz = TrackingVisualizer()
        behavior_viz = BehaviorVisualizer()
        
        # Test with dummy frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Test basic visualization (should not crash)
        output_frame = tracking_viz.draw_statistics_overlay(test_frame, {
            'active_tracks': 0,
            'fps': 0.0,
            'total_transitions': 0,
            'known_faces': 0,
            'zones': 0,
            'uptime': '00:00:00'
        })
        
        print("✅ Visualizers initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Visualizer test failed: {e}")
        return False

def test_camera_access():
    """Test camera access"""
    print("\n📹 Testing Camera Access...")
    
    try:
        # Try to open default camera
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✅ Camera access successful - Frame shape: {frame.shape}")
                cap.release()
                return True
            else:
                print("⚠️ Camera opened but no frame received")
                cap.release()
                return False
        else:
            print("⚠️ No camera available (this is okay for testing)")
            return True
            
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False

def test_configuration():
    """Test configuration loading"""
    print("\n⚙️ Testing Configuration...")
    
    try:
        import yaml
        
        config_path = "config/config.yaml"
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            print("✅ Configuration file loaded successfully")
        else:
            print("⚠️ Configuration file not found (using defaults)")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Face Tracking & Location Monitoring System - Test Suite")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Face Tracker Test", test_face_tracker),
        ("Zone Monitor Test", test_zone_monitor),
        ("Behavior Detector Test", test_behavior_detector),
        ("Drowsiness Analyzer Test", test_drowsiness_analyzer),
        ("Database Test", test_database),
        ("Visualizers Test", test_visualizers),
        ("Camera Access Test", test_camera_access),
        ("Configuration Test", test_configuration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
    
    print("\n" + "=" * 60)
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        print("\nTo run the system:")
        print("  python main.py")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the error messages above.")
        print("\nCommon solutions:")
        print("  • Install missing dependencies: pip install -r requirements.txt")
        print("  • Check camera permissions")
        print("  • Ensure all directories exist")
        return 1

if __name__ == "__main__":
    exit(main())