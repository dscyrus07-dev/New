#!/usr/bin/env python3
"""
Face Tracking & Location Monitoring System

Main application that integrates face tracking with DeepSort and 
facial behavior analysis for comprehensive monitoring.
"""

import cv2
import numpy as np
import yaml
import argparse
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional
import uuid
import sys
import os

# Add modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from modules.face_tracking import FaceTracker, ZoneMonitor, TrackingDatabase, TrackingVisualizer
from modules.behavior_analysis import BehaviorDetector, DrowsinessAnalyzer, BehaviorVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/logs/system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FaceTrackingSystem:
    """
    Integrated Face Tracking & Location Monitoring System
    
    Combines real-time face tracking, zone monitoring, and behavior analysis
    for comprehensive surveillance and monitoring applications.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the tracking system.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Generate session ID
        self.session_id = str(uuid.uuid4())
        
        # Initialize components
        self._init_components()
        
        # System state
        self.running = False
        self.start_time = time.time()
        self.frame_count = 0
        
        logger.info(f"FaceTrackingSystem initialized with session ID: {self.session_id}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            # Return default configuration
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'system': {'debug': False, 'log_level': 'INFO', 'fps_target': 10, 'gpu_enabled': True},
            'camera': {'source': 0, 'width': 1280, 'height': 720, 'fps': 30},
            'face_detection': {'model': 'hog', 'scale_factor': 0.5, 'confidence_threshold': 0.6},
            'tracking': {'max_age': 30, 'min_hits': 3, 'iou_threshold': 0.3},
            'zones': {},
            'behavior': {'eye_aspect_ratio_threshold': 0.25, 'eye_closure_frames': 16},
            'database': {'path': 'data/tracking.db'},
            'logging': {'log_dir': 'data/logs'}
        }
    
    def _init_components(self):
        """Initialize all system components"""
        # Face tracker
        self.face_tracker = FaceTracker(
            max_age=self.config['tracking']['max_age'],
            min_hits=self.config['tracking']['min_hits'],
            iou_threshold=self.config['tracking']['iou_threshold'],
            face_model=self.config['face_detection']['model'],
            recognition_threshold=self.config['face_detection']['confidence_threshold']
        )
        
        # Zone monitor
        self.zone_monitor = ZoneMonitor(self.config.get('zones', {}))
        
        # Database
        self.database = TrackingDatabase(self.config['database']['path'])
        
        # Visualizers
        self.tracking_visualizer = TrackingVisualizer()
        
        # Behavior analysis components
        self.behavior_detector = BehaviorDetector(
            ear_threshold=self.config['behavior']['eye_aspect_ratio_threshold'],
            ear_consecutive_frames=self.config['behavior']['eye_closure_frames']
        )
        
        self.drowsiness_analyzer = DrowsinessAnalyzer()
        self.behavior_visualizer = BehaviorVisualizer()
        
        logger.info("All system components initialized")
    
    def load_known_faces(self, faces_dir: str = "../Facial Recognition/employee_photos"):
        """
        Load known faces from the facial recognition system.
        
        Args:
            faces_dir: Directory containing employee photos
        """
        import face_recognition
        import pickle
        import os
        
        known_encodings = []
        known_names = []
        
        try:
            # Try to load from existing database first
            db_path = os.path.join(faces_dir, "..", "surveillance.db")
            if os.path.exists(db_path):
                import sqlite3
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                cursor.execute("SELECT name, face_encoding FROM employees WHERE face_encoding IS NOT NULL")
                for row in cursor.fetchall():
                    name, encoding_blob = row
                    if encoding_blob:
                        encoding = pickle.loads(encoding_blob)
                        known_encodings.append(encoding)
                        known_names.append(name)
                
                conn.close()
                logger.info(f"Loaded {len(known_encodings)} known faces from database")
            
            # Fallback: load from photo directory
            elif os.path.exists(faces_dir):
                for filename in os.listdir(faces_dir):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_path = os.path.join(faces_dir, filename)
                        image = face_recognition.load_image_file(image_path)
                        encodings = face_recognition.face_encodings(image)
                        
                        if encodings:
                            known_encodings.append(encodings[0])
                            known_names.append(os.path.splitext(filename)[0])
                
                logger.info(f"Loaded {len(known_encodings)} known faces from photos")
            
            # Load into face tracker
            self.face_tracker.load_known_faces(known_encodings, known_names)
            
        except Exception as e:
            logger.error(f"Failed to load known faces: {e}")
    
    def run(self, camera_source: Optional[str] = None, display: bool = True):
        """
        Run the face tracking system.
        
        Args:
            camera_source: Camera source (0 for webcam, RTSP URL for IP camera)
            display: Whether to display video output
        """
        # Determine camera source
        if camera_source is None:
            camera_source = self.config['camera']['source']
        
        # Initialize camera
        cap = cv2.VideoCapture(camera_source)
        
        if not cap.isOpened():
            logger.error(f"Failed to open camera source: {camera_source}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['camera']['width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['camera']['height'])
        cap.set(cv2.CAP_PROP_FPS, self.config['camera']['fps'])
        
        self.running = True
        self.start_time = time.time()
        
        logger.info("Face tracking system started")
        
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    break
                
                # Process frame
                processed_frame = self._process_frame(frame)
                
                # Display frame if requested
                if display:
                    cv2.imshow('Face Tracking & Behavior Analysis System', processed_frame)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        self._save_screenshot(processed_frame)
                    elif key == ord('r'):
                        self._reset_system()
                    elif key == ord('c'):
                        self._calibrate_behavior()
                
                # Update frame counter
                self.frame_count += 1
                
                # Periodic cleanup
                if self.frame_count % 1000 == 0:
                    self._periodic_cleanup()
        
        except KeyboardInterrupt:
            logger.info("System interrupted by user")
        
        finally:
            self._shutdown(cap, display)
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame through the complete pipeline.
        
        Args:
            frame: Input frame
            
        Returns:
            Processed frame with all visualizations
        """
        # 1. Face detection and tracking
        detections = self.face_tracker.detect_faces(
            frame, 
            scale_factor=self.config['face_detection']['scale_factor']
        )
        
        tracks = self.face_tracker.update(frame, detections)
        
        # 2. Zone monitoring
        zone_transitions = self.zone_monitor.update(tracks)
        
        # 3. Behavior analysis
        behavior_data_list = []
        drowsiness_scores = []
        
        for track in tracks:
            # Extract face region for behavior analysis
            x, y, w, h = track.bbox
            face_boxes = [(x, y, w, h)]
            
            # Detect behaviors
            behaviors = self.behavior_detector.detect_behaviors(frame, face_boxes)
            
            if behaviors:
                behavior_data = behaviors[0]  # Take first match
                behavior_data_list.append(behavior_data)
                
                # Analyze drowsiness
                drowsiness_score = self.drowsiness_analyzer.analyze_drowsiness(
                    track.track_id, behavior_data
                )
                drowsiness_scores.append(drowsiness_score)
                
                # Store in database
                self.database.save_track(track, self.session_id)
        
        # Store zone transitions
        for transition in zone_transitions:
            self.database.save_zone_transition(transition, self.session_id)
        
        # 4. Visualization
        output_frame = frame.copy()
        
        # Draw tracking visualization
        output_frame = self.tracking_visualizer.draw_tracks(output_frame, tracks)
        
        # Draw zones
        zones = self.zone_monitor.zones
        occupancy = self.zone_monitor.get_all_occupancy()
        output_frame = self.tracking_visualizer.draw_zones_with_occupancy(
            output_frame, zones, occupancy
        )
        
        # Draw behavior analysis
        for i, behavior_data in enumerate(behavior_data_list):
            drowsiness_score = drowsiness_scores[i] if i < len(drowsiness_scores) else None
            output_frame = self.behavior_visualizer.draw_behavior_analysis(
                output_frame, behavior_data, drowsiness_score
            )
        
        # Draw behavior dashboard
        if behavior_data_list:
            output_frame = self.behavior_visualizer.draw_behavior_dashboard(
                output_frame, behavior_data_list, drowsiness_scores
            )
        
        # Draw system statistics
        stats = self._get_system_stats(tracks, zone_transitions, behavior_data_list, drowsiness_scores)
        output_frame = self.tracking_visualizer.draw_statistics_overlay(output_frame, stats)
        
        # Draw recent zone transitions
        recent_transitions = self.zone_monitor.get_recent_transitions(minutes=5)
        output_frame = self.tracking_visualizer.draw_zone_transitions_log(
            output_frame, recent_transitions
        )
        
        # Update FPS counters
        self.face_tracker.update_fps()
        self.behavior_detector._update_fps()
        
        return output_frame
    
    def _get_system_stats(self, tracks, zone_transitions, behavior_data_list, drowsiness_scores) -> Dict:
        """Get current system statistics"""
        uptime = time.time() - self.start_time
        uptime_str = f"{int(uptime // 3600):02d}:{int((uptime % 3600) // 60):02d}:{int(uptime % 60):02d}"
        
        # Count drowsiness alerts
        drowsy_count = sum(1 for score in drowsiness_scores if score.is_drowsy)
        severe_count = sum(1 for score in drowsiness_scores if score.alert_level == "Severe")
        
        return {
            'active_tracks': len(tracks),
            'fps': self.face_tracker.get_fps(),
            'behavior_fps': self.behavior_detector.get_fps(),
            'total_transitions': len(zone_transitions),
            'known_faces': len(self.face_tracker.known_faces),
            'zones': len(self.zone_monitor.zones),
            'uptime': uptime_str,
            'frame_count': self.frame_count,
            'drowsy_alerts': drowsy_count,
            'severe_alerts': severe_count,
            'faces_analyzed': len(behavior_data_list)
        }
    
    def _save_screenshot(self, frame: np.ndarray):
        """Save current frame as screenshot"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/screenshots/screenshot_{timestamp}.jpg"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        cv2.imwrite(filename, frame)
        logger.info(f"Screenshot saved: {filename}")
    
    def _reset_system(self):
        """Reset system state"""
        self.face_tracker.cleanup_old_tracks()
        self.zone_monitor.cleanup_old_data(hours=1)
        self.behavior_detector.reset_face_tracking()
        self.drowsiness_analyzer.cleanup_old_data(hours=1)
        logger.info("System state reset")
    
    def _calibrate_behavior(self):
        """Calibrate behavior analysis baselines"""
        # Calibrate for all currently tracked faces
        for track_id in self.face_tracker.active_tracks.keys():
            self.drowsiness_analyzer.calibrate_baseline(track_id)
        logger.info("Behavior analysis calibrated")
    
    def _periodic_cleanup(self):
        """Perform periodic cleanup tasks"""
        self.face_tracker.cleanup_old_tracks()
        self.zone_monitor.cleanup_old_data()
        self.behavior_detector.cleanup_old_tracking_data()
        self.drowsiness_analyzer.cleanup_old_data()
        self.database.cleanup_old_data()
        logger.debug("Periodic cleanup completed")
    
    def _shutdown(self, cap, display: bool):
        """Shutdown system gracefully"""
        self.running = False
        
        if cap:
            cap.release()
        
        if display:
            cv2.destroyAllWindows()
        
        # Final cleanup
        self._periodic_cleanup()
        
        # Log final statistics
        uptime = time.time() - self.start_time
        logger.info(f"System shutdown - Uptime: {uptime:.1f}s, Frames processed: {self.frame_count}")

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description='Face Tracking & Location Monitoring System')
    
    parser.add_argument('--config', '-c', default='config/config.yaml',
                       help='Configuration file path')
    parser.add_argument('--camera', '-cam', default=None,
                       help='Camera source (0 for webcam, RTSP URL for IP camera)')
    parser.add_argument('--no-display', action='store_true',
                       help='Run without video display (headless mode)')
    parser.add_argument('--load-faces', default="../Facial Recognition/employee_photos",
                       help='Directory to load known faces from')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create necessary directories
    os.makedirs('data/logs', exist_ok=True)
    os.makedirs('data/screenshots', exist_ok=True)
    os.makedirs('data/tracking_data', exist_ok=True)
    
    print("🔍 Face Tracking & Location Monitoring System")
    print("=" * 50)
    print("Features:")
    print("  • Real-time face tracking with DeepSort")
    print("  • Zone monitoring and transition logging")
    print("  • Facial behavior analysis (eye state, head pose, yawning)")
    print("  • Drowsiness detection and alerting")
    print("  • Movement pattern analysis")
    print("  • Comprehensive data logging and visualization")
    print("=" * 50)
    print("Controls:")
    print("  • 'q' - Quit system")
    print("  • 's' - Save screenshot")
    print("  • 'r' - Reset system state")
    print("  • 'c' - Calibrate behavior analysis")
    print("=" * 50)
    
    try:
        # Initialize system
        system = FaceTrackingSystem(args.config)
        
        # Load known faces
        if args.load_faces:
            system.load_known_faces(args.load_faces)
        
        # Run system
        system.run(
            camera_source=args.camera,
            display=not args.no_display
        )
        
    except Exception as e:
        logger.error(f"System error: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())