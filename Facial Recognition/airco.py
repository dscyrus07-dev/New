#!/usr/bin/env python3
"""
AirCo Secure Surveillance System
Complete implementation with employee registration and real-time monitoring
"""

import cv2
import face_recognition
import numpy as np
import sqlite3
import pickle
import os
import json
import threading
import time
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
import logging
from dataclasses import dataclass, field
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from collections import defaultdict
import dlib
import torch
from torch import nn
from torchvision import transforms
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('surveillance.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Check for GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using device: {device}")

# Configure face recognition settings
FACE_DETECTION_MODEL = 'cnn' if device == 'cuda' else 'hog'  # Use CNN for GPU, HOG for CPU
FACE_ENCODING_MODEL = 'large'  # More accurate but slower than 'small'
FACE_NUM_JITTERS = 1  # Number of times to re-sample the face for encoding (higher = more accurate but slower)
FACE_DETECTION_SCALE = 0.5  # Scale factor for face detection (faster processing)

# Face alignment settings
FACE_LANDMARKS_PATH = "shape_predictor_68_face_landmarks.dat"
if not os.path.exists(FACE_LANDMARKS_PATH):
    logger.warning(f"Face landmarks file not found at {FACE_LANDMARKS_PATH}. Face alignment will be disabled.")
    face_aligner = None
else:
    face_aligner = dlib.shape_predictor(FACE_LANDMARKS_PATH)

@dataclass
class Employee:
    """Employee data structure with multiple face encodings"""
    id: str
    name: str
    department: str
    position: str
    face_encodings: List[np.ndarray] = field(default_factory=list)  # Store multiple encodings
    phone: str = ""
    email: str = ""
    is_active: bool = True

    def add_face_encoding(self, encoding: np.ndarray, max_encodings: int = 10):
        """Add a new face encoding, keeping only the most recent ones"""
        self.face_encodings.append(encoding)
        if len(self.face_encodings) > max_encodings:
            self.face_encodings = self.face_encodings[-max_encodings:]

    def get_average_encoding(self) -> np.ndarray:
        """Get the average of all face encodings"""
        if not self.face_encodings:
            return None
        return np.mean(self.face_encodings, axis=0)

    def get_all_encodings(self) -> List[np.ndarray]:
        """Get all face encodings"""
        return self.face_encodings

@dataclass
class DetectionResult:
    """Face detection result structure"""
    employee_id: str
    name: str
    confidence: float
    face_location: Tuple[int, int, int, int]
    timestamp: datetime
    is_known: bool

class DatabaseManager:
    """Enhanced database manager for surveillance system"""

    def __init__(self, db_path: str = "surveillance.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database with all required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Employees table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS employees (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    department TEXT,
                    position TEXT,
                    face_encodings BLOB,
                    phone TEXT,
                    email TEXT,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Attendance logs
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    employee_id TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    action TEXT DEFAULT 'entry',
                    confidence REAL,
                    image_path TEXT,
                    FOREIGN KEY (employee_id) REFERENCES employees (id)
                )
            ''')

            # Unknown faces alerts
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS unknown_faces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_path TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    alert_sent BOOLEAN DEFAULT FALSE,
                    notes TEXT
                )
            ''')

            # System settings
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            ''')

            conn.commit()
            logger.info("Database initialized successfully")

    def add_employee(self, employee: Employee) -> bool:
        """Add new employee to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Ensure face_encodings is a list of numpy arrays
                if employee.face_encodings is None:
                    encoding_blob = None
                else:
                    # Convert numpy arrays to lists for consistent serialization
                    encodings_to_store = [e.tolist() if hasattr(e, 'tolist') else e 
                                        for e in employee.face_encodings]
                    encoding_blob = pickle.dumps(encodings_to_store)

                cursor.execute('''
                    INSERT INTO employees (id, name, department, position, face_encodings, phone, email)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (employee.id, employee.name, employee.department, employee.position,
                      encoding_blob, employee.phone, employee.email))

                conn.commit()
                logger.info(f"Employee {employee.name} added successfully")
                return True

        except sqlite3.IntegrityError:
            logger.error(f"Employee {employee.id} already exists")
            return False
        except Exception as e:
            logger.error(f"Error adding employee: {str(e)}")
            return False

    def get_all_employees(self) -> List[Employee]:
        """Get all active employees"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM employees WHERE is_active = TRUE')
                rows = cursor.fetchall()

                employees = []
                for row in rows:
                    face_encodings = None
                    if row[4]:  # If there are face encodings
                        try:
                            # Load the pickled encodings and convert back to numpy arrays
                            encodings_list = pickle.loads(row[4])
                            if isinstance(encodings_list, list):
                                face_encodings = [np.array(e) for e in encodings_list 
                                                if e is not None]
                            else:
                                # Handle case where a single encoding was stored
                                face_encodings = [np.array(encodings_list)]
                        except Exception as e:
                            logger.error(f"Error loading face encodings: {str(e)}")
                            face_encodings = None
                    
                    employee = Employee(
                        id=row[0],
                        name=row[1],
                        department=row[2],
                        position=row[3],
                        face_encodings=face_encodings or [],
                        phone=row[5],
                        email=row[6],
                        is_active=row[7]
                    )
                    employees.append(employee)

                return employees

        except Exception as e:
            logger.error(f"Error retrieving employees: {str(e)}")
            return []

    def log_attendance(self, employee_id: str, confidence: float, image_path: str = None):
        """Log employee attendance"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO attendance (employee_id, confidence, image_path)
                    VALUES (?, ?, ?)
                ''', (employee_id, confidence, image_path))
                conn.commit()
                logger.info(f"Attendance logged for {employee_id}")

        except Exception as e:
            logger.error(f"Error logging attendance: {str(e)}")

    def log_unknown_face(self, image_path: str):
        """Log unknown face detection"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO unknown_faces (image_path)
                    VALUES (?)
                ''', (image_path,))
                conn.commit()
                logger.info(f"Unknown face logged: {image_path}")

        except Exception as e:
            logger.error(f"Error logging unknown face: {str(e)}")

    def get_today_attendance(self) -> List[Dict]:
        """Get today's attendance records"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                today = datetime.now().strftime('%Y-%m-%d')
                cursor.execute('''
                    SELECT a.*, e.name FROM attendance a
                    JOIN employees e ON a.employee_id = e.id
                    WHERE date(a.timestamp) = ?
                    ORDER BY a.timestamp DESC
                ''', (today,))

                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Error retrieving attendance: {str(e)}")
            return []

class EmployeeRegistration:
    """Employee registration system"""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def register_employee(self, employee_id: str, name: str, department: str,
                         position: str, phone: str = "", email: str = ""):
        """Register new employee with face capture"""

        print(f"\n=== Employee Registration ===")
        print(f"ID: {employee_id}")
        print(f"Name: {name}")
        print(f"Department: {department}")
        print(f"Position: {position}")
        print("\nStarting face capture...")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return False

        face_encodings = []
        capture_count = 0
        required_captures = 5

        print(f"Please look at the camera and press SPACE to capture ({required_captures} photos needed)")
        print("Press 'q' to quit")

        while capture_count < required_captures:
            ret, frame = cap.read()
            if not ret:
                continue

            # Display instructions
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Captures: {capture_count}/{required_captures}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, "Press SPACE to capture, 'q' to quit",
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, "Look straight at camera",
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow('Employee Registration', display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):  # Space key
                # Detect and encode face
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)

                if len(face_locations) == 1:  # Exactly one face
                    face_encodings_frame = face_recognition.face_encodings(rgb_frame, face_locations)
                    if face_encodings_frame:
                        face_encodings.append(face_encodings_frame[0])
                        capture_count += 1
                        print(f" Captured face {capture_count}/{required_captures}")

                        # Save the capture
                        os.makedirs("employee_photos", exist_ok=True)
                        cv2.imwrite(f"employee_photos/{employee_id}_{capture_count}.jpg", frame)

                        # Brief pause
                        time.sleep(0.5)
                elif len(face_locations) == 0:
                    print("No face detected. Please position yourself in front of the camera.")
                else:
                    print("Multiple faces detected. Please ensure only one person is in frame.")

            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        if len(face_encodings) >= 3:  # Minimum 3 captures
            # Create employee object
            employee = Employee(
                id=employee_id,
                name=name,
                department=department,
                position=position,
                face_encodings=face_encodings,
                phone=phone,
                email=email
            )

            # Save to database
            if self.db_manager.add_employee(employee):
                print(f" Employee {name} registered successfully with {len(face_encodings)} face samples!")
                return True
            else:
                print(" Error saving employee to database")
                return False
        else:
            print(f" Not enough face samples captured ({len(face_encodings)}/{required_captures})")
            return False

class AlertSystem:
    """Alert system for unknown faces"""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.email_enabled = self.config.get('email_enabled', False)
        self.email_settings = self.config.get('email_settings', {})

    def send_alert(self, image_path: str, timestamp: datetime):
        """Send alert for unknown face"""
        message = f" SECURITY ALERT: Unknown person detected at {timestamp.strftime('%Y-%m-%d %H:%M:%S')}"

        print(f"\n{message}")
        print(f"Image saved: {image_path}")

        # Save alert to file
        with open("alerts.log", "a") as f:
            f.write(f"{timestamp.isoformat()}: Unknown face detected - {image_path}\n")

        # Email alert (if configured)
        if self.email_enabled:
            self._send_email_alert(message, image_path)

    def _send_email_alert(self, message: str, image_path: str):
        """Send email alert (placeholder - configure as needed)"""
        try:
            # Configure your email settings here
            pass
        except Exception as e:
            logger.error(f"Error sending email alert: {str(e)}")

class SurveillanceSystem:
    """Main surveillance system"""

    def __init__(self, camera_source: str = '0', db_path: str = 'surveillance.db',
                 recognition_threshold: float = 0.6, 
                 face_detection_model: str = 'cnn',  # 'cnn' or 'hog'
                 enable_gpu: bool = True):
        """
        Initialize the surveillance system.
        
        Args:
            camera_source: Can be an integer for webcam (0, 1, etc.) or an RTSP URL
            db_path: Path to the SQLite database file
            recognition_threshold: Threshold for face recognition (lower is more strict)
            face_detection_model: 'cnn' (more accurate, slower) or 'hog' (faster, less accurate)
            enable_gpu: Whether to use GPU acceleration if available
        """
        self.recognition_threshold = recognition_threshold

        # Initialize components
        self.db_manager = DatabaseManager()
        self.alert_system = AlertSystem()

        # Load known faces
        self.known_faces = {}
        self.load_known_faces()

        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.attendance_cooldown = {}  # Prevent spam attendance logs
        self.processing_times = []

        # Create directories
        os.makedirs("unknown_faces", exist_ok=True)
        os.makedirs("attendance_photos", exist_ok=True)
        os.makedirs("employee_photos", exist_ok=True)

        # Initialize face detection models
        self.face_detector = dlib.get_frontal_face_detector()

    def load_known_faces(self):
        """Load known faces from database with support for multiple encodings"""
        employees = self.db_manager.get_all_employees()
        self.known_faces = {}

        for employee in employees:
            if employee.face_encodings is not None and len(employee.face_encodings) > 0:
                encodings = employee.face_encodings
                self.known_faces[employee.id] = {
                    'name': employee.name,
                    'department': employee.department,
                    'position': employee.position,
                    'encoding': encodings[0],  # Keep first encoding for backward compatibility
                    'all_encodings': encodings  # Store all encodings for better matching
                }

        logger.info(f"Loaded {len(self.known_faces)} known faces")

    def _align_face(self, image: np.ndarray, face_location: Tuple[int, int, int, int]) -> np.ndarray:
        """Align face using facial landmarks for better recognition"""
        if face_aligner is None:
            return image

        top, right, bottom, left = face_location
        face_rect = dlib.rectangle(left, top, right, bottom)

        # Convert to dlib format and get landmarks
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        landmarks = face_aligner(gray, face_rect)

        # Simple alignment: rotate to align eyes
        left_eye = (landmarks.part(36).x, landmarks.part(36).y)
        right_eye = (landmarks.part(45).x, landmarks.part(45).y)

        # Calculate angle between the eye corners
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))

        # Get rotation matrix and apply affine transformation
        center = ((left + right) // 2, (top + bottom) // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned_face = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)

        return aligned_face

    def recognize_faces(self, frame: np.ndarray) -> List[DetectionResult]:
        """Recognize faces in frame with GPU acceleration and improved accuracy"""
        if not self.known_faces:
            return []
        
        start_time = time.time()
        
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=FACE_DETECTION_SCALE, fy=FACE_DETECTION_SCALE)
        
        # Convert BGR to RGB
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Find face locations with the selected model (CNN for GPU, HOG for CPU)
        face_locations = face_recognition.face_locations(
            rgb_small_frame,
            model=FACE_DETECTION_MODEL,
            number_of_times_to_upsample=1
        )
        
        # If no faces found, return empty results
        if not face_locations:
            return []
        
        # Scale face locations back to original frame size
        face_locations = [(
            int(top / FACE_DETECTION_SCALE), 
            int(right / FACE_DETECTION_SCALE), 
            int(bottom / FACE_DETECTION_SCALE), 
            int(left / FACE_DETECTION_SCALE)
        ) for (top, right, bottom, left) in face_locations]
        
        # Get face encodings with alignment
        face_encodings = []
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        for face_location in face_locations:
            # Align face before encoding
            aligned_face = self._align_face(rgb_frame, face_location)
            
            # Get encoding for aligned face
            encodings = face_recognition.face_encodings(
                aligned_face,
                known_face_locations=[face_location],
                model=FACE_ENCODING_MODEL,
                num_jitters=FACE_NUM_JITTERS
            )
            
            if encodings:
                face_encodings.append(encodings[0])
            else:
                face_encodings.append(None)
        
        results = []
        
        for i, (face_encoding, face_location) in enumerate(zip(face_encodings, face_locations)):
            if face_encoding is None:
                continue
                
            # Compare with all known face encodings
            matches = []
            face_distances = []
            known_ids = []
            
            for emp_id, emp_data in self.known_faces.items():
                # Compare with all stored encodings for this employee
                emp_encodings = emp_data['all_encodings'] if 'all_encodings' in emp_data else [emp_data['encoding']]
                
                # Calculate distances to all stored encodings
                try:
                    # Ensure emp_encodings is a list of numpy arrays
                    encodings_list = [e for e in emp_encodings if e is not None and isinstance(e, np.ndarray)]
                    if not encodings_list:
                        distances = [1.0]  # Default high distance if no valid encodings
                    else:
                        # Convert to numpy array and ensure proper shape
                        encodings_array = np.array(encodings_list)
                        if len(encodings_array.shape) == 1:
                            encodings_array = encodings_array.reshape(1, -1)
                        distances = face_recognition.face_distance(encodings_array, face_encoding)
                    
                    min_distance = np.min(distances) if len(distances) > 0 else 1.0
                except Exception as e:
                    logger.error(f"Error calculating face distance: {str(e)}")
                    min_distance = 1.0  # Default to maximum distance on error
                
                matches.append(min_distance <= self.recognition_threshold)
                face_distances.append(min_distance)
                known_ids.append(emp_id)
            
            # Find the best match
            employee_id = "Unknown"
            name = "Unknown Person"
            is_known = False
            confidence = 0.0
            
            if matches and any(matches):
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    employee_id = known_ids[best_match_index]
                    employee_data = self.known_faces[employee_id]
                    name = employee_data['name']
                    is_known = True
                    confidence = 1.0 - (face_distances[best_match_index] / 2.0)  # Convert to 0-1 scale
                    
                    # Adjust confidence based on number of stored encodings
                    num_encodings = len(employee_data.get('all_encodings', [employee_data['encoding']]))
                    confidence = min(1.0, confidence * (1.0 + 0.1 * min(5, num_encodings - 1)))
            
            result = DetectionResult(
                employee_id=employee_id,
                name=name,
                confidence=confidence,
                face_location=face_location,
                timestamp=datetime.now(),
                is_known=is_known
            )
            
            results.append(result)
        
        logger.debug(f"Face recognition took {time.time() - start_time:.3f} seconds")
        return results

    def process_detections(self, frame: np.ndarray, results: List[DetectionResult]):
        """Process detection results with improved unknown face handling"""
        current_time = datetime.now()
        
        # Calculate FPS
        self.frame_count += 1
        if self.frame_count % 30 == 0:  # Update FPS every 30 frames
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed
            logger.info(f"Processing at {fps:.1f} FPS")
            self.frame_count = 0
            self.start_time = time.time()

        for result in results:
            if result.is_known:
                # Check attendance cooldown (prevent spam)
                cooldown_key = result.employee_id

                if cooldown_key in self.attendance_cooldown:
                    last_log = self.attendance_cooldown[cooldown_key]
                    if (current_time - last_log).seconds < 30:  # 30 second cooldown
                        continue
                
                # Log attendance
                self.attendance_cooldown[cooldown_key] = current_time
                
                # Save attendance photo
                timestamp_str = current_time.strftime("%Y%m%d_%H%M%S")
                photo_path = f"attendance_photos/{result.employee_id}_{timestamp_str}.jpg"
                cv2.imwrite(photo_path, frame)
                
                # Log to database
                self.db_manager.log_attendance(result.employee_id, result.confidence, photo_path)
                
                print(f"✓ Attendance: {result.name} ({result.confidence:.2f})")
            
            else:
                # Unknown face detected
                timestamp_str = current_time.strftime("%Y%m%d_%H%M%S")
                unknown_path = f"unknown_faces/unknown_{timestamp_str}.jpg"
                
                # Save unknown face image
                top, right, bottom, left = result.face_location
                face_image = frame[top:bottom, left:right]
                cv2.imwrite(unknown_path, face_image)
                
                # Log to database
                self.db_manager.log_unknown_face(unknown_path)
                
                # Send alert
                self.alert_system.send_alert(unknown_path, current_time)
    
    def draw_results(self, frame: np.ndarray, results: List[DetectionResult]) -> np.ndarray:
        """Draw detection results on frame"""
        output_frame = frame.copy()
        
        for result in results:
            top, right, bottom, left = result.face_location
            
            # Choose color based on recognition
            if result.is_known:
                color = (0, 255, 0)  # Green for known
                label = f"{result.name} ({result.confidence:.2f})"
            else:
                color = (0, 0, 255)  # Red for unknown
                label = "⚠️ UNKNOWN PERSON"
            
            # Draw bounding box
            cv2.rectangle(output_frame, (left, top), (right, bottom), color, 2)
            
            # Draw label background
            cv2.rectangle(output_frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            
            # Draw label text
            cv2.putText(output_frame, label, (left + 6, bottom - 6),
                       cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
        
        return output_frame
    
    def run(self, rtsp_url: str = None):
        """Run the surveillance system
        
        Args:
            rtsp_url: Optional RTSP URL to use instead of default camera
        """
        print("🚀 Starting AirCo Surveillance System...")
        print("Press 'q' to quit, 'r' to reload faces, 's' to show stats")
        
        # Set up video capture
        if rtsp_url:
            print(f"Connecting to RTSP stream: {rtsp_url}")
            # For RTSP, we need to set some additional parameters for better reliability
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            # RTSP specific settings
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 5)  # Slightly larger buffer for RTSP
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
        else:
            # Default webcam settings
            cap = cv2.VideoCapture(0)  # Default camera
            # Set camera properties for higher FPS
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 60)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer size for lower latency
        
        if not cap.isOpened():
            error_msg = f"Error: Could not open video source"
            if rtsp_url:
                error_msg += f" (RTSP: {rtsp_url})"
            print(error_msg)
            return
        
        # Initialize frame buffer for smooth display
        from collections import deque
        import time
        
        # Buffer for smooth display (smaller buffer for lower latency)
        buffer_size = 15  # 0.25 seconds at 60 FPS
        frame_buffer = deque(maxlen=buffer_size)
        
        # Performance tracking
        last_frame_time = time.time()
        last_processing_time = last_frame_time
        frame_times = deque(maxlen=60)  # Track frame times for FPS calculation
        
        # Initial fill of the buffer (smaller initial fill for faster startup)
        for _ in range(5):
            ret, frame = cap.read()
            if ret:
                frame_buffer.append(frame)
        
        while True:
            # Read new frame
            ret, new_frame = cap.read()
            if not ret:
                continue
            
            # Update frame timing for FPS calculation
            current_time = time.time()
            frame_times.append(current_time)
            
            # Calculate actual display FPS
            display_fps = 0
            if len(frame_times) > 1:
                display_fps = len(frame_times) / (frame_times[-1] - frame_times[0] + 1e-6)
            
            # Use the most recent frame for display (minimal delay)
            frame_buffer.append(new_frame)
            display_frame = frame_buffer[-1]  # Use the most recent frame for display
            
            # Process frames at a consistent rate (every 0.5 seconds)
            if current_time - last_processing_time >= 0.5:  # Process every 0.5 seconds
                self.frame_count += 1
                # Use a separate thread for face recognition to avoid blocking
                processing_frame = display_frame.copy()
                results = self.recognize_faces(processing_frame)
                self.process_detections(processing_frame, results)
                last_processing_time = current_time
            else:
                results = []
            
            # Draw results on the display frame
            output_frame = self.draw_results(display_frame, results)
            
            # Calculate processing FPS
            elapsed = time.time() - self.start_time
            processing_fps = self.frame_count / elapsed if elapsed > 0 else 0
            
            # Display performance info
            cv2.putText(output_frame, f"Display: {display_fps:.1f} FPS", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(output_frame, f"Processing: {processing_fps:.1f} FPS", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(output_frame, f"Known: {len(self.known_faces)}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(output_frame, f"Detected: {len(results)}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('AirCo Surveillance System', output_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("Reloading known faces...")
                self.load_known_faces()
            elif key == ord('s'):
                self.show_stats()
        
        cap.release()
        cv2.destroyAllWindows()
        print("Surveillance system stopped")
    
    def show_stats(self):
        """Show system statistics"""
        attendance_today = self.db_manager.get_today_attendance()
        print(f"\n=== Today's Statistics ===")
        print(f"Total attendance entries: {len(attendance_today)}")
        print(f"Known faces in system: {len(self.known_faces)}")
        print(f"System uptime: {(time.time() - self.start_time) / 3600:.1f} hours")
        print("Recent attendance:")
        for record in attendance_today[:5]:
            print(f"  {record['name']} - {record['timestamp']}")

def main():
    """Main application entry point"""
    print("🏢 AirCo Secure Surveillance System")
    print("=====================================")
    
    db_manager = DatabaseManager()
    
    while True:
        print("\nSelect an option:")
        print("1. Register New Employee")
        print("2. Start Surveillance System")
        print("3. View Today's Attendance")
        print("4. View All Employees")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            # Employee registration
            registration = EmployeeRegistration(db_manager)
            
            print("\n=== Employee Registration ===")
            employee_id = input("Employee ID: ").strip()
            name = input("Full Name: ").strip()
            department = input("Department: ").strip()
            position = input("Position: ").strip()
            phone = input("Phone (optional): ").strip()
            email = input("Email (optional): ").strip()
            
            if employee_id and name:
                registration.register_employee(employee_id, name, department, position, phone, email)
            else:
                print("Employee ID and Name are required!")
        
        elif choice == '2':
            # Start surveillance
            print("\n=== Surveillance Options ===")
            print("1. Use default camera")
            print("2. Use RTSP stream")
            cam_choice = input("Select camera source (1-2): ").strip()
            
            if cam_choice == '1':
                surveillance = SurveillanceSystem()
                surveillance.run()
            elif cam_choice == '2':
                rtsp_url = input("Enter RTSP URL (e.g., rtsp://username:password@ip:port/stream): ").strip()
                if rtsp_url:
                    surveillance = SurveillanceSystem()
                    surveillance.run(rtsp_url=rtsp_url)
                else:
                    print("Error: RTSP URL cannot be empty")
            else:
                print("Invalid choice. Using default camera.")
                surveillance = SurveillanceSystem()
                surveillance.run()
        
        elif choice == '3':
            # View attendance
            attendance = db_manager.get_today_attendance()
            print(f"\n=== Today's Attendance ({len(attendance)} entries) ===")
            for record in attendance:
                print(f"{record['name']} - {record['timestamp']} (Confidence: {record['confidence']:.2f})")
        
        elif choice == '4':
            # View employees
            employees = db_manager.get_all_employees()
            print(f"\n=== All Employees ({len(employees)} total) ===")
            for emp in employees:
                print(f"{emp.id}: {emp.name} - {emp.department} ({emp.position})")
        
        elif choice == '5':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()