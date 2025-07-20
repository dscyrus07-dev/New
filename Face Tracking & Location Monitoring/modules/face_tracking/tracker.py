"""
Face Tracker with DeepSort Integration

This module implements real-time face tracking using DeepSort algorithm
with face recognition capabilities.
"""

import cv2
import numpy as np
import face_recognition
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import time
from collections import defaultdict
import logging

try:
    from deep_sort_realtime import DeepSort
except ImportError:
    print("Warning: deep_sort_realtime not available. Install with: pip install deep-sort-realtime")
    DeepSort = None

logger = logging.getLogger(__name__)

@dataclass
class Detection:
    """Face detection data structure"""
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    encoding: Optional[np.ndarray] = None
    landmarks: Optional[np.ndarray] = None

@dataclass
class Track:
    """Face track data structure"""
    track_id: int
    bbox: Tuple[int, int, int, int]
    confidence: float
    encoding: Optional[np.ndarray] = None
    identity: Optional[str] = None
    identity_confidence: float = 0.0
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    last_seen: float = field(default_factory=time.time)
    center_history: List[Tuple[int, int]] = field(default_factory=list)

class FaceTracker:
    """
    Advanced face tracker using DeepSort algorithm with face recognition integration.
    Provides unique track IDs for detected faces and maintains tracking across frames.
    """
    
    def __init__(self, 
                 max_age: int = 30,
                 min_hits: int = 3,
                 iou_threshold: float = 0.3,
                 max_cosine_distance: float = 0.2,
                 nn_budget: int = 100,
                 face_model: str = "hog",
                 recognition_threshold: float = 0.6):
        """
        Initialize the face tracker.
        
        Args:
            max_age: Maximum frames to keep track without detection
            min_hits: Minimum detections before creating track
            iou_threshold: IOU threshold for matching
            max_cosine_distance: Maximum cosine distance for feature matching
            nn_budget: Maximum number of features to store
            face_model: Face detection model ("hog" or "cnn")
            recognition_threshold: Threshold for face recognition
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.face_model = face_model
        self.recognition_threshold = recognition_threshold
        
        # Initialize DeepSort tracker
        if DeepSort is not None:
            self.tracker = DeepSort(
                max_age=max_age,
                n_init=min_hits,
                max_cosine_distance=max_cosine_distance,
                nn_budget=nn_budget
            )
        else:
            logger.warning("DeepSort not available, using simple tracker")
            self.tracker = None
            
        # Known faces database
        self.known_encodings: List[np.ndarray] = []
        self.known_names: List[str] = []
        
        # Tracking statistics
        self.active_tracks: Dict[int, Track] = {}
        self.track_history: Dict[int, List[Dict]] = defaultdict(list)
        self.next_track_id = 1
        
        # Performance metrics
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        
        logger.info(f"FaceTracker initialized with model: {face_model}")
    
    def load_known_faces(self, known_encodings: List[np.ndarray], known_names: List[str]):
        """
        Load known face encodings and names for recognition.
        
        Args:
            known_encodings: List of face encodings
            known_names: List of corresponding names
        """
        self.known_encodings = known_encodings
        self.known_names = known_names
        logger.info(f"Loaded {len(known_encodings)} known faces")
    
    def detect_faces(self, frame: np.ndarray, scale_factor: float = 0.5) -> List[Detection]:
        """
        Detect faces in the given frame.
        
        Args:
            frame: Input frame
            scale_factor: Scale factor for faster detection
            
        Returns:
            List of face detections
        """
        # Resize frame for faster processing
        if scale_factor != 1.0:
            small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
        else:
            small_frame = frame
        
        # Convert to RGB
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_frame, model=self.face_model)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        detections = []
        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            # Scale back to original frame size
            if scale_factor != 1.0:
                top = int(top / scale_factor)
                right = int(right / scale_factor)
                bottom = int(bottom / scale_factor)
                left = int(left / scale_factor)
            
            # Convert to (x, y, w, h) format
            bbox = (left, top, right - left, bottom - top)
            
            detection = Detection(
                bbox=bbox,
                confidence=0.9,  # face_recognition doesn't provide confidence
                encoding=encoding
            )
            detections.append(detection)
        
        return detections
    
    def recognize_face(self, encoding: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Recognize a face based on its encoding.
        
        Args:
            encoding: Face encoding to recognize
            
        Returns:
            Tuple of (name, confidence) or (None, 0.0) if not recognized
        """
        if not self.known_encodings:
            return None, 0.0
        
        # Compare with known faces
        distances = face_recognition.face_distance(self.known_encodings, encoding)
        min_distance = np.min(distances)
        
        if min_distance <= self.recognition_threshold:
            best_match_index = np.argmin(distances)
            name = self.known_names[best_match_index]
            confidence = 1.0 - min_distance
            return name, confidence
        
        return None, 0.0
    
    def update(self, frame: np.ndarray, detections: List[Detection]) -> List[Track]:
        """
        Update tracker with new detections.
        
        Args:
            frame: Current frame
            detections: List of face detections
            
        Returns:
            List of active tracks
        """
        current_time = time.time()
        
        if self.tracker is not None:
            # Use DeepSort tracker
            return self._update_with_deepsort(frame, detections, current_time)
        else:
            # Use simple tracker
            return self._update_simple(frame, detections, current_time)
    
    def _update_with_deepsort(self, frame: np.ndarray, detections: List[Detection], current_time: float) -> List[Track]:
        """Update using DeepSort tracker"""
        # Prepare detections for DeepSort
        bbs = []
        confidences = []
        features = []
        
        for det in detections:
            x, y, w, h = det.bbox
            bbs.append([x, y, x + w, y + h])  # Convert to [x1, y1, x2, y2]
            confidences.append(det.confidence)
            
            # Use face encoding as feature if available
            if det.encoding is not None:
                features.append(det.encoding)
            else:
                features.append(np.random.rand(128))  # Dummy feature
        
        if bbs:
            tracks = self.tracker.update_tracks(bbs, confidences=confidences, features=features, frame=frame)
        else:
            tracks = self.tracker.update_tracks([], frame=frame)
        
        # Convert DeepSort tracks to our Track objects
        active_tracks = []
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            bbox = track.to_ltrb()  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = map(int, bbox)
            track_bbox = (x1, y1, x2 - x1, y2 - y1)  # Convert to (x, y, w, h)
            
            # Find corresponding detection for face recognition
            identity = None
            identity_confidence = 0.0
            encoding = None
            
            for det in detections:
                det_x, det_y, det_w, det_h = det.bbox
                if self._bbox_overlap(track_bbox, det.bbox) > 0.5:
                    if det.encoding is not None:
                        identity, identity_confidence = self.recognize_face(det.encoding)
                        encoding = det.encoding
                    break
            
            track_obj = Track(
                track_id=track.track_id,
                bbox=track_bbox,
                confidence=0.9,
                encoding=encoding,
                identity=identity,
                identity_confidence=identity_confidence,
                age=track.age,
                hits=track.hits,
                time_since_update=track.time_since_update,
                last_seen=current_time
            )
            
            # Update center history
            center_x = x1 + (x2 - x1) // 2
            center_y = y1 + (y2 - y1) // 2
            track_obj.center_history.append((center_x, center_y))
            
            # Keep only last 30 positions
            if len(track_obj.center_history) > 30:
                track_obj.center_history = track_obj.center_history[-30:]
            
            active_tracks.append(track_obj)
            self.active_tracks[track.track_id] = track_obj
        
        return active_tracks
    
    def _update_simple(self, frame: np.ndarray, detections: List[Detection], current_time: float) -> List[Track]:
        """Simple tracker implementation as fallback"""
        # This is a basic implementation - in production, you'd want a more sophisticated tracker
        active_tracks = []
        
        for det in detections:
            # For simplicity, create a new track for each detection
            track_id = self.next_track_id
            self.next_track_id += 1
            
            identity, identity_confidence = None, 0.0
            if det.encoding is not None:
                identity, identity_confidence = self.recognize_face(det.encoding)
            
            track = Track(
                track_id=track_id,
                bbox=det.bbox,
                confidence=det.confidence,
                encoding=det.encoding,
                identity=identity,
                identity_confidence=identity_confidence,
                last_seen=current_time
            )
            
            active_tracks.append(track)
            self.active_tracks[track_id] = track
        
        return active_tracks
    
    def _bbox_overlap(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate overlap between two bounding boxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        left = max(x1, x2)
        top = max(y1, y2)
        right = min(x1 + w1, x2 + w2)
        bottom = min(y1 + h1, y2 + h2)
        
        if right <= left or bottom <= top:
            return 0.0
        
        intersection = (right - left) * (bottom - top)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def get_track_history(self, track_id: int) -> List[Dict]:
        """Get movement history for a specific track"""
        return self.track_history.get(track_id, [])
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def get_fps(self) -> float:
        """Get current FPS"""
        return self.current_fps
    
    def get_active_tracks_count(self) -> int:
        """Get number of active tracks"""
        return len(self.active_tracks)
    
    def cleanup_old_tracks(self, max_age_seconds: int = 300):
        """Remove old tracks from memory"""
        current_time = time.time()
        to_remove = []
        
        for track_id, track in self.active_tracks.items():
            if current_time - track.last_seen > max_age_seconds:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.active_tracks[track_id]
        
        logger.debug(f"Cleaned up {len(to_remove)} old tracks")