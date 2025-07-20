"""
Facial Behavior Detector

This module implements real-time facial behavior detection using MediaPipe
including eye state monitoring, head pose estimation, and yawn detection.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import time
import math
import logging
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class EyeState:
    """Eye state information"""
    left_ear: float = 0.0  # Eye Aspect Ratio
    right_ear: float = 0.0
    avg_ear: float = 0.0
    is_closed: bool = False
    closure_duration: float = 0.0

@dataclass
class HeadPose:
    """Head pose information"""
    pitch: float = 0.0  # Up/down rotation
    yaw: float = 0.0    # Left/right rotation
    roll: float = 0.0   # Tilt rotation
    is_looking_away: bool = False

@dataclass
class MouthState:
    """Mouth state information"""
    mouth_aspect_ratio: float = 0.0
    is_yawning: bool = False
    yawn_duration: float = 0.0

@dataclass
class BehaviorData:
    """Complete behavior analysis data"""
    timestamp: float
    eye_state: EyeState
    head_pose: HeadPose
    mouth_state: MouthState
    face_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    landmarks: Optional[np.ndarray] = None
    confidence: float = 0.0

class BehaviorDetector:
    """
    Real-time facial behavior detector using MediaPipe.
    Detects eye state, head pose, and mouth movements for behavior analysis.
    """
    
    def __init__(self,
                 ear_threshold: float = 0.25,
                 ear_consecutive_frames: int = 16,
                 yawn_threshold: float = 20.0,
                 head_pose_threshold: float = 30.0,
                 detection_confidence: float = 0.5,
                 tracking_confidence: float = 0.5):
        """
        Initialize behavior detector.
        
        Args:
            ear_threshold: Eye aspect ratio threshold for closed eyes
            ear_consecutive_frames: Consecutive frames for eye closure detection
            yawn_threshold: Mouth aspect ratio threshold for yawn detection
            head_pose_threshold: Head pose angle threshold for looking away
            detection_confidence: MediaPipe detection confidence
            tracking_confidence: MediaPipe tracking confidence
        """
        self.ear_threshold = ear_threshold
        self.ear_consecutive_frames = ear_consecutive_frames
        self.yawn_threshold = yawn_threshold
        self.head_pose_threshold = head_pose_threshold
        
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=10,
            refine_landmarks=True,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        
        # Face landmark indices
        self._init_landmark_indices()
        
        # Tracking variables
        self.face_behaviors: Dict[int, Dict] = {}  # Track behaviors per face
        self.frame_counters: Dict[int, Dict] = {}  # Frame counters per face
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        
        logger.info("BehaviorDetector initialized with MediaPipe")
    
    def _init_landmark_indices(self):
        """Initialize facial landmark indices for different features"""
        # Eye landmarks (MediaPipe 468 face landmarks)
        self.LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        
        # Simplified eye indices for EAR calculation
        self.LEFT_EYE_EAR_INDICES = [362, 385, 387, 263, 373, 380]  # [outer, top1, top2, inner, bottom1, bottom2]
        self.RIGHT_EYE_EAR_INDICES = [33, 160, 158, 133, 153, 144]
        
        # Mouth landmarks
        self.MOUTH_INDICES = [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
        
        # Mouth landmarks for yawn detection
        self.MOUTH_YAWN_INDICES = [13, 14, 269, 270, 267, 271, 272]  # Top and bottom lip points
        
        # Head pose landmarks
        self.HEAD_POSE_INDICES = [1, 33, 263, 61, 291, 199]  # Nose tip, eye corners, mouth corners
    
    def detect_behaviors(self, frame: np.ndarray, face_boxes: List[Tuple[int, int, int, int]] = None) -> List[BehaviorData]:
        """
        Detect facial behaviors in the frame.
        
        Args:
            frame: Input frame
            face_boxes: Optional list of face bounding boxes to focus analysis
            
        Returns:
            List of behavior data for detected faces
        """
        if frame is None or frame.size == 0:
            return []
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        behavior_data_list = []
        
        if results.multi_face_landmarks:
            for face_id, face_landmarks in enumerate(results.multi_face_landmarks):
                # Convert landmarks to numpy array
                landmarks = self._landmarks_to_numpy(face_landmarks, frame.shape)
                
                # Calculate face bounding box
                face_bbox = self._calculate_face_bbox(landmarks)
                
                # Skip if face box is provided and doesn't match
                if face_boxes and not self._bbox_matches_any(face_bbox, face_boxes):
                    continue
                
                # Analyze eye state
                eye_state = self._analyze_eye_state(landmarks, face_id)
                
                # Analyze head pose
                head_pose = self._analyze_head_pose(landmarks, frame.shape)
                
                # Analyze mouth state
                mouth_state = self._analyze_mouth_state(landmarks, face_id)
                
                # Create behavior data
                behavior_data = BehaviorData(
                    timestamp=time.time(),
                    eye_state=eye_state,
                    head_pose=head_pose,
                    mouth_state=mouth_state,
                    face_bbox=face_bbox,
                    landmarks=landmarks,
                    confidence=0.9  # MediaPipe doesn't provide direct confidence
                )
                
                behavior_data_list.append(behavior_data)
        
        # Update FPS
        self._update_fps()
        
        return behavior_data_list
    
    def _landmarks_to_numpy(self, landmarks, frame_shape) -> np.ndarray:
        """Convert MediaPipe landmarks to numpy array"""
        h, w = frame_shape[:2]
        points = []
        
        for landmark in landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            points.append([x, y])
        
        return np.array(points)
    
    def _calculate_face_bbox(self, landmarks: np.ndarray) -> Tuple[int, int, int, int]:
        """Calculate bounding box from landmarks"""
        x_min = np.min(landmarks[:, 0])
        x_max = np.max(landmarks[:, 0])
        y_min = np.min(landmarks[:, 1])
        y_max = np.max(landmarks[:, 1])
        
        return (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
    
    def _bbox_matches_any(self, face_bbox: Tuple[int, int, int, int], 
                         face_boxes: List[Tuple[int, int, int, int]]) -> bool:
        """Check if face bbox matches any of the provided boxes"""
        fx, fy, fw, fh = face_bbox
        face_center = (fx + fw // 2, fy + fh // 2)
        
        for box in face_boxes:
            bx, by, bw, bh = box
            if bx <= face_center[0] <= bx + bw and by <= face_center[1] <= by + bh:
                return True
        
        return False
    
    def _analyze_eye_state(self, landmarks: np.ndarray, face_id: int) -> EyeState:
        """Analyze eye state using Eye Aspect Ratio (EAR)"""
        # Calculate EAR for both eyes
        left_ear = self._calculate_ear(landmarks, self.LEFT_EYE_EAR_INDICES)
        right_ear = self._calculate_ear(landmarks, self.RIGHT_EYE_EAR_INDICES)
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Initialize face tracking if not exists
        if face_id not in self.face_behaviors:
            self.face_behaviors[face_id] = {
                'eye_closure_start': None,
                'eye_closure_frames': 0,
                'yawn_start': None,
                'yawn_frames': 0
            }
        
        face_data = self.face_behaviors[face_id]
        
        # Determine if eyes are closed
        is_closed = avg_ear < self.ear_threshold
        closure_duration = 0.0
        
        if is_closed:
            if face_data['eye_closure_start'] is None:
                face_data['eye_closure_start'] = time.time()
                face_data['eye_closure_frames'] = 1
            else:
                face_data['eye_closure_frames'] += 1
                closure_duration = time.time() - face_data['eye_closure_start']
        else:
            face_data['eye_closure_start'] = None
            face_data['eye_closure_frames'] = 0
        
        return EyeState(
            left_ear=left_ear,
            right_ear=right_ear,
            avg_ear=avg_ear,
            is_closed=is_closed,
            closure_duration=closure_duration
        )
    
    def _calculate_ear(self, landmarks: np.ndarray, eye_indices: List[int]) -> float:
        """Calculate Eye Aspect Ratio"""
        if len(eye_indices) < 6:
            return 0.0
        
        # Get eye landmark points
        eye_points = landmarks[eye_indices]
        
        # Calculate vertical distances
        vertical_1 = np.linalg.norm(eye_points[1] - eye_points[5])  # top1 - bottom1
        vertical_2 = np.linalg.norm(eye_points[2] - eye_points[4])  # top2 - bottom2
        
        # Calculate horizontal distance
        horizontal = np.linalg.norm(eye_points[0] - eye_points[3])  # outer - inner
        
        # Calculate EAR
        if horizontal == 0:
            return 0.0
        
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear
    
    def _analyze_head_pose(self, landmarks: np.ndarray, frame_shape) -> HeadPose:
        """Analyze head pose using facial landmarks"""
        h, w = frame_shape[:2]
        
        # 3D model points (approximate face model)
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float64)
        
        # 2D image points from landmarks
        if len(landmarks) > max(self.HEAD_POSE_INDICES):
            image_points = np.array([
                landmarks[1],    # Nose tip
                landmarks[152],  # Chin
                landmarks[226],  # Left eye left corner
                landmarks[446],  # Right eye right corner
                landmarks[57],   # Left mouth corner
                landmarks[287]   # Right mouth corner
            ], dtype=np.float64)
        else:
            # Fallback if not enough landmarks
            return HeadPose()
        
        # Camera matrix (approximate)
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Distortion coefficients (assuming no distortion)
        dist_coeffs = np.zeros((4, 1))
        
        try:
            # Solve PnP
            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs
            )
            
            if success:
                # Convert rotation vector to rotation matrix
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                
                # Calculate Euler angles
                pitch, yaw, roll = self._rotation_matrix_to_euler_angles(rotation_matrix)
                
                # Convert to degrees
                pitch_deg = math.degrees(pitch)
                yaw_deg = math.degrees(yaw)
                roll_deg = math.degrees(roll)
                
                # Determine if looking away
                is_looking_away = (
                    abs(pitch_deg) > self.head_pose_threshold or
                    abs(yaw_deg) > self.head_pose_threshold
                )
                
                return HeadPose(
                    pitch=pitch_deg,
                    yaw=yaw_deg,
                    roll=roll_deg,
                    is_looking_away=is_looking_away
                )
        
        except Exception as e:
            logger.debug(f"Head pose calculation failed: {e}")
        
        return HeadPose()
    
    def _rotation_matrix_to_euler_angles(self, R: np.ndarray) -> Tuple[float, float, float]:
        """Convert rotation matrix to Euler angles"""
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        
        singular = sy < 1e-6
        
        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0
        
        return x, y, z
    
    def _analyze_mouth_state(self, landmarks: np.ndarray, face_id: int) -> MouthState:
        """Analyze mouth state for yawn detection"""
        # Calculate mouth aspect ratio
        mouth_ratio = self._calculate_mouth_aspect_ratio(landmarks)
        
        face_data = self.face_behaviors[face_id]
        
        # Determine if yawning
        is_yawning = mouth_ratio > self.yawn_threshold
        yawn_duration = 0.0
        
        if is_yawning:
            if face_data['yawn_start'] is None:
                face_data['yawn_start'] = time.time()
                face_data['yawn_frames'] = 1
            else:
                face_data['yawn_frames'] += 1
                yawn_duration = time.time() - face_data['yawn_start']
        else:
            face_data['yawn_start'] = None
            face_data['yawn_frames'] = 0
        
        return MouthState(
            mouth_aspect_ratio=mouth_ratio,
            is_yawning=is_yawning,
            yawn_duration=yawn_duration
        )
    
    def _calculate_mouth_aspect_ratio(self, landmarks: np.ndarray) -> float:
        """Calculate mouth aspect ratio for yawn detection"""
        try:
            # Get mouth landmarks
            mouth_top = landmarks[13]     # Upper lip top
            mouth_bottom = landmarks[14]  # Lower lip bottom
            mouth_left = landmarks[61]    # Left mouth corner
            mouth_right = landmarks[291]  # Right mouth corner
            
            # Calculate vertical and horizontal distances
            vertical_dist = np.linalg.norm(mouth_top - mouth_bottom)
            horizontal_dist = np.linalg.norm(mouth_left - mouth_right)
            
            if horizontal_dist == 0:
                return 0.0
            
            # Calculate mouth aspect ratio
            mouth_ratio = vertical_dist / horizontal_dist
            return mouth_ratio * 100  # Scale for easier threshold setting
            
        except (IndexError, ValueError):
            return 0.0
    
    def _update_fps(self):
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
    
    def reset_face_tracking(self, face_id: int = None):
        """Reset tracking data for a specific face or all faces"""
        if face_id is not None:
            if face_id in self.face_behaviors:
                del self.face_behaviors[face_id]
            if face_id in self.frame_counters:
                del self.frame_counters[face_id]
        else:
            self.face_behaviors.clear()
            self.frame_counters.clear()
    
    def get_behavior_summary(self, face_id: int) -> Dict[str, Any]:
        """Get behavior summary for a specific face"""
        if face_id not in self.face_behaviors:
            return {}
        
        face_data = self.face_behaviors[face_id]
        
        return {
            'eye_closure_frames': face_data.get('eye_closure_frames', 0),
            'yawn_frames': face_data.get('yawn_frames', 0),
            'has_eye_closure_start': face_data.get('eye_closure_start') is not None,
            'has_yawn_start': face_data.get('yawn_start') is not None
        }
    
    def cleanup_old_tracking_data(self, max_age_seconds: int = 300):
        """Clean up old tracking data"""
        current_time = time.time()
        to_remove = []
        
        for face_id, data in self.face_behaviors.items():
            # Remove faces that haven't been updated recently
            last_update = data.get('last_update', current_time)
            if current_time - last_update > max_age_seconds:
                to_remove.append(face_id)
        
        for face_id in to_remove:
            self.reset_face_tracking(face_id)
        
        logger.debug(f"Cleaned up {len(to_remove)} old face tracking entries")