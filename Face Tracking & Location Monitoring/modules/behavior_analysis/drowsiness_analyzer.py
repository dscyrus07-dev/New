"""
Drowsiness Analyzer

This module analyzes facial behavior data to calculate drowsiness scores
and detect fatigue patterns in real-time.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import time
from datetime import datetime, timedelta
from collections import deque
import logging

from .behavior_detector import BehaviorData, EyeState, HeadPose, MouthState

logger = logging.getLogger(__name__)

@dataclass
class DrowsinessScore:
    """Drowsiness analysis result"""
    overall_score: float = 0.0  # 0-100 scale
    eye_score: float = 0.0
    head_pose_score: float = 0.0
    yawn_score: float = 0.0
    is_drowsy: bool = False
    alert_level: str = "Normal"  # Normal, Mild, Moderate, Severe
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class BehaviorPattern:
    """Behavior pattern over time"""
    avg_eye_closure_rate: float = 0.0
    max_eye_closure_duration: float = 0.0
    yawn_frequency: float = 0.0
    head_movement_variance: float = 0.0
    attention_score: float = 100.0

class DrowsinessAnalyzer:
    """
    Analyzes facial behavior patterns to detect drowsiness and fatigue.
    Combines eye state, head pose, and yawn data to calculate comprehensive scores.
    """
    
    def __init__(self,
                 time_window: int = 30,  # seconds
                 drowsiness_threshold: float = 60.0,
                 severe_threshold: float = 80.0,
                 eye_closure_weight: float = 0.5,
                 head_pose_weight: float = 0.3,
                 yawn_weight: float = 0.2):
        """
        Initialize drowsiness analyzer.
        
        Args:
            time_window: Time window for analysis in seconds
            drowsiness_threshold: Threshold for drowsiness detection
            severe_threshold: Threshold for severe drowsiness
            eye_closure_weight: Weight for eye closure in final score
            head_pose_weight: Weight for head pose in final score
            yawn_weight: Weight for yawn frequency in final score
        """
        self.time_window = time_window
        self.drowsiness_threshold = drowsiness_threshold
        self.severe_threshold = severe_threshold
        self.eye_closure_weight = eye_closure_weight
        self.head_pose_weight = head_pose_weight
        self.yawn_weight = yawn_weight
        
        # Behavior history for each face
        self.behavior_history: Dict[int, deque] = {}
        self.drowsiness_scores: Dict[int, List[DrowsinessScore]] = {}
        
        # Alert tracking
        self.alert_counts: Dict[int, Dict[str, int]] = {}
        self.last_alert_time: Dict[int, float] = {}
        
        # Calibration parameters
        self.baseline_ear: Dict[int, float] = {}
        self.baseline_head_variance: Dict[int, float] = {}
        
        logger.info("DrowsinessAnalyzer initialized")
    
    def analyze_drowsiness(self, face_id: int, behavior_data: BehaviorData) -> DrowsinessScore:
        """
        Analyze drowsiness for a specific face.
        
        Args:
            face_id: Face identifier
            behavior_data: Current behavior data
            
        Returns:
            Drowsiness score and analysis
        """
        # Initialize face tracking if not exists
        if face_id not in self.behavior_history:
            self.behavior_history[face_id] = deque(maxlen=self.time_window * 10)  # Assume ~10 FPS
            self.drowsiness_scores[face_id] = []
            self.alert_counts[face_id] = {"Normal": 0, "Mild": 0, "Moderate": 0, "Severe": 0}
            self.last_alert_time[face_id] = 0.0
        
        # Add current behavior to history
        self.behavior_history[face_id].append(behavior_data)
        
        # Calculate individual scores
        eye_score = self._calculate_eye_drowsiness_score(face_id)
        head_pose_score = self._calculate_head_pose_score(face_id)
        yawn_score = self._calculate_yawn_score(face_id)
        
        # Calculate overall drowsiness score
        overall_score = (
            eye_score * self.eye_closure_weight +
            head_pose_score * self.head_pose_weight +
            yawn_score * self.yawn_weight
        )
        
        # Determine alert level
        alert_level = self._determine_alert_level(overall_score)
        is_drowsy = overall_score >= self.drowsiness_threshold
        
        # Create drowsiness score object
        drowsiness_score = DrowsinessScore(
            overall_score=overall_score,
            eye_score=eye_score,
            head_pose_score=head_pose_score,
            yawn_score=yawn_score,
            is_drowsy=is_drowsy,
            alert_level=alert_level
        )
        
        # Store score
        self.drowsiness_scores[face_id].append(drowsiness_score)
        
        # Update alert counts
        self.alert_counts[face_id][alert_level] += 1
        
        # Log alerts
        if is_drowsy:
            current_time = time.time()
            if current_time - self.last_alert_time[face_id] > 10.0:  # Throttle alerts
                logger.warning(f"Drowsiness detected for face {face_id}: "
                             f"Score={overall_score:.1f}, Level={alert_level}")
                self.last_alert_time[face_id] = current_time
        
        return drowsiness_score
    
    def _calculate_eye_drowsiness_score(self, face_id: int) -> float:
        """Calculate drowsiness score based on eye behavior"""
        history = list(self.behavior_history[face_id])
        if not history:
            return 0.0
        
        # Calculate metrics over time window
        recent_history = [h for h in history if time.time() - h.timestamp < self.time_window]
        
        if not recent_history:
            return 0.0
        
        # Eye closure metrics
        ear_values = [h.eye_state.avg_ear for h in recent_history]
        closure_durations = [h.eye_state.closure_duration for h in recent_history if h.eye_state.is_closed]
        closed_frames = sum(1 for h in recent_history if h.eye_state.is_closed)
        
        # Calculate baseline EAR if not set
        if face_id not in self.baseline_ear:
            open_ear_values = [ear for ear, h in zip(ear_values, recent_history) if not h.eye_state.is_closed]
            if open_ear_values:
                self.baseline_ear[face_id] = np.mean(open_ear_values)
            else:
                self.baseline_ear[face_id] = 0.3  # Default baseline
        
        baseline = self.baseline_ear[face_id]
        
        # Score components
        score = 0.0
        
        # 1. Eye closure rate (0-40 points)
        closure_rate = closed_frames / len(recent_history)
        score += min(closure_rate * 200, 40)  # Max 40 points
        
        # 2. Average EAR deviation from baseline (0-30 points)
        if ear_values:
            avg_ear = np.mean(ear_values)
            ear_deviation = max(0, (baseline - avg_ear) / baseline)
            score += min(ear_deviation * 100, 30)  # Max 30 points
        
        # 3. Long eye closures (0-30 points)
        if closure_durations:
            max_closure = max(closure_durations)
            long_closure_score = min(max_closure * 10, 30)  # Max 30 points
            score += long_closure_score
        
        return min(score, 100)
    
    def _calculate_head_pose_score(self, face_id: int) -> float:
        """Calculate drowsiness score based on head pose"""
        history = list(self.behavior_history[face_id])
        if not history:
            return 0.0
        
        recent_history = [h for h in history if time.time() - h.timestamp < self.time_window]
        
        if not recent_history:
            return 0.0
        
        # Head pose metrics
        pitch_values = [h.head_pose.pitch for h in recent_history]
        yaw_values = [h.head_pose.yaw for h in recent_history]
        looking_away_frames = sum(1 for h in recent_history if h.head_pose.is_looking_away)
        
        score = 0.0
        
        # 1. Head nodding (pitch variance) - drowsy people nod
        if len(pitch_values) > 5:
            pitch_variance = np.var(pitch_values)
            # High variance indicates nodding
            nodding_score = min(pitch_variance / 100, 40)  # Max 40 points
            score += nodding_score
        
        # 2. Looking away rate
        looking_away_rate = looking_away_frames / len(recent_history)
        score += min(looking_away_rate * 60, 30)  # Max 30 points
        
        # 3. Head tilt consistency (drowsy people may tilt head)
        if len(recent_history) > 10:
            recent_pitch = pitch_values[-10:]  # Last 10 frames
            if recent_pitch:
                avg_recent_pitch = np.mean(recent_pitch)
                if abs(avg_recent_pitch) > 15:  # Head tilted
                    score += min(abs(avg_recent_pitch), 30)  # Max 30 points
        
        return min(score, 100)
    
    def _calculate_yawn_score(self, face_id: int) -> float:
        """Calculate drowsiness score based on yawn frequency"""
        history = list(self.behavior_history[face_id])
        if not history:
            return 0.0
        
        recent_history = [h for h in history if time.time() - h.timestamp < self.time_window]
        
        if not recent_history:
            return 0.0
        
        # Yawn metrics
        yawn_frames = sum(1 for h in recent_history if h.mouth_state.is_yawning)
        yawn_durations = [h.mouth_state.yawn_duration for h in recent_history if h.mouth_state.is_yawning]
        
        score = 0.0
        
        # 1. Yawn frequency (0-60 points)
        yawn_rate = yawn_frames / len(recent_history)
        score += min(yawn_rate * 300, 60)  # Max 60 points
        
        # 2. Long yawns (0-40 points)
        if yawn_durations:
            max_yawn_duration = max(yawn_durations)
            long_yawn_score = min(max_yawn_duration * 20, 40)  # Max 40 points
            score += long_yawn_score
        
        return min(score, 100)
    
    def _determine_alert_level(self, score: float) -> str:
        """Determine alert level based on score"""
        if score >= self.severe_threshold:
            return "Severe"
        elif score >= self.drowsiness_threshold:
            return "Moderate"
        elif score >= self.drowsiness_threshold * 0.7:
            return "Mild"
        else:
            return "Normal"
    
    def get_behavior_pattern(self, face_id: int, minutes: int = 5) -> Optional[BehaviorPattern]:
        """
        Get behavior pattern analysis for a face over specified time period.
        
        Args:
            face_id: Face identifier
            minutes: Time period in minutes
            
        Returns:
            Behavior pattern analysis or None
        """
        if face_id not in self.behavior_history:
            return None
        
        # Get history for specified time period
        cutoff_time = time.time() - (minutes * 60)
        history = [h for h in self.behavior_history[face_id] if h.timestamp >= cutoff_time]
        
        if not history:
            return None
        
        # Calculate pattern metrics
        closed_frames = sum(1 for h in history if h.eye_state.is_closed)
        total_frames = len(history)
        avg_eye_closure_rate = closed_frames / total_frames if total_frames > 0 else 0.0
        
        closure_durations = [h.eye_state.closure_duration for h in history if h.eye_state.is_closed]
        max_eye_closure_duration = max(closure_durations) if closure_durations else 0.0
        
        yawn_frames = sum(1 for h in history if h.mouth_state.is_yawning)
        yawn_frequency = yawn_frames / total_frames if total_frames > 0 else 0.0
        
        # Head movement variance
        pitch_values = [h.head_pose.pitch for h in history]
        yaw_values = [h.head_pose.yaw for h in history]
        head_movement_variance = 0.0
        if len(pitch_values) > 1:
            head_movement_variance = np.var(pitch_values) + np.var(yaw_values)
        
        # Attention score (inverse of looking away rate)
        looking_away_frames = sum(1 for h in history if h.head_pose.is_looking_away)
        attention_rate = 1.0 - (looking_away_frames / total_frames) if total_frames > 0 else 1.0
        attention_score = attention_rate * 100
        
        return BehaviorPattern(
            avg_eye_closure_rate=avg_eye_closure_rate,
            max_eye_closure_duration=max_eye_closure_duration,
            yawn_frequency=yawn_frequency,
            head_movement_variance=head_movement_variance,
            attention_score=attention_score
        )
    
    def get_drowsiness_trend(self, face_id: int, minutes: int = 10) -> List[float]:
        """
        Get drowsiness score trend over time.
        
        Args:
            face_id: Face identifier
            minutes: Time period in minutes
            
        Returns:
            List of drowsiness scores over time
        """
        if face_id not in self.drowsiness_scores:
            return []
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_scores = [
            score.overall_score for score in self.drowsiness_scores[face_id]
            if score.timestamp >= cutoff_time
        ]
        
        return recent_scores
    
    def get_alert_statistics(self, face_id: int) -> Dict[str, Any]:
        """Get alert statistics for a face"""
        if face_id not in self.alert_counts:
            return {}
        
        total_alerts = sum(self.alert_counts[face_id].values())
        
        return {
            'total_detections': total_alerts,
            'alert_breakdown': self.alert_counts[face_id].copy(),
            'severe_alert_rate': self.alert_counts[face_id]["Severe"] / total_alerts if total_alerts > 0 else 0,
            'last_alert_time': self.last_alert_time.get(face_id, 0)
        }
    
    def reset_face_analysis(self, face_id: int):
        """Reset analysis data for a specific face"""
        if face_id in self.behavior_history:
            del self.behavior_history[face_id]
        if face_id in self.drowsiness_scores:
            del self.drowsiness_scores[face_id]
        if face_id in self.alert_counts:
            del self.alert_counts[face_id]
        if face_id in self.last_alert_time:
            del self.last_alert_time[face_id]
        if face_id in self.baseline_ear:
            del self.baseline_ear[face_id]
        if face_id in self.baseline_head_variance:
            del self.baseline_head_variance[face_id]
    
    def calibrate_baseline(self, face_id: int, calibration_seconds: int = 10):
        """
        Calibrate baseline values for a face during alert state.
        
        Args:
            face_id: Face identifier
            calibration_seconds: Duration for calibration
        """
        if face_id not in self.behavior_history:
            return
        
        # Get recent history for calibration
        cutoff_time = time.time() - calibration_seconds
        recent_history = [h for h in self.behavior_history[face_id] if h.timestamp >= cutoff_time]
        
        if not recent_history:
            return
        
        # Calculate baseline EAR from alert state
        alert_ear_values = [h.eye_state.avg_ear for h in recent_history if not h.eye_state.is_closed]
        if alert_ear_values:
            self.baseline_ear[face_id] = np.mean(alert_ear_values)
            logger.info(f"Calibrated baseline EAR for face {face_id}: {self.baseline_ear[face_id]:.3f}")
    
    def export_analysis_data(self, face_id: int, hours: int = 1) -> Dict[str, Any]:
        """
        Export analysis data for external processing.
        
        Args:
            face_id: Face identifier
            hours: Hours of data to export
            
        Returns:
            Dictionary containing analysis data
        """
        if face_id not in self.behavior_history:
            return {}
        
        cutoff_time = time.time() - (hours * 3600)
        
        # Export behavior history
        recent_behavior = [
            {
                'timestamp': h.timestamp,
                'eye_ear': h.eye_state.avg_ear,
                'eye_closed': h.eye_state.is_closed,
                'eye_closure_duration': h.eye_state.closure_duration,
                'head_pitch': h.head_pose.pitch,
                'head_yaw': h.head_pose.yaw,
                'head_looking_away': h.head_pose.is_looking_away,
                'mouth_ratio': h.mouth_state.mouth_aspect_ratio,
                'is_yawning': h.mouth_state.is_yawning,
                'yawn_duration': h.mouth_state.yawn_duration
            }
            for h in self.behavior_history[face_id]
            if h.timestamp >= cutoff_time
        ]
        
        # Export drowsiness scores
        recent_scores = [
            {
                'timestamp': score.timestamp.isoformat(),
                'overall_score': score.overall_score,
                'eye_score': score.eye_score,
                'head_pose_score': score.head_pose_score,
                'yawn_score': score.yawn_score,
                'is_drowsy': score.is_drowsy,
                'alert_level': score.alert_level
            }
            for score in self.drowsiness_scores.get(face_id, [])
            if score.timestamp.timestamp() >= cutoff_time
        ]
        
        return {
            'face_id': face_id,
            'export_time': datetime.now().isoformat(),
            'time_window_hours': hours,
            'behavior_data': recent_behavior,
            'drowsiness_scores': recent_scores,
            'alert_statistics': self.get_alert_statistics(face_id),
            'baseline_ear': self.baseline_ear.get(face_id),
            'total_data_points': len(recent_behavior)
        }
    
    def cleanup_old_data(self, hours: int = 24):
        """Clean up old analysis data"""
        cutoff_time = time.time() - (hours * 3600)
        cutoff_datetime = datetime.now() - timedelta(hours=hours)
        
        for face_id in list(self.behavior_history.keys()):
            # Clean behavior history
            if self.behavior_history[face_id]:
                # Convert deque to list, filter, and convert back
                filtered_history = [h for h in self.behavior_history[face_id] if h.timestamp >= cutoff_time]
                self.behavior_history[face_id].clear()
                self.behavior_history[face_id].extend(filtered_history)
            
            # Clean drowsiness scores
            if face_id in self.drowsiness_scores:
                self.drowsiness_scores[face_id] = [
                    score for score in self.drowsiness_scores[face_id]
                    if score.timestamp >= cutoff_datetime
                ]
        
        logger.info(f"Cleaned up analysis data older than {hours} hours")