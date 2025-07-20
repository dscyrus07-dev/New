"""
Behavior Analysis Visualizer

This module provides visualization tools for facial behavior analysis
including eye state, head pose, yawn detection, and drowsiness indicators.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import logging

from .behavior_detector import BehaviorData
from .drowsiness_analyzer import DrowsinessScore, BehaviorPattern

logger = logging.getLogger(__name__)

class BehaviorVisualizer:
    """
    Visualization tools for facial behavior analysis.
    Provides methods to draw behavior indicators and analysis results on video frames.
    """
    
    def __init__(self,
                 show_landmarks: bool = False,
                 show_eye_state: bool = True,
                 show_head_pose: bool = True,
                 show_drowsiness: bool = True):
        """
        Initialize behavior visualizer.
        
        Args:
            show_landmarks: Whether to show facial landmarks
            show_eye_state: Whether to show eye state indicators
            show_head_pose: Whether to show head pose indicators
            show_drowsiness: Whether to show drowsiness score
        """
        self.show_landmarks = show_landmarks
        self.show_eye_state = show_eye_state
        self.show_head_pose = show_head_pose
        self.show_drowsiness = show_drowsiness
        
        # Color scheme
        self.colors = {
            'normal': (0, 255, 0),      # Green
            'mild': (0, 255, 255),      # Yellow
            'moderate': (0, 165, 255),  # Orange
            'severe': (0, 0, 255),      # Red
            'eye_open': (0, 255, 0),    # Green
            'eye_closed': (0, 0, 255),  # Red
            'yawning': (255, 0, 255),   # Magenta
            'looking_away': (255, 255, 0)  # Cyan
        }
        
        logger.info("BehaviorVisualizer initialized")
    
    def draw_behavior_analysis(self, frame: np.ndarray, 
                             behavior_data: BehaviorData,
                             drowsiness_score: Optional[DrowsinessScore] = None) -> np.ndarray:
        """
        Draw complete behavior analysis on frame.
        
        Args:
            frame: Input frame
            behavior_data: Behavior analysis data
            drowsiness_score: Optional drowsiness score
            
        Returns:
            Frame with behavior analysis drawn
        """
        output_frame = frame.copy()
        
        # Draw facial landmarks if enabled
        if self.show_landmarks and behavior_data.landmarks is not None:
            output_frame = self._draw_facial_landmarks(output_frame, behavior_data.landmarks)
        
        # Draw eye state indicators
        if self.show_eye_state:
            output_frame = self._draw_eye_state(output_frame, behavior_data)
        
        # Draw head pose indicators
        if self.show_head_pose:
            output_frame = self._draw_head_pose(output_frame, behavior_data)
        
        # Draw drowsiness indicators
        if self.show_drowsiness and drowsiness_score:
            output_frame = self._draw_drowsiness_indicators(output_frame, behavior_data, drowsiness_score)
        
        return output_frame
    
    def _draw_facial_landmarks(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Draw facial landmarks"""
        for point in landmarks:
            cv2.circle(frame, tuple(point.astype(int)), 1, (0, 255, 0), -1)
        return frame
    
    def _draw_eye_state(self, frame: np.ndarray, behavior_data: BehaviorData) -> np.ndarray:
        """Draw eye state indicators"""
        x, y, w, h = behavior_data.face_bbox
        eye_state = behavior_data.eye_state
        
        # Eye state indicator
        eye_color = self.colors['eye_closed'] if eye_state.is_closed else self.colors['eye_open']
        eye_text = f"Eyes: {'CLOSED' if eye_state.is_closed else 'OPEN'}"
        
        # Draw eye state
        self._draw_text_with_background(frame, eye_text, (x, y - 60), eye_color, 0.6)
        
        # Draw EAR values
        ear_text = f"EAR: L={eye_state.left_ear:.3f} R={eye_state.right_ear:.3f} Avg={eye_state.avg_ear:.3f}"
        self._draw_text_with_background(frame, ear_text, (x, y - 40), (255, 255, 255), 0.5)
        
        # Draw closure duration if eyes are closed
        if eye_state.is_closed and eye_state.closure_duration > 0:
            duration_text = f"Closed: {eye_state.closure_duration:.1f}s"
            self._draw_text_with_background(frame, duration_text, (x, y - 20), self.colors['eye_closed'], 0.5)
        
        return frame
    
    def _draw_head_pose(self, frame: np.ndarray, behavior_data: BehaviorData) -> np.ndarray:
        """Draw head pose indicators"""
        x, y, w, h = behavior_data.face_bbox
        head_pose = behavior_data.head_pose
        
        # Head pose angles
        pose_text = f"Head: P={head_pose.pitch:.1f}° Y={head_pose.yaw:.1f}° R={head_pose.roll:.1f}°"
        pose_color = self.colors['looking_away'] if head_pose.is_looking_away else (255, 255, 255)
        
        self._draw_text_with_background(frame, pose_text, (x + w + 10, y), pose_color, 0.5)
        
        # Looking away indicator
        if head_pose.is_looking_away:
            away_text = "LOOKING AWAY"
            self._draw_text_with_background(frame, away_text, (x + w + 10, y + 20), self.colors['looking_away'], 0.6)
        
        # Draw head pose visualization (optional)
        if behavior_data.landmarks is not None:
            self._draw_head_pose_axes(frame, behavior_data)
        
        return frame
    
    def _draw_head_pose_axes(self, frame: np.ndarray, behavior_data: BehaviorData):
        """Draw 3D head pose axes"""
        try:
            # Get face center
            x, y, w, h = behavior_data.face_bbox
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Convert angles to radians
            pitch = np.radians(behavior_data.head_pose.pitch)
            yaw = np.radians(behavior_data.head_pose.yaw)
            roll = np.radians(behavior_data.head_pose.roll)
            
            # Calculate axis endpoints
            axis_length = min(w, h) // 3
            
            # X-axis (red) - left/right
            x_end_x = int(center_x + axis_length * np.cos(yaw))
            x_end_y = int(center_y + axis_length * np.sin(pitch))
            cv2.arrowedLine(frame, (center_x, center_y), (x_end_x, x_end_y), (0, 0, 255), 2)
            
            # Y-axis (green) - up/down
            y_end_x = int(center_x - axis_length * np.sin(yaw))
            y_end_y = int(center_y - axis_length * np.cos(pitch))
            cv2.arrowedLine(frame, (center_x, center_y), (y_end_x, y_end_y), (0, 255, 0), 2)
            
            # Z-axis (blue) - forward/backward
            z_end_x = int(center_x + axis_length * np.sin(roll) * 0.5)
            z_end_y = int(center_y + axis_length * np.cos(roll) * 0.5)
            cv2.arrowedLine(frame, (center_x, center_y), (z_end_x, z_end_y), (255, 0, 0), 2)
            
        except Exception as e:
            logger.debug(f"Failed to draw head pose axes: {e}")
    
    def _draw_drowsiness_indicators(self, frame: np.ndarray, 
                                  behavior_data: BehaviorData,
                                  drowsiness_score: DrowsinessScore) -> np.ndarray:
        """Draw drowsiness indicators"""
        x, y, w, h = behavior_data.face_bbox
        
        # Get alert level color
        alert_colors = {
            'Normal': self.colors['normal'],
            'Mild': self.colors['mild'],
            'Moderate': self.colors['moderate'],
            'Severe': self.colors['severe']
        }
        
        alert_color = alert_colors.get(drowsiness_score.alert_level, (255, 255, 255))
        
        # Draw bounding box with alert color
        thickness = 3 if drowsiness_score.is_drowsy else 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), alert_color, thickness)
        
        # Draw drowsiness score
        score_text = f"Drowsiness: {drowsiness_score.overall_score:.1f}%"
        self._draw_text_with_background(frame, score_text, (x, y + h + 20), alert_color, 0.7)
        
        # Draw alert level
        level_text = f"Level: {drowsiness_score.alert_level}"
        self._draw_text_with_background(frame, level_text, (x, y + h + 45), alert_color, 0.6)
        
        # Draw component scores
        components_text = f"Eye:{drowsiness_score.eye_score:.0f} Head:{drowsiness_score.head_pose_score:.0f} Yawn:{drowsiness_score.yawn_score:.0f}"
        self._draw_text_with_background(frame, components_text, (x, y + h + 70), (255, 255, 255), 0.5)
        
        # Draw yawning indicator
        if behavior_data.mouth_state.is_yawning:
            yawn_text = f"YAWNING ({behavior_data.mouth_state.yawn_duration:.1f}s)"
            self._draw_text_with_background(frame, yawn_text, (x + w + 10, y + 40), self.colors['yawning'], 0.6)
        
        return frame
    
    def _draw_text_with_background(self, frame: np.ndarray, text: str, position: Tuple[int, int],
                                 color: Tuple[int, int, int], scale: float = 0.6):
        """Draw text with background"""
        x, y = position
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        
        # Draw background rectangle
        cv2.rectangle(frame,
                     (x - 2, y - text_height - baseline - 2),
                     (x + text_width + 2, y + baseline + 2),
                     (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)
    
    def draw_behavior_dashboard(self, frame: np.ndarray,
                              behavior_data_list: List[BehaviorData],
                              drowsiness_scores: List[DrowsinessScore]) -> np.ndarray:
        """
        Draw behavior analysis dashboard.
        
        Args:
            frame: Input frame
            behavior_data_list: List of behavior data for all faces
            drowsiness_scores: List of drowsiness scores
            
        Returns:
            Frame with dashboard
        """
        output_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # Create dashboard area
        dashboard_height = 200
        dashboard_width = 400
        dashboard = np.zeros((dashboard_height, dashboard_width, 3), dtype=np.uint8)
        dashboard.fill(30)  # Dark background
        
        # Dashboard title
        cv2.putText(dashboard, "Behavior Analysis Dashboard", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Statistics
        y_offset = 50
        line_height = 20
        
        if behavior_data_list:
            total_faces = len(behavior_data_list)
            closed_eyes = sum(1 for bd in behavior_data_list if bd.eye_state.is_closed)
            yawning = sum(1 for bd in behavior_data_list if bd.mouth_state.is_yawning)
            looking_away = sum(1 for bd in behavior_data_list if bd.head_pose.is_looking_away)
            
            stats_text = [
                f"Total Faces: {total_faces}",
                f"Closed Eyes: {closed_eyes}",
                f"Yawning: {yawning}",
                f"Looking Away: {looking_away}"
            ]
            
            for i, text in enumerate(stats_text):
                y_pos = y_offset + i * line_height
                cv2.putText(dashboard, text, (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Drowsiness alerts
        if drowsiness_scores:
            alert_counts = {'Normal': 0, 'Mild': 0, 'Moderate': 0, 'Severe': 0}
            for score in drowsiness_scores:
                alert_counts[score.alert_level] += 1
            
            alert_y_offset = y_offset + 100
            cv2.putText(dashboard, "Drowsiness Alerts:", (10, alert_y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            
            for i, (level, count) in enumerate(alert_counts.items()):
                if count > 0:
                    color = {
                        'Normal': (0, 255, 0),
                        'Mild': (0, 255, 255),
                        'Moderate': (0, 165, 255),
                        'Severe': (0, 0, 255)
                    }.get(level, (255, 255, 255))
                    
                    text = f"{level}: {count}"
                    y_pos = alert_y_offset + 20 + i * 15
                    cv2.putText(dashboard, text, (20, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Place dashboard on frame
        dashboard_x = width - dashboard_width - 10
        dashboard_y = height - dashboard_height - 10
        
        # Blend dashboard with frame
        roi = output_frame[dashboard_y:dashboard_y + dashboard_height,
                          dashboard_x:dashboard_x + dashboard_width]
        cv2.addWeighted(roi, 0.7, dashboard, 0.3, 0, roi)
        
        return output_frame
    
    def create_behavior_chart(self, behavior_patterns: Dict[int, BehaviorPattern],
                            save_path: str = None) -> np.ndarray:
        """
        Create behavior analysis chart.
        
        Args:
            behavior_patterns: Dictionary of face_id -> BehaviorPattern
            save_path: Optional path to save chart
            
        Returns:
            Chart image
        """
        if not behavior_patterns:
            return np.zeros((400, 600, 3), dtype=np.uint8)
        
        # Prepare data
        face_ids = list(behavior_patterns.keys())
        eye_closure_rates = [pattern.avg_eye_closure_rate for pattern in behavior_patterns.values()]
        yawn_frequencies = [pattern.yawn_frequency for pattern in behavior_patterns.values()]
        attention_scores = [pattern.attention_score for pattern in behavior_patterns.values()]
        
        # Create matplotlib figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Eye closure rates
        ax1.bar(face_ids, eye_closure_rates, alpha=0.8, color='red')
        ax1.set_xlabel('Face ID')
        ax1.set_ylabel('Eye Closure Rate')
        ax1.set_title('Average Eye Closure Rates')
        ax1.grid(True, alpha=0.3)
        
        # Yawn frequencies
        ax2.bar(face_ids, yawn_frequencies, alpha=0.8, color='orange')
        ax2.set_xlabel('Face ID')
        ax2.set_ylabel('Yawn Frequency')
        ax2.set_title('Yawn Frequencies')
        ax2.grid(True, alpha=0.3)
        
        # Attention scores
        ax3.bar(face_ids, attention_scores, alpha=0.8, color='green')
        ax3.set_xlabel('Face ID')
        ax3.set_ylabel('Attention Score (%)')
        ax3.set_title('Attention Scores')
        ax3.grid(True, alpha=0.3)
        
        # Head movement variance
        head_variances = [pattern.head_movement_variance for pattern in behavior_patterns.values()]
        ax4.bar(face_ids, head_variances, alpha=0.8, color='blue')
        ax4.set_xlabel('Face ID')
        ax4.set_ylabel('Head Movement Variance')
        ax4.set_title('Head Movement Variance')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to OpenCV image
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        if save_path:
            cv2.imwrite(save_path, img)
        
        plt.close(fig)
        return img
    
    def create_drowsiness_trend_plot(self, drowsiness_trends: Dict[int, List[float]],
                                   save_path: str = None) -> np.ndarray:
        """
        Create drowsiness trend plot.
        
        Args:
            drowsiness_trends: Dictionary of face_id -> list of scores
            save_path: Optional path to save plot
            
        Returns:
            Plot image
        """
        if not drowsiness_trends:
            return np.zeros((400, 600, 3), dtype=np.uint8)
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for i, (face_id, scores) in enumerate(drowsiness_trends.items()):
            if scores:
                color = colors[i % len(colors)]
                time_points = list(range(len(scores)))
                ax.plot(time_points, scores, label=f'Face {face_id}', 
                       color=color, linewidth=2, marker='o', markersize=3)
        
        ax.set_xlabel('Time Points')
        ax.set_ylabel('Drowsiness Score')
        ax.set_title('Drowsiness Score Trends Over Time')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add threshold lines
        ax.axhline(y=60, color='orange', linestyle='--', alpha=0.7, label='Drowsy Threshold')
        ax.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='Severe Threshold')
        
        plt.tight_layout()
        
        # Convert to OpenCV image
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        if save_path:
            cv2.imwrite(save_path, img)
        
        plt.close(fig)
        return img
    
    def save_annotated_frame(self, frame: np.ndarray,
                           behavior_data_list: List[BehaviorData],
                           drowsiness_scores: List[DrowsinessScore],
                           filename: str):
        """
        Save frame with complete behavior analysis annotations.
        
        Args:
            frame: Input frame
            behavior_data_list: List of behavior data
            drowsiness_scores: List of drowsiness scores
            filename: Output filename
        """
        # Apply all visualizations
        annotated_frame = frame.copy()
        
        # Draw individual behavior analysis for each face
        for i, behavior_data in enumerate(behavior_data_list):
            drowsiness_score = drowsiness_scores[i] if i < len(drowsiness_scores) else None
            annotated_frame = self.draw_behavior_analysis(
                annotated_frame, behavior_data, drowsiness_score
            )
        
        # Draw dashboard
        annotated_frame = self.draw_behavior_dashboard(
            annotated_frame, behavior_data_list, drowsiness_scores
        )
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(annotated_frame, f"Behavior Analysis - {timestamp}",
                   (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imwrite(filename, annotated_frame)
        logger.info(f"Saved behavior analysis frame: {filename}")