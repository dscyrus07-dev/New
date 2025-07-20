"""
Visualization Module for Face Tracking & Location Monitoring

This module provides visualization tools for drawing tracks, zones,
and movement paths on video frames.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime
import logging

from .tracker import Track
from .zone_monitor import Zone, ZoneOccupancy

logger = logging.getLogger(__name__)

class TrackingVisualizer:
    """
    Visualization tools for face tracking and zone monitoring.
    Provides methods to draw tracks, zones, and statistics on video frames.
    """
    
    def __init__(self, 
                 show_track_id: bool = True,
                 show_identity: bool = True,
                 show_confidence: bool = True,
                 show_trail: bool = True,
                 trail_length: int = 30):
        """
        Initialize the visualizer.
        
        Args:
            show_track_id: Whether to show track IDs
            show_identity: Whether to show identity labels
            show_confidence: Whether to show confidence scores
            show_trail: Whether to show movement trails
            trail_length: Length of movement trails
        """
        self.show_track_id = show_track_id
        self.show_identity = show_identity
        self.show_confidence = show_confidence
        self.show_trail = show_trail
        self.trail_length = trail_length
        
        # Color palette for tracks
        self.colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
            (0, 128, 128),  # Teal
            (128, 128, 0),  # Olive
        ]
        
        logger.info("TrackingVisualizer initialized")
    
    def get_track_color(self, track_id: int) -> Tuple[int, int, int]:
        """
        Get consistent color for a track ID.
        
        Args:
            track_id: Track ID
            
        Returns:
            RGB color tuple
        """
        return self.colors[track_id % len(self.colors)]
    
    def draw_tracks(self, frame: np.ndarray, tracks: List[Track]) -> np.ndarray:
        """
        Draw tracks on the frame.
        
        Args:
            frame: Input frame
            tracks: List of tracks to draw
            
        Returns:
            Frame with tracks drawn
        """
        output_frame = frame.copy()
        
        for track in tracks:
            color = self.get_track_color(track.track_id)
            x, y, w, h = track.bbox
            
            # Draw bounding box
            cv2.rectangle(output_frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw movement trail
            if self.show_trail and len(track.center_history) > 1:
                self._draw_trail(output_frame, track.center_history, color)
            
            # Prepare label text
            label_parts = []
            
            if self.show_track_id:
                label_parts.append(f"ID:{track.track_id}")
            
            if self.show_identity and track.identity:
                label_parts.append(track.identity)
            elif self.show_identity:
                label_parts.append("Unknown")
            
            if self.show_confidence:
                if track.identity_confidence > 0:
                    label_parts.append(f"{track.identity_confidence:.2f}")
                else:
                    label_parts.append(f"{track.confidence:.2f}")
            
            # Draw label
            if label_parts:
                label = " | ".join(label_parts)
                self._draw_label(output_frame, (x, y - 10), label, color)
        
        return output_frame
    
    def _draw_trail(self, frame: np.ndarray, trail: List[Tuple[int, int]], color: Tuple[int, int, int]):
        """Draw movement trail"""
        if len(trail) < 2:
            return
        
        # Draw trail with fading effect
        for i in range(1, len(trail)):
            alpha = i / len(trail)  # Fade from 0 to 1
            thickness = max(1, int(3 * alpha))
            
            # Fade color
            faded_color = tuple(int(c * alpha) for c in color)
            
            cv2.line(frame, trail[i-1], trail[i], faded_color, thickness)
        
        # Draw current position
        if trail:
            cv2.circle(frame, trail[-1], 4, color, -1)
    
    def _draw_label(self, frame: np.ndarray, position: Tuple[int, int], 
                   text: str, color: Tuple[int, int, int]):
        """Draw text label with background"""
        x, y = position
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        # Draw background rectangle
        cv2.rectangle(frame,
                     (x - 2, y - text_height - baseline - 2),
                     (x + text_width + 2, y + baseline + 2),
                     (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def draw_zones_with_occupancy(self, frame: np.ndarray, zones: Dict[str, Zone], 
                                 occupancy: Dict[str, ZoneOccupancy]) -> np.ndarray:
        """
        Draw zones with occupancy information.
        
        Args:
            frame: Input frame
            zones: Dictionary of zones
            occupancy: Dictionary of zone occupancy
            
        Returns:
            Frame with zones drawn
        """
        output_frame = frame.copy()
        
        for zone_name, zone in zones.items():
            if zone.polygon is None:
                continue
            
            zone_occupancy = occupancy.get(zone_name)
            occupancy_count = len(zone_occupancy.track_ids) if zone_occupancy else 0
            
            # Draw zone polygon
            points = np.array(zone.points, dtype=np.int32)
            
            # Color intensity based on occupancy
            base_color = zone.color
            if occupancy_count > 0:
                # Brighten color based on occupancy
                intensity = min(1.0, occupancy_count / 5.0)  # Max intensity at 5 people
                color = tuple(int(c * (0.5 + 0.5 * intensity)) for c in base_color)
            else:
                color = tuple(int(c * 0.3) for c in base_color)
            
            # Draw zone outline
            cv2.polylines(output_frame, [points], True, color, 2)
            
            # Fill zone with transparent color
            overlay = output_frame.copy()
            cv2.fillPoly(overlay, [points], color)
            cv2.addWeighted(output_frame, 0.85, overlay, 0.15, 0, output_frame)
            
            # Draw zone label and occupancy
            label_x = int(np.mean([p[0] for p in zone.points]))
            label_y = int(np.mean([p[1] for p in zone.points]))
            
            label_text = f"{zone.name}"
            occupancy_text = f"Occupancy: {occupancy_count}"
            
            # Draw zone name
            self._draw_label(output_frame, (label_x - 50, label_y - 10), label_text, color)
            
            # Draw occupancy count
            self._draw_label(output_frame, (label_x - 50, label_y + 15), occupancy_text, color)
            
            # Draw identities if available
            if zone_occupancy and zone_occupancy.identities:
                identities_text = ", ".join(list(zone_occupancy.identities)[:3])  # Show max 3
                if len(zone_occupancy.identities) > 3:
                    identities_text += "..."
                self._draw_label(output_frame, (label_x - 50, label_y + 35), 
                               identities_text, (255, 255, 255))
        
        return output_frame
    
    def draw_statistics_overlay(self, frame: np.ndarray, stats: Dict) -> np.ndarray:
        """
        Draw statistics overlay on frame.
        
        Args:
            frame: Input frame
            stats: Statistics dictionary
            
        Returns:
            Frame with statistics overlay
        """
        output_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = np.zeros((150, 300, 3), dtype=np.uint8)
        overlay.fill(50)  # Dark gray background
        
        # Statistics text
        y_offset = 25
        line_height = 20
        
        stats_text = [
            f"Active Tracks: {stats.get('active_tracks', 0)}",
            f"FPS: {stats.get('fps', 0.0):.1f}",
            f"Total Transitions: {stats.get('total_transitions', 0)}",
            f"Known Faces: {stats.get('known_faces', 0)}",
            f"Zones: {stats.get('zones', 0)}",
            f"Uptime: {stats.get('uptime', '00:00:00')}"
        ]
        
        for i, text in enumerate(stats_text):
            y_pos = y_offset + i * line_height
            cv2.putText(overlay, text, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Place overlay on frame
        overlay_x = width - 320
        overlay_y = 10
        
        # Blend overlay with frame
        roi = output_frame[overlay_y:overlay_y + overlay.shape[0], 
                          overlay_x:overlay_x + overlay.shape[1]]
        cv2.addWeighted(roi, 0.7, overlay, 0.3, 0, roi)
        
        return output_frame
    
    def draw_zone_transitions_log(self, frame: np.ndarray, 
                                 recent_transitions: List, max_entries: int = 5) -> np.ndarray:
        """
        Draw recent zone transitions log.
        
        Args:
            frame: Input frame
            recent_transitions: List of recent transitions
            max_entries: Maximum number of entries to show
            
        Returns:
            Frame with transitions log
        """
        output_frame = frame.copy()
        height, width = frame.shape[:2]
        
        if not recent_transitions:
            return output_frame
        
        # Create log overlay
        log_height = min(max_entries, len(recent_transitions)) * 25 + 30
        overlay = np.zeros((log_height, 400, 3), dtype=np.uint8)
        overlay.fill(30)  # Dark background
        
        # Header
        cv2.putText(overlay, "Recent Zone Transitions", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        # Show recent transitions
        for i, transition in enumerate(recent_transitions[:max_entries]):
            y_pos = 45 + i * 25
            
            # Format transition text
            from_zone = transition.from_zone or "Outside"
            to_zone = transition.to_zone or "Outside"
            identity = transition.identity or f"Track{transition.track_id}"
            time_str = transition.timestamp.strftime("%H:%M:%S")
            
            text = f"{time_str} | {identity}: {from_zone} -> {to_zone}"
            
            # Truncate if too long
            if len(text) > 45:
                text = text[:42] + "..."
            
            cv2.putText(overlay, text, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Place overlay on frame (bottom left)
        overlay_x = 10
        overlay_y = height - log_height - 10
        
        # Blend overlay with frame
        roi = output_frame[overlay_y:overlay_y + overlay.shape[0], 
                          overlay_x:overlay_x + overlay.shape[1]]
        cv2.addWeighted(roi, 0.8, overlay, 0.2, 0, roi)
        
        return output_frame
    
    def create_heatmap(self, frame_shape: Tuple[int, int], 
                      track_positions: List[Tuple[int, int]], 
                      save_path: str = None) -> np.ndarray:
        """
        Create heatmap of track positions.
        
        Args:
            frame_shape: Shape of the frame (height, width)
            track_positions: List of (x, y) positions
            save_path: Optional path to save heatmap
            
        Returns:
            Heatmap image
        """
        height, width = frame_shape
        
        # Create heatmap data
        heatmap_data = np.zeros((height, width), dtype=np.float32)
        
        # Add Gaussian blobs for each position
        sigma = 20  # Gaussian sigma
        for x, y in track_positions:
            if 0 <= x < width and 0 <= y < height:
                # Create Gaussian kernel
                kernel_size = int(6 * sigma)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                
                # Calculate bounds
                x_min = max(0, x - kernel_size // 2)
                x_max = min(width, x + kernel_size // 2 + 1)
                y_min = max(0, y - kernel_size // 2)
                y_max = min(height, y + kernel_size // 2 + 1)
                
                # Create meshgrid for the region
                xx, yy = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
                
                # Calculate Gaussian
                gaussian = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
                
                # Add to heatmap
                heatmap_data[y_min:y_max, x_min:x_max] += gaussian
        
        # Normalize heatmap
        if heatmap_data.max() > 0:
            heatmap_data = heatmap_data / heatmap_data.max()
        
        # Apply colormap
        heatmap_colored = plt.cm.hot(heatmap_data)
        heatmap_bgr = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
        heatmap_bgr = cv2.cvtColor(heatmap_bgr, cv2.COLOR_RGB2BGR)
        
        if save_path:
            cv2.imwrite(save_path, heatmap_bgr)
        
        return heatmap_bgr
    
    def create_movement_plot(self, track_histories: Dict[int, List[Tuple[int, int]]], 
                           frame_shape: Tuple[int, int], save_path: str = None) -> np.ndarray:
        """
        Create movement plot showing track paths.
        
        Args:
            track_histories: Dictionary of track_id -> list of positions
            frame_shape: Shape of the frame
            save_path: Optional path to save plot
            
        Returns:
            Movement plot image
        """
        height, width = frame_shape
        
        # Create blank image
        plot_img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw each track's path
        for track_id, positions in track_histories.items():
            if len(positions) < 2:
                continue
            
            color = self.get_track_color(track_id)
            
            # Draw path
            for i in range(1, len(positions)):
                cv2.line(plot_img, positions[i-1], positions[i], color, 2)
            
            # Mark start and end points
            if positions:
                cv2.circle(plot_img, positions[0], 5, (0, 255, 0), -1)  # Green start
                cv2.circle(plot_img, positions[-1], 5, (0, 0, 255), -1)  # Red end
        
        if save_path:
            cv2.imwrite(save_path, plot_img)
        
        return plot_img
    
    def create_zone_activity_chart(self, zone_stats: Dict[str, Dict], 
                                  save_path: str = None) -> np.ndarray:
        """
        Create zone activity chart.
        
        Args:
            zone_stats: Dictionary of zone statistics
            save_path: Optional path to save chart
            
        Returns:
            Chart image
        """
        if not zone_stats:
            return np.zeros((400, 600, 3), dtype=np.uint8)
        
        # Prepare data
        zone_names = list(zone_stats.keys())
        entries = [stats.get('entries', 0) for stats in zone_stats.values()]
        exits = [stats.get('exits', 0) for stats in zone_stats.values()]
        occupancy = [stats.get('current_occupancy', 0) for stats in zone_stats.values()]
        
        # Create matplotlib figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Bar chart for entries/exits
        x = np.arange(len(zone_names))
        width = 0.35
        
        ax1.bar(x - width/2, entries, width, label='Entries', alpha=0.8)
        ax1.bar(x + width/2, exits, width, label='Exits', alpha=0.8)
        ax1.set_xlabel('Zones')
        ax1.set_ylabel('Count')
        ax1.set_title('Zone Entries and Exits')
        ax1.set_xticks(x)
        ax1.set_xticklabels(zone_names, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Bar chart for current occupancy
        ax2.bar(zone_names, occupancy, alpha=0.8, color='orange')
        ax2.set_xlabel('Zones')
        ax2.set_ylabel('Current Occupancy')
        ax2.set_title('Current Zone Occupancy')
        ax2.grid(True, alpha=0.3)
        
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
    
    def save_frame_with_annotations(self, frame: np.ndarray, tracks: List[Track],
                                   zones: Dict[str, Zone], occupancy: Dict[str, ZoneOccupancy],
                                   stats: Dict, filename: str):
        """
        Save annotated frame with all visualizations.
        
        Args:
            frame: Input frame
            tracks: List of tracks
            zones: Dictionary of zones
            occupancy: Zone occupancy data
            stats: Statistics dictionary
            filename: Output filename
        """
        # Apply all visualizations
        annotated_frame = self.draw_tracks(frame, tracks)
        annotated_frame = self.draw_zones_with_occupancy(annotated_frame, zones, occupancy)
        annotated_frame = self.draw_statistics_overlay(annotated_frame, stats)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(annotated_frame, timestamp, (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imwrite(filename, annotated_frame)
        logger.info(f"Saved annotated frame: {filename}")