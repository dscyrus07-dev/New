"""
Zone Monitoring Module

This module implements virtual zone monitoring for tracking person movement
between different areas like entry, desk, cafeteria, and exit zones.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
import time
from datetime import datetime
import logging
from shapely.geometry import Point, Polygon
from collections import defaultdict

from .tracker import Track

logger = logging.getLogger(__name__)

@dataclass
class Zone:
    """Zone definition"""
    name: str
    points: List[Tuple[int, int]]
    color: Tuple[int, int, int] = (0, 255, 0)
    polygon: Optional[Polygon] = None
    
    def __post_init__(self):
        """Create polygon from points after initialization"""
        if len(self.points) >= 3:
            self.polygon = Polygon(self.points)
        else:
            logger.warning(f"Zone {self.name} has less than 3 points, cannot create polygon")

@dataclass
class ZoneTransition:
    """Zone transition event"""
    track_id: int
    from_zone: Optional[str]
    to_zone: str
    timestamp: datetime
    confidence: float
    identity: Optional[str] = None

@dataclass
class ZoneOccupancy:
    """Current zone occupancy information"""
    zone_name: str
    track_ids: Set[int] = field(default_factory=set)
    identities: Set[str] = field(default_factory=set)
    last_updated: datetime = field(default_factory=datetime.now)

class ZoneMonitor:
    """
    Zone monitoring system for tracking person movement between defined areas.
    Logs zone transitions with timestamps and provides occupancy information.
    """
    
    def __init__(self, zones_config: Dict):
        """
        Initialize zone monitor with zone configurations.
        
        Args:
            zones_config: Dictionary containing zone definitions
        """
        self.zones: Dict[str, Zone] = {}
        self.zone_occupancy: Dict[str, ZoneOccupancy] = {}
        self.track_zones: Dict[int, str] = {}  # Current zone for each track
        self.zone_transitions: List[ZoneTransition] = []
        self.transition_history: Dict[int, List[ZoneTransition]] = defaultdict(list)
        
        # Load zones from configuration
        self._load_zones(zones_config)
        
        # Statistics
        self.total_transitions = 0
        self.zone_stats: Dict[str, Dict] = defaultdict(lambda: {
            'entries': 0,
            'exits': 0,
            'current_occupancy': 0,
            'max_occupancy': 0,
            'total_time': 0.0
        })
        
        logger.info(f"ZoneMonitor initialized with {len(self.zones)} zones")
    
    def _load_zones(self, zones_config: Dict):
        """Load zone definitions from configuration"""
        for zone_name, zone_data in zones_config.items():
            try:
                zone = Zone(
                    name=zone_data.get('name', zone_name),
                    points=zone_data['points'],
                    color=tuple(zone_data.get('color', [0, 255, 0]))
                )
                
                self.zones[zone_name] = zone
                self.zone_occupancy[zone_name] = ZoneOccupancy(zone_name=zone_name)
                
                logger.info(f"Loaded zone: {zone_name} with {len(zone.points)} points")
                
            except Exception as e:
                logger.error(f"Failed to load zone {zone_name}: {e}")
    
    def update(self, tracks: List[Track]) -> List[ZoneTransition]:
        """
        Update zone monitoring with current tracks.
        
        Args:
            tracks: List of active tracks
            
        Returns:
            List of new zone transitions
        """
        new_transitions = []
        current_time = datetime.now()
        
        # Clear current occupancy
        for occupancy in self.zone_occupancy.values():
            occupancy.track_ids.clear()
            occupancy.identities.clear()
            occupancy.last_updated = current_time
        
        # Process each track
        for track in tracks:
            current_zone = self._get_track_zone(track)
            previous_zone = self.track_zones.get(track.track_id)
            
            # Update occupancy
            if current_zone:
                self.zone_occupancy[current_zone].track_ids.add(track.track_id)
                if track.identity:
                    self.zone_occupancy[current_zone].identities.add(track.identity)
            
            # Check for zone transition
            if current_zone != previous_zone:
                transition = ZoneTransition(
                    track_id=track.track_id,
                    from_zone=previous_zone,
                    to_zone=current_zone,
                    timestamp=current_time,
                    confidence=track.confidence,
                    identity=track.identity
                )
                
                new_transitions.append(transition)
                self.zone_transitions.append(transition)
                self.transition_history[track.track_id].append(transition)
                
                # Update track zone
                if current_zone:
                    self.track_zones[track.track_id] = current_zone
                elif track.track_id in self.track_zones:
                    del self.track_zones[track.track_id]
                
                # Update statistics
                self._update_zone_stats(transition)
                
                logger.info(f"Zone transition: Track {track.track_id} "
                          f"({track.identity or 'Unknown'}) "
                          f"{previous_zone} -> {current_zone}")
        
        # Update zone statistics
        self._update_occupancy_stats()
        
        return new_transitions
    
    def _get_track_zone(self, track: Track) -> Optional[str]:
        """
        Determine which zone a track is currently in.
        
        Args:
            track: Track object
            
        Returns:
            Zone name or None if not in any zone
        """
        # Get center point of bounding box
        x, y, w, h = track.bbox
        center_x = x + w // 2
        center_y = y + h // 2
        point = Point(center_x, center_y)
        
        # Check each zone
        for zone_name, zone in self.zones.items():
            if zone.polygon and zone.polygon.contains(point):
                return zone_name
        
        return None
    
    def _update_zone_stats(self, transition: ZoneTransition):
        """Update zone statistics"""
        self.total_transitions += 1
        
        if transition.to_zone:
            self.zone_stats[transition.to_zone]['entries'] += 1
        
        if transition.from_zone:
            self.zone_stats[transition.from_zone]['exits'] += 1
    
    def _update_occupancy_stats(self):
        """Update occupancy statistics"""
        for zone_name, occupancy in self.zone_occupancy.items():
            current_count = len(occupancy.track_ids)
            self.zone_stats[zone_name]['current_occupancy'] = current_count
            
            if current_count > self.zone_stats[zone_name]['max_occupancy']:
                self.zone_stats[zone_name]['max_occupancy'] = current_count
    
    def get_zone_occupancy(self, zone_name: str) -> ZoneOccupancy:
        """
        Get current occupancy for a specific zone.
        
        Args:
            zone_name: Name of the zone
            
        Returns:
            ZoneOccupancy object
        """
        return self.zone_occupancy.get(zone_name, ZoneOccupancy(zone_name=zone_name))
    
    def get_all_occupancy(self) -> Dict[str, ZoneOccupancy]:
        """Get occupancy for all zones"""
        return self.zone_occupancy.copy()
    
    def get_track_transitions(self, track_id: int) -> List[ZoneTransition]:
        """
        Get all transitions for a specific track.
        
        Args:
            track_id: Track ID
            
        Returns:
            List of transitions
        """
        return self.transition_history.get(track_id, [])
    
    def get_recent_transitions(self, minutes: int = 10) -> List[ZoneTransition]:
        """
        Get recent transitions within specified time window.
        
        Args:
            minutes: Time window in minutes
            
        Returns:
            List of recent transitions
        """
        cutoff_time = datetime.now().timestamp() - (minutes * 60)
        
        return [
            transition for transition in self.zone_transitions
            if transition.timestamp.timestamp() >= cutoff_time
        ]
    
    def get_zone_stats(self) -> Dict[str, Dict]:
        """Get comprehensive zone statistics"""
        return dict(self.zone_stats)
    
    def draw_zones(self, frame: np.ndarray, show_labels: bool = True, 
                   show_occupancy: bool = True) -> np.ndarray:
        """
        Draw zones on the frame.
        
        Args:
            frame: Input frame
            show_labels: Whether to show zone labels
            show_occupancy: Whether to show occupancy count
            
        Returns:
            Frame with zones drawn
        """
        output_frame = frame.copy()
        
        for zone_name, zone in self.zones.items():
            if zone.polygon is None:
                continue
            
            # Draw zone polygon
            points = np.array(zone.points, dtype=np.int32)
            cv2.polylines(output_frame, [points], True, zone.color, 2)
            
            # Fill zone with transparent color
            overlay = output_frame.copy()
            cv2.fillPoly(overlay, [points], zone.color)
            cv2.addWeighted(output_frame, 0.8, overlay, 0.2, 0, output_frame)
            
            if show_labels or show_occupancy:
                # Calculate label position (centroid of polygon)
                label_x = int(np.mean([p[0] for p in zone.points]))
                label_y = int(np.mean([p[1] for p in zone.points]))
                
                label_text = zone.name
                if show_occupancy:
                    occupancy = self.get_zone_occupancy(zone_name)
                    count = len(occupancy.track_ids)
                    label_text += f" ({count})"
                
                # Draw label background
                (text_width, text_height), _ = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                
                cv2.rectangle(output_frame,
                            (label_x - text_width // 2 - 5, label_y - text_height - 5),
                            (label_x + text_width // 2 + 5, label_y + 5),
                            (0, 0, 0), -1)
                
                # Draw label text
                cv2.putText(output_frame, label_text,
                          (label_x - text_width // 2, label_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return output_frame
    
    def export_transitions_csv(self, filename: str, hours: int = 24):
        """
        Export transitions to CSV file.
        
        Args:
            filename: Output CSV filename
            hours: Number of hours of data to export
        """
        import pandas as pd
        
        cutoff_time = datetime.now().timestamp() - (hours * 3600)
        recent_transitions = [
            t for t in self.zone_transitions
            if t.timestamp.timestamp() >= cutoff_time
        ]
        
        if not recent_transitions:
            logger.warning("No transitions to export")
            return
        
        # Convert to DataFrame
        data = []
        for transition in recent_transitions:
            data.append({
                'timestamp': transition.timestamp,
                'track_id': transition.track_id,
                'identity': transition.identity or 'Unknown',
                'from_zone': transition.from_zone or 'Outside',
                'to_zone': transition.to_zone or 'Outside',
                'confidence': transition.confidence
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        logger.info(f"Exported {len(data)} transitions to {filename}")
    
    def get_movement_patterns(self, track_id: int) -> Dict[str, Any]:
        """
        Analyze movement patterns for a specific track.
        
        Args:
            track_id: Track ID to analyze
            
        Returns:
            Dictionary containing movement analysis
        """
        transitions = self.get_track_transitions(track_id)
        
        if not transitions:
            return {'error': 'No transitions found for track'}
        
        # Calculate dwell times
        zone_times = defaultdict(float)
        current_zone = None
        zone_start_time = None
        
        for transition in transitions:
            if current_zone and zone_start_time:
                dwell_time = (transition.timestamp - zone_start_time).total_seconds()
                zone_times[current_zone] += dwell_time
            
            current_zone = transition.to_zone
            zone_start_time = transition.timestamp
        
        # Most visited zones
        zone_visits = defaultdict(int)
        for transition in transitions:
            if transition.to_zone:
                zone_visits[transition.to_zone] += 1
        
        return {
            'total_transitions': len(transitions),
            'zone_dwell_times': dict(zone_times),
            'zone_visit_counts': dict(zone_visits),
            'first_seen': transitions[0].timestamp,
            'last_seen': transitions[-1].timestamp,
            'most_visited_zone': max(zone_visits.items(), key=lambda x: x[1])[0] if zone_visits else None
        }
    
    def cleanup_old_data(self, hours: int = 24):
        """
        Remove old transition data to free memory.
        
        Args:
            hours: Keep data newer than this many hours
        """
        cutoff_time = datetime.now().timestamp() - (hours * 3600)
        
        # Filter transitions
        self.zone_transitions = [
            t for t in self.zone_transitions
            if t.timestamp.timestamp() >= cutoff_time
        ]
        
        # Filter transition history
        for track_id in list(self.transition_history.keys()):
            self.transition_history[track_id] = [
                t for t in self.transition_history[track_id]
                if t.timestamp.timestamp() >= cutoff_time
            ]
            
            if not self.transition_history[track_id]:
                del self.transition_history[track_id]
        
        logger.info(f"Cleaned up transition data older than {hours} hours")