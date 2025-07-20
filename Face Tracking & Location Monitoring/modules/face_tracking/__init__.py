"""
Face Tracking & Location Monitoring Module

This module provides real-time face tracking with DeepSort and zone monitoring capabilities.
"""

from .tracker import FaceTracker
from .zone_monitor import ZoneMonitor
from .database_manager import TrackingDatabase
from .visualization import TrackingVisualizer

__version__ = "1.0.0"
__author__ = "Face Tracking System"

__all__ = [
    "FaceTracker",
    "ZoneMonitor", 
    "TrackingDatabase",
    "TrackingVisualizer"
]