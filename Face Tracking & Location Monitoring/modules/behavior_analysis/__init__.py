"""
Facial Behavior Analysis Module

This module provides real-time facial behavior detection including
eye state monitoring, head pose estimation, yawn detection, and drowsiness analysis.
"""

from .behavior_detector import BehaviorDetector
from .drowsiness_analyzer import DrowsinessAnalyzer
from .behavior_visualizer import BehaviorVisualizer

__version__ = "1.0.0"
__author__ = "Behavior Analysis System"

__all__ = [
    "BehaviorDetector",
    "DrowsinessAnalyzer", 
    "BehaviorVisualizer"
]