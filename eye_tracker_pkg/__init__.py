"""
Eye Tracker Package
------------------

A simple and easy to use eye tracking library that uses your webcam
to track eye movements and estimate gaze position.
"""

from .tracker import EyeTracker
from .visualization import GazeOverlay, VisualizationStyle
from .types import GazePoint, EyePosition

__version__ = "0.1.0"
__all__ = ["EyeTracker", "GazeOverlay", "VisualizationStyle", "GazePoint", "EyePosition"] 