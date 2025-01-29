from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

@dataclass
class GazePoint:
    """Represents a point where the user is looking on the screen."""
    x: float
    y: float
    confidence: float

    def as_tuple(self) -> Tuple[float, float]:
        """Return the gaze point as a (x, y) tuple."""
        return (self.x, self.y)

@dataclass
class EyePosition:
    """Represents the position and shape of an eye."""
    center: Tuple[int, int]
    contour: np.ndarray
    landmarks: List[Tuple[float, float]]
    is_left: bool
    
    @property
    def x(self) -> int:
        """X coordinate of eye center."""
        return self.center[0]
    
    @property
    def y(self) -> int:
        """Y coordinate of eye center."""
        return self.center[1] 