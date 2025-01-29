"""
Visualization module for eye tracking with configurable styles and overlay options.
"""

import tkinter as tk
from typing import Optional, Tuple, List, Dict, Any
import platform
import threading
import screeninfo
from dataclasses import dataclass
import cv2
import numpy as np
from .tracker import EyeTracker

@dataclass
class VisualizationStyle:
    """Configuration for gaze visualization appearance."""
    name: str
    sizes: List[int]  # Sizes for concentric circles
    opacities: List[int]  # Opacity levels for circles
    color: str  # Base color (hex)
    history_size: int  # Number of historical points to show
    history_fade: float  # How quickly history fades (0-1)
    show_center: bool  # Whether to show center point
    gradient_steps: int  # Number of gradient steps for smooth effect

# Predefined visualization styles
STYLES = {
    "natural": VisualizationStyle(
        name="natural",
        sizes=[250, 200, 150, 100],  # Even larger circles
        opacities=[60, 80, 100, 120],  # Much higher opacity
        color="#ff0000",  # Pure red
        history_size=6,
        history_fade=0.6,
        show_center=True,
        gradient_steps=2
    ),
    "precise": VisualizationStyle(
        name="precise",
        sizes=[100, 80, 60, 40],
        opacities=[80, 100, 120, 140],
        color="#00ff00",  # Pure green
        history_size=4,
        history_fade=0.7,
        show_center=True,
        gradient_steps=2
    ),
    "minimal": VisualizationStyle(
        name="minimal",
        sizes=[120],
        opacities=[100],
        color="#ffffff",
        history_size=0,
        history_fade=1.0,
        show_center=True,
        gradient_steps=1
    )
}

class GazeOverlay:
    """
    A transparent overlay window that visualizes gaze tracking data.
    
    Example:
        ```python
        overlay = GazeOverlay(
            style="natural",  # or "precise" or "minimal"
            opacity=0.3,
            show_preview=True
        )
        overlay.start()
        ```
    """
    
    def __init__(
        self,
        style: str = "natural",
        opacity: float = 0.3,
        show_preview: bool = True,  # Default to True for debugging
        custom_style: Optional[VisualizationStyle] = None,
        tracker: Optional[EyeTracker] = None
    ):
        """Initialize the overlay window."""
        # Get the primary monitor
        self.screen = screeninfo.get_monitors()[0]
        
        # Set visualization style
        self.style = custom_style or STYLES.get(style, STYLES["natural"])
        
        # Create the main window
        self.root = tk.Tk()
        self.root.title("Gaze Overlay")
        
        # Make it cover the full screen
        self.root.geometry(f"{self.screen.width}x{self.screen.height}+0+0")
        
        # Configure window attributes for macOS
        if platform.system() == "Darwin":  # macOS
            self.root.attributes("-topmost", True)
            self.root.attributes("-alpha", 1.0)  # Full opacity for the window
            self.root.configure(bg='black')
            self.bg_color = 'black'
            self.is_macos = True
        else:  # Windows/Linux
            self.root.attributes("-topmost", True)
            self.root.overrideredirect(True)
            self.root.attributes("-alpha", opacity)
            self.root.attributes("-transparentcolor", "black")
            self.bg_color = 'black'
            self.is_macos = False
        
        # Create a canvas for drawing
        self.canvas = tk.Canvas(
            self.root,
            width=self.screen.width,
            height=self.screen.height,
            highlightthickness=0,
            bg=self.bg_color
        )
        self.canvas.pack(fill="both", expand=True)
        
        # Store opacity for macOS
        self.opacity = opacity
        
        # Variables for gaze visualization
        self.current_gaze: Optional[Tuple[float, float]] = None
        self.gaze_history: List[Tuple[float, float]] = []
        self.show_preview = show_preview
        
        # Add quit button (press 'q' to exit)
        self.root.bind('<Key>', self._on_key)
        
        # Use existing tracker or create new one
        self.tracker = tracker or EyeTracker()
        
        # Start eye tracking in a separate thread
        self.is_running = True
        self.tracking_thread = threading.Thread(target=self._run_eye_tracking)
        self.tracking_thread.daemon = True
        
        # Create a separate window for camera preview
        self.preview_window = None
        if show_preview:
            self.preview_window = cv2.namedWindow("Eye Tracking Preview", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Eye Tracking Preview", 400, 300)  # Smaller preview window
        
        self.latest_frame = None
        self.latest_landmarks = None
    
    def _map_gaze_to_screen(self, gaze_point) -> Tuple[float, float]:
        """Map the gaze coordinates from camera space to screen space."""
        return (
            gaze_point.x * self.screen.width / 640,
            gaze_point.y * self.screen.height / 480
        )
    
    def _on_key(self, event):
        """Handle key events."""
        if event.char == 'q':
            self.stop()
    
    def _draw_gaze_circle(self, x: float, y: float, size: float, opacity: int):
        """Draw a single gaze circle with the current style."""
        base_color = self.style.color
        r = int(base_color[1:3], 16)
        g = int(base_color[3:5], 16)
        b = int(base_color[5:7], 16)
        
        if self.is_macos:
            # On macOS, use fill with adjusted opacity
            opacity = int(opacity * self.opacity)  # Apply window opacity to each circle
            color = f"#{r:02x}{g:02x}{b:02x}"
            
            # Draw filled circle with outline for better visibility
            self.canvas.create_oval(
                x - size/2, y - size/2,
                x + size/2, y + size/2,
                fill=color,
                outline=color,
                width=2,
                stipple='gray50'  # Use stipple pattern for transparency
            )
        else:
            # Original implementation for other platforms
            self.canvas.create_oval(
                x - size/2, y - size/2,
                x + size/2, y + size/2,
                fill=base_color,
                outline="",
                width=0
            )
    
    def _update_visualization(self):
        """Update the gaze visualization on the overlay."""
        self.canvas.delete("all")  # Clear previous drawings
        
        # Draw gaze history
        if self.style.history_size > 0 and self.gaze_history:
            for i, (x, y) in enumerate(self.gaze_history):
                progress = (i + 1) / len(self.gaze_history)
                size_scale = progress * 0.8  # Reduce size of history points
                opacity_scale = progress * self.style.history_fade
                
                for base_size, base_opacity in zip(self.style.sizes, self.style.opacities):
                    size = base_size * size_scale
                    opacity = int(base_opacity * opacity_scale)
                    self._draw_gaze_circle(x, y, size, opacity)
        
        # Draw current gaze point
        if self.current_gaze:
            x, y = self.current_gaze
            
            # Draw main gaze circles
            for size, opacity in zip(self.style.sizes, self.style.opacities):
                self._draw_gaze_circle(x, y, size, opacity)
            
            # Draw center point if enabled
            if self.style.show_center:
                self.canvas.create_oval(
                    x - 3, y - 3, x + 3, y + 3,
                    fill="white", outline="white",
                    width=1
                )
        
        # Schedule next update
        if self.is_running:
            self.root.after(16, self._update_visualization)  # ~60 FPS
    
    def _draw_face_landmarks(self, frame, face_landmarks):
        """Draw face mesh landmarks and eye regions on the preview frame."""
        if face_landmarks is None:
            return frame
        
        frame_height, frame_width = frame.shape[:2]
        
        # Draw face mesh
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        
        # Draw eye regions
        def draw_eye_region(landmarks, indices, color):
            points = np.array([(int(face_landmarks.landmark[i].x * frame_width),
                              int(face_landmarks.landmark[i].y * frame_height))
                             for i in indices], np.int32)
            cv2.polylines(frame, [points], True, color, 2)
            # Draw pupil center
            center = np.mean(points, axis=0).astype(int)
            cv2.circle(frame, tuple(center), 3, (0, 0, 255), -1)
        
        # Draw left and right eyes
        draw_eye_region(face_landmarks, self.LEFT_EYE, (255, 0, 0))  # Blue for left eye
        draw_eye_region(face_landmarks, self.RIGHT_EYE, (0, 255, 255))  # Cyan for right eye
        
        return frame
    
    def _run_eye_tracking(self):
        """Run eye tracking in a separate thread."""
        try:
            for gaze_point, left_eye, right_eye in self.tracker.track():
                if not self.is_running:
                    break
                
                if gaze_point:
                    # Map gaze coordinates to screen space
                    screen_x, screen_y = self._map_gaze_to_screen(gaze_point)
                    self.current_gaze = (screen_x, screen_y)
                    
                    # Update gaze history
                    self.gaze_history.append((screen_x, screen_y))
                    if len(self.gaze_history) > self.style.history_size:
                        self.gaze_history.pop(0)
                else:
                    self.current_gaze = None
                
                # Update preview if enabled
                if self.show_preview and hasattr(self.tracker, 'latest_frame'):
                    preview_frame = self.tracker.latest_frame.copy()
                    if hasattr(self.tracker, 'latest_landmarks'):
                        preview_frame = self._draw_face_landmarks(
                            preview_frame, 
                            self.tracker.latest_landmarks
                        )
                    cv2.imshow("Eye Tracking Preview", preview_frame)
                    cv2.waitKey(1)
        except Exception as e:
            print(f"Error in tracking thread: {e}")
            self.stop()
    
    def start(self):
        """Start the overlay application."""
        print(f"Starting gaze overlay with {self.style.name} style...")
        print("Press 'q' to quit")
        
        # Start the tracking thread
        self.tracking_thread.start()
        
        # Start visualization updates
        self._update_visualization()
        
        # Start the main loop
        self.root.mainloop()
    
    def stop(self):
        """Stop the overlay application."""
        self.is_running = False
        if self.show_preview:
            cv2.destroyAllWindows()
        self.root.quit()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()

    # Add face mesh landmark indices as class attributes
    LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246] 