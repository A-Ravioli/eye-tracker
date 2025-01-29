"""
Desktop application for eye tracking with calibration and gaze visualization.
"""

import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from eye_tracker_pkg import GazeOverlay, EyeTracker
import threading
import json
import os
from pathlib import Path

class EyeTrackingApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Eye Tracking Setup")
        
        # Center the window
        window_width = 600
        window_height = 400
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # Create main container
        self.main_frame = ttk.Frame(self.root, padding="20")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Status variables
        self.calibration_status = tk.StringVar(value="Not calibrated")
        self.camera_status = tk.StringVar(value="Camera not initialized")
        
        # Create UI elements
        self._create_ui()
        
        # Initialize eye tracker
        self.tracker = None
        self.overlay = None
        self.is_tracking = False
        
        # Load previous calibration if available
        self.calibration_file = Path("calibration.json")
        self.has_calibration = self.calibration_file.exists()
        if self.has_calibration:
            self.calibration_status.set("Previous calibration found")
        
        # Start camera check
        self._check_camera()
    
    def _create_ui(self):
        """Create the user interface elements."""
        # Status section
        status_frame = ttk.LabelFrame(self.main_frame, text="Status", padding="10")
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(status_frame, text="Camera:").grid(row=0, column=0, sticky=tk.W)
        ttk.Label(status_frame, textvariable=self.camera_status).grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(status_frame, text="Calibration:").grid(row=1, column=0, sticky=tk.W)
        ttk.Label(status_frame, textvariable=self.calibration_status).grid(row=1, column=1, sticky=tk.W)
        
        # Buttons section
        buttons_frame = ttk.Frame(self.main_frame)
        buttons_frame.pack(fill=tk.X, pady=10)
        
        self.calibrate_btn = ttk.Button(
            buttons_frame, 
            text="Start Calibration",
            command=self._start_calibration
        )
        self.calibrate_btn.pack(side=tk.LEFT, padx=5)
        
        self.start_btn = ttk.Button(
            buttons_frame,
            text="Start Tracking",
            command=self._start_tracking,
            state=tk.DISABLED
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        # Instructions
        instructions = """
Instructions:

1. Ensure your camera is working (status should show "Camera ready")
2. Complete the calibration process:
   - Look at each red dot that appears on screen
   - Keep your head relatively still
   - The process takes about 30 seconds
3. After calibration, click "Start Tracking" to begin
4. Press 'Q' to quit the tracking overlay

Tips:
- Sit at a comfortable distance from the screen
- Ensure good lighting on your face
- Try to minimize head movement during tracking
"""
        instructions_frame = ttk.LabelFrame(self.main_frame, text="Instructions", padding="10")
        instructions_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        ttk.Label(
            instructions_frame, 
            text=instructions,
            justify=tk.LEFT,
            wraplength=500
        ).pack(fill=tk.BOTH, expand=True)
    
    def _check_camera(self):
        """Check if the camera is available and working."""
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                self.camera_status.set("Camera ready")
                if self.has_calibration:
                    self.start_btn.config(state=tk.NORMAL)
            else:
                self.camera_status.set("Camera not available")
            cap.release()
        except Exception as e:
            self.camera_status.set(f"Camera error: {str(e)}")
    
    def _start_calibration(self):
        """Start the calibration process."""
        self.root.withdraw()  # Hide the main window
        
        # Initialize tracker if needed
        if self.tracker is None:
            self.tracker = EyeTracker()
        
        # Create and run calibration window
        calibration = CalibrationWindow(self.tracker)
        calibration.run()
        
        if calibration.success:
            self.calibration_status.set("Calibration successful")
            self.start_btn.config(state=tk.NORMAL)
            self.has_calibration = True
            
            # Save calibration data
            if self.tracker.transform_matrix is not None:
                calibration_data = {
                    "transform_matrix": self.tracker.transform_matrix.tolist()
                }
                with open(self.calibration_file, 'w') as f:
                    json.dump(calibration_data, f)
        else:
            self.calibration_status.set("Calibration failed")
        
        self.root.deiconify()  # Show the main window again
    
    def _start_tracking(self):
        """Start the gaze tracking overlay."""
        if self.is_tracking:
            return
        
        try:
            self.is_tracking = True
            self.root.withdraw()  # Hide the main window
            
            # Load saved calibration if available
            if self.has_calibration and self.tracker is None:
                print("Loading saved calibration...")
                self.tracker = EyeTracker()
                with open(self.calibration_file, 'r') as f:
                    calibration_data = json.load(f)
                    self.tracker.transform_matrix = np.array(calibration_data["transform_matrix"])
                    self.tracker.calibration_complete = True
                print("Calibration loaded successfully")
            
            if not self.tracker:
                print("No tracker available. Please calibrate first.")
                return
            
            print("Starting tracking with calibrated tracker...")
            print(f"Calibration status: {self.tracker.calibration_complete}")
            if hasattr(self.tracker, 'transform_matrix'):
                print("Transform matrix available")
            
            # Create and start overlay with the calibrated tracker
            self.overlay = GazeOverlay(
                style="natural",
                opacity=0.3,
                show_preview=True,
                tracker=self.tracker  # Pass the calibrated tracker
            )
            
            try:
                self.overlay.start()
            except Exception as e:
                print(f"Error during tracking: {str(e)}")
                import traceback
                traceback.print_exc()
        except Exception as e:
            print(f"Error starting tracking: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_tracking = False
            self.root.deiconify()  # Show the main window again
    
    def run(self):
        """Start the application."""
        self.root.mainloop()

class CalibrationWindow:
    def __init__(self, tracker: EyeTracker):
        self.root = tk.Tk()
        self.root.attributes('-fullscreen', True)
        self.root.configure(bg='black')
        
        # Get screen dimensions
        self.width = self.root.winfo_screenwidth()
        self.height = self.root.winfo_screenheight()
        
        # Create canvas
        self.canvas = tk.Canvas(
            self.root, 
            width=self.width, 
            height=self.height,
            bg='black',
            highlightthickness=0
        )
        self.canvas.pack()
        
        # Calibration points (9-point calibration)
        margin = 100  # pixels from edge
        self.points = [
            (margin, margin),  # Top-left
            (self.width//2, margin),  # Top-center
            (self.width-margin, margin),  # Top-right
            (margin, self.height//2),  # Middle-left
            (self.width//2, self.height//2),  # Center
            (self.width-margin, self.height//2),  # Middle-right
            (margin, self.height-margin),  # Bottom-left
            (self.width//2, self.height-margin),  # Bottom-center
            (self.width-margin, self.height-margin),  # Bottom-right
        ]
        
        self.current_point = 0
        self.samples_per_point = 30
        self.current_samples = 0
        
        # Use existing tracker
        self.tracker = tracker
        self.success = False
        
        # Create preview window
        cv2.namedWindow("Calibration Preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Calibration Preview", 400, 300)
        
        # Bind escape key to quit
        self.root.bind('<Escape>', lambda e: self.quit())
        
        # Instructions label
        self.instructions = tk.Label(
            self.root,
            text="Look at the red dot\nKeep your head still\nPress ESC to cancel",
            font=("Arial", 24),
            fg="white",
            bg="black"
        )
        self.instructions.place(relx=0.5, rely=0.1, anchor="center")
        
        print("Starting calibration...")
        print("Look at each point as it appears.")
        print("Press ESC to quit.")
        
        # Start calibration after a short delay
        self.root.after(1000, self.calibration_loop)
    
    def draw_point(self, x, y, size=20, color='red'):
        """Draw a calibration point with animation."""
        # Clear canvas
        self.canvas.delete('all')
        
        # Draw outer circle
        self.canvas.create_oval(
            x - size, y - size,
            x + size, y + size,
            outline=color,
            width=2
        )
        
        # Draw inner circle
        self.canvas.create_oval(
            x - size//4, y - size//4,
            x + size//4, y + size//4,
            fill=color
        )
    
    def calibration_loop(self):
        """Main calibration loop."""
        if self.current_point >= len(self.points):
            # Calibration complete
            print("Computing calibration...")
            self.success = self.tracker.compute_calibration()
            if self.success:
                print("Calibration successful!")
            else:
                print("Calibration failed. Please try again.")
            self.quit()
            return
        
        # Get current point
        x, y = self.points[self.current_point]
        self.draw_point(x, y)
        
        # Read from eye tracker
        ret, frame = self.tracker.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            
            # Process frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.tracker.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                self.tracker.latest_landmarks = results.multi_face_landmarks[0]
                self.tracker.latest_frame = frame
                
                # Add calibration point
                self.tracker.add_calibration_point(x, y)
                self.current_samples += 1
                
                # Draw face mesh on preview
                preview_frame = frame.copy()
                for landmark in results.multi_face_landmarks[0].landmark:
                    lx = int(landmark.x * frame.shape[1])
                    ly = int(landmark.y * frame.shape[0])
                    cv2.circle(preview_frame, (lx, ly), 1, (0, 255, 0), -1)
                
                # Show preview
                cv2.imshow("Calibration Preview", preview_frame)
                cv2.waitKey(1)
            
            if self.current_samples >= self.samples_per_point:
                print(f"Point {self.current_point + 1}/{len(self.points)} complete")
                self.current_point += 1
                self.current_samples = 0
        
        # Continue loop
        self.root.after(16, self.calibration_loop)  # ~60 FPS
    
    def quit(self):
        """Clean up and close windows."""
        cv2.destroyAllWindows()
        self.root.quit()
        self.root.destroy()
    
    def run(self):
        """Start the calibration window."""
        self.root.mainloop()

def main():
    """Run the eye tracking application."""
    app = EyeTrackingApp()
    app.run()

if __name__ == "__main__":
    main() 