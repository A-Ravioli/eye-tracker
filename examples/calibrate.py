"""
Calibration script for the eye tracker.
Shows a series of points on screen and collects eye direction data.
"""

import tkinter as tk
from eye_tracker_pkg import EyeTracker
import time
import numpy as np
import cv2

class CalibrationWindow:
    def __init__(self):
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
        
        # Initialize eye tracker
        self.tracker = EyeTracker()
        
        # Create preview window
        cv2.namedWindow("Calibration Preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Calibration Preview", 400, 300)
        
        # Bind escape key to quit
        self.root.bind('<Escape>', lambda e: self.quit())
        
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
            success = self.tracker.compute_calibration()
            if success:
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
        self.tracker.stop()
        cv2.destroyAllWindows()
        self.root.quit()
    
    def run(self):
        """Start the calibration window."""
        self.root.mainloop()

def main():
    """Run the calibration process."""
    window = CalibrationWindow()
    window.run()

if __name__ == "__main__":
    main() 