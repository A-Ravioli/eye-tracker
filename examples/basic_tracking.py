"""
Basic example showing how to use the eye tracker package.
"""

from eye_tracker_pkg import EyeTracker

def main():
    print("Starting eye tracking...")
    print("Press 'q' to quit")
    
    # Example using callback
    def on_gaze_update(gaze_point, left_eye, right_eye):
        if gaze_point:
            print(f"Looking at: ({gaze_point.x:.1f}, {gaze_point.y:.1f})")
    
    # Initialize tracker and start tracking with preview window
    with EyeTracker() as tracker:
        tracker.start(callback=on_gaze_update, show_preview=True)

if __name__ == "__main__":
    main() 