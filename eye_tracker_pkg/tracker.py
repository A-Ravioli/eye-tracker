import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, Optional, List, Generator, Dict
from .types import GazePoint, EyePosition

class EyeTracker:
    """
    A class for tracking eye movements and estimating gaze position using a webcam.
    
    Example:
        ```python
        from eye_tracker_pkg import EyeTracker
        
        # Initialize the tracker
        tracker = EyeTracker()
        
        # Start tracking in callback mode
        def on_gaze_update(gaze_point, left_eye, right_eye):
            print(f"Looking at: ({gaze_point.x}, {gaze_point.y})")
        
        tracker.start(callback=on_gaze_update)
        
        # Or use it in a loop
        for gaze_data in tracker.track():
            gaze_point = gaze_data.gaze_point
            print(f"Looking at: ({gaze_point.x}, {gaze_point.y})")
        ```
    """
    
    def __init__(self, camera_id: int = 0, min_detection_confidence: float = 0.5):
        """
        Initialize the eye tracker.
        
        Args:
            camera_id: ID of the camera to use (default: 0 for built-in webcam)
            min_detection_confidence: Minimum confidence for face detection (default: 0.5)
        """
        # Initialize video capture first to get dimensions
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera with ID {camera_id}")
        
        # Set lower resolution for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Get actual dimensions (might be different from requested)
        self.screen_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.screen_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Camera initialized with resolution: {self.screen_width}x{self.screen_height}")
        
        # Initialize MediaPipe Face Mesh with more stable settings
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence
        )
        
        # Initialize drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # MediaPipe indices for eyes and iris
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        
        # Iris landmarks (from MediaPipe Face Mesh)
        self.LEFT_IRIS = [474, 475, 476, 477]   # Left eye iris landmarks
        self.RIGHT_IRIS = [469, 470, 471, 472]  # Right eye iris landmarks
        
        # Store latest frame and landmarks for visualization
        self.latest_frame = None
        self.latest_landmarks = None
        
        # Calibration data
        self.calibration_points: Dict[Tuple[int, int], List[Tuple[float, float, float, float]]] = {}
        self.calibration_complete = False
        self.transform_matrix = None
        
        # Gaze smoothing
        self.gaze_history = []
        self.smoothing_window = 5
        
        # Error handling
        self.consecutive_failures = 0
        self.max_failures = 10
        
        # Test MediaPipe initialization
        test_frame_success, test_frame = self.cap.read()
        if test_frame_success:
            test_frame = cv2.flip(test_frame, 1)
            rgb_frame = cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB)
            try:
                results = self.face_mesh.process(rgb_frame)
                print("MediaPipe Face Mesh initialized successfully")
                if results.multi_face_landmarks:
                    print("Face detection working")
            except Exception as e:
                print(f"Warning: MediaPipe initialization test failed: {e}")
                print("Trying to reinitialize MediaPipe...")
                # Try reinitializing with different settings
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=True,  # Try static mode
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.7,  # Higher confidence threshold
                    min_tracking_confidence=0.7
                )
                try:
                    results = self.face_mesh.process(rgb_frame)
                    print("MediaPipe Face Mesh reinitialized successfully")
                except Exception as e:
                    print(f"Error: MediaPipe reinitialization also failed: {e}")
                    raise RuntimeError("Failed to initialize MediaPipe Face Mesh")
    
    def _calculate_eye_direction(self, landmarks, eye_indices, iris_indices) -> Tuple[float, float]:
        """Calculate the eye direction vector using iris and eye corner positions."""
        # Get eye corner points (leftmost and rightmost points of eye)
        eye_points = np.array([(landmarks.landmark[i].x, landmarks.landmark[i].y) for i in eye_indices])
        left_corner = eye_points[eye_points[:, 0].argmin()]
        right_corner = eye_points[eye_points[:, 0].argmax()]
        
        # Get iris center
        iris_points = np.array([(landmarks.landmark[i].x, landmarks.landmark[i].y) for i in iris_indices])
        iris_center = np.mean(iris_points, axis=0)
        
        # Calculate eye width and iris position relative to corners
        eye_width = np.linalg.norm(right_corner - left_corner)
        iris_x = (iris_center[0] - left_corner[0]) / eye_width  # Normalized x position (0 to 1)
        iris_y = iris_center[1]  # Keep absolute Y position
        
        return iris_x, iris_y
    
    def _process_frame(self, frame) -> Tuple[Optional[GazePoint], Optional[EyePosition], Optional[EyePosition]]:
        """Process a single frame and return gaze and eye position data."""
        try:
            # Store the latest frame
            self.latest_frame = frame.copy()
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            try:
                results = self.face_mesh.process(rgb_frame)
            except Exception as e:
                print(f"Error in MediaPipe processing: {e}")
                return None, None, None
            
            if not results.multi_face_landmarks:
                self.consecutive_failures += 1
                if self.consecutive_failures > self.max_failures:
                    print("Warning: Face detection consistently failing")
                self.latest_landmarks = None
                return None, None, None
            
            self.consecutive_failures = 0  # Reset on successful detection
            
            # Store the latest landmarks
            self.latest_landmarks = results.multi_face_landmarks[0]
            
            # Draw face mesh on preview frame
            if self.latest_frame is not None:
                try:
                    self.mp_drawing.draw_landmarks(
                        image=self.latest_frame,
                        landmark_list=self.latest_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                except Exception as e:
                    print(f"Warning: Could not draw landmarks: {e}")
            
            try:
                left_eye = self._get_eye_position(self.latest_landmarks, frame, self.LEFT_EYE, True)
                right_eye = self._get_eye_position(self.latest_landmarks, frame, self.RIGHT_EYE, False)
            except Exception as e:
                print(f"Error getting eye positions: {e}")
                return None, None, None
            
            try:
                # Calculate gaze directions for both eyes
                left_x, left_y = self._calculate_eye_direction(
                    self.latest_landmarks, self.LEFT_EYE, self.LEFT_IRIS)
                right_x, right_y = self._calculate_eye_direction(
                    self.latest_landmarks, self.RIGHT_EYE, self.RIGHT_IRIS)
                
                # Average the directions from both eyes
                avg_x = (left_x + right_x) / 2
                avg_y = (left_y + right_y) / 2
                
                if self.calibration_complete and self.transform_matrix is not None:
                    # Apply calibration transform
                    screen_coords = np.dot(self.transform_matrix, 
                                         np.array([avg_x, avg_y, 1.0]))
                    screen_x = screen_coords[0] / screen_coords[2]
                    screen_y = screen_coords[1] / screen_coords[2]
                    
                    # Apply smoothing
                    self.gaze_history.append((screen_x, screen_y))
                    if len(self.gaze_history) > self.smoothing_window:
                        self.gaze_history.pop(0)
                    
                    smooth_x = np.mean([p[0] for p in self.gaze_history])
                    smooth_y = np.mean([p[1] for p in self.gaze_history])
                    
                    gaze_point = GazePoint(smooth_x, smooth_y, 1.0)
                else:
                    # Return raw gaze direction when not calibrated
                    gaze_point = GazePoint(avg_x * self.screen_width, 
                                         avg_y * self.screen_height, 0.5)
            except Exception as e:
                print(f"Error calculating gaze: {e}")
                import traceback
                traceback.print_exc()
                gaze_point = None
            
            return gaze_point, left_eye, right_eye
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def add_calibration_point(self, screen_x: int, screen_y: int):
        """Add a calibration point during the calibration phase."""
        if self.latest_landmarks:
            # Get current eye directions
            left_x, left_y = self._calculate_eye_direction(
                self.latest_landmarks, self.LEFT_EYE, self.LEFT_IRIS)
            right_x, right_y = self._calculate_eye_direction(
                self.latest_landmarks, self.RIGHT_EYE, self.RIGHT_IRIS)
            
            # Average the directions
            avg_x = (left_x + right_x) / 2
            avg_y = (left_y + right_y) / 2
            
            # Store calibration data
            point = (screen_x, screen_y)
            if point not in self.calibration_points:
                self.calibration_points[point] = []
            self.calibration_points[point].append((avg_x, avg_y))
    
    def compute_calibration(self):
        """Compute the calibration transform matrix."""
        if len(self.calibration_points) < 4:
            return False  # Need at least 4 points for good calibration
        
        # Prepare matrices for solving the homography
        src_points = []  # Eye directions
        dst_points = []  # Screen coordinates
        
        for (screen_x, screen_y), measures in self.calibration_points.items():
            if measures:  # Use average of measurements for each point
                avg_x = np.mean([m[0] for m in measures])
                avg_y = np.mean([m[1] for m in measures])
                src_points.append([avg_x, avg_y])
                dst_points.append([screen_x, screen_y])
        
        if len(src_points) >= 4:
            # Compute homography matrix
            src_points = np.array(src_points, dtype=np.float32)
            dst_points = np.array(dst_points, dtype=np.float32)
            self.transform_matrix, _ = cv2.findHomography(
                src_points, dst_points, cv2.RANSAC, 5.0)
            self.calibration_complete = True
            return True
        
        return False
    
    def _get_eye_position(self, face_landmarks, frame, eye_indices: List[int], is_left: bool) -> Optional[EyePosition]:
        """Calculate the position and shape of an eye."""
        points = [(int(face_landmarks.landmark[i].x * frame.shape[1]),
                  int(face_landmarks.landmark[i].y * frame.shape[0]))
                 for i in eye_indices]
        
        landmarks = [(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y)
                    for i in eye_indices]
        
        points_array = np.array(points)
        center = np.mean(points_array, axis=0).astype(int)
        
        return EyePosition(
            center=tuple(center),
            contour=points_array,
            landmarks=landmarks,
            is_left=is_left
        )
    
    def track(self) -> Generator[Tuple[Optional[GazePoint], Optional[EyePosition], Optional[EyePosition]], None, None]:
        """
        Generator that yields eye tracking data for each frame.
        
        Yields:
            Tuple[Optional[GazePoint], Optional[EyePosition], Optional[EyePosition]]:
                Gaze point and eye positions for each frame
        """
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break
            
            frame = cv2.flip(frame, 1)  # Mirror image for selfie view
            yield self._process_frame(frame)
    
    def start(self, callback, show_preview: bool = False):
        """
        Start tracking in callback mode.
        
        Args:
            callback: Function to call for each frame with signature:
                     callback(gaze_point: Optional[GazePoint],
                             left_eye: Optional[EyePosition],
                             right_eye: Optional[EyePosition])
            show_preview: If True, shows a preview window with the camera feed
        """
        for gaze_data in self.track():
            callback(*gaze_data)
    
    def _draw_preview(self, frame, gaze_point: Optional[GazePoint],
                     left_eye: Optional[EyePosition], right_eye: Optional[EyePosition]):
        """Draw debug visualization on the preview frame."""
        if left_eye:
            cv2.polylines(frame, [left_eye.contour], True, (0, 255, 0), 1)
        if right_eye:
            cv2.polylines(frame, [right_eye.contour], True, (0, 255, 0), 1)
        if gaze_point:
            cv2.circle(frame, (int(gaze_point.x), int(gaze_point.y)), 5, (0, 0, 255), -1)
    
    def stop(self):
        """Stop tracking and release resources."""
        self.cap.release()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop() 