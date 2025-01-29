import cv2
import mediapipe as mp
import numpy as np

class EyeTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        
        # Get screen dimensions
        self.screen_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.screen_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # MediaPipe indices for left and right eyes
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        
    def get_eye_position(self, face_landmarks, frame):
        """Calculate the center position of both eyes"""
        left_eye_points = np.array([(int(face_landmarks.landmark[i].x * frame.shape[1]),
                                   int(face_landmarks.landmark[i].y * frame.shape[0]))
                                  for i in self.LEFT_EYE])
        
        right_eye_points = np.array([(int(face_landmarks.landmark[i].x * frame.shape[1]),
                                    int(face_landmarks.landmark[i].y * frame.shape[0]))
                                   for i in self.RIGHT_EYE])
        
        # Calculate the center of each eye
        left_eye_center = np.mean(left_eye_points, axis=0).astype(int)
        right_eye_center = np.mean(right_eye_points, axis=0).astype(int)
        
        # Calculate the average center between both eyes
        eye_center = ((left_eye_center + right_eye_center) // 2).astype(int)
        
        return eye_center, left_eye_points, right_eye_points
    
    def run(self):
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                print("Failed to capture frame")
                break
                
            # Flip the frame horizontally for a later selfie-view display
            frame = cv2.flip(frame, 1)
            
            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame and detect faces
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                
                # Get eye positions
                eye_center, left_eye_points, right_eye_points = self.get_eye_position(face_landmarks, frame)
                
                # Draw eye contours
                cv2.polylines(frame, [left_eye_points], True, (0, 255, 0), 1)
                cv2.polylines(frame, [right_eye_points], True, (0, 255, 0), 1)
                
                # Draw gaze point
                cv2.circle(frame, tuple(eye_center), 5, (0, 0, 255), -1)
            
            # Display the frame
            cv2.imshow('Eye Tracking', frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = EyeTracker()
    tracker.run() 