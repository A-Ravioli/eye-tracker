"""
Simple test script to verify camera access and face detection.
"""

import cv2
import mediapipe as mp
import numpy as np

def main():
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Initialize camera
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    print("Camera opened successfully")
    print("Press 'q' to quit")

    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Flip frame horizontally for selfie view
        frame = cv2.flip(frame, 1)

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = face_mesh.process(rgb_frame)

        # Draw face mesh if detected
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw all landmarks
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

                # Draw eye regions
                LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
                RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

                def draw_eye(indices, color):
                    points = np.array([(int(face_landmarks.landmark[i].x * frame.shape[1]),
                                      int(face_landmarks.landmark[i].y * frame.shape[0]))
                                     for i in indices], np.int32)
                    cv2.polylines(frame, [points], True, color, 2)
                    center = np.mean(points, axis=0).astype(int)
                    cv2.circle(frame, tuple(center), 3, (0, 0, 255), -1)

                draw_eye(LEFT_EYE, (255, 0, 0))   # Blue for left eye
                draw_eye(RIGHT_EYE, (0, 255, 255)) # Cyan for right eye

        # Add text to show if face is detected
        status = "Face Detected" if results.multi_face_landmarks else "No Face Detected"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow('Camera Test', frame)

        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 