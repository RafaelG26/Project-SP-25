import cv2                    # Import OpenCV for image processing
import mediapipe as mp         # Import Mediapipe for hand and face detection
import numpy as np             # Import NumPy for array manipulation

# Initialize Mediapipe for hand and face detection
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection

# Initialize the hand detection model with minimum confidence thresholds
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize the face detection model with minimum confidence thresholds
face_detector = mp_face.FaceDetection(min_detection_confidence=0.7)

# OpenCV video capture (for webcam)
cap = cv2.VideoCapture(0)  # Use webcam (0 for default webcam)

while cap.isOpened():  # Check if the video capture is open
    ret, frame = cap.read()  # Capture frame from the video feed

    # Flip the frame to make it mirror-like (common in webcam applications)
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB since Mediapipe works in RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for hand landmarks
    results = hands.process(rgb_frame)

    # Process the frame for face detection
    face_results = face_detector.process(rgb_frame)

    # If face detections are found, draw bounding boxes around faces
    if face_results.detections:
        for detection in face_results.detections:
            # Get bounding box coordinates for the detected face
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape  # Get image height and width
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw a rectangle around the face
    
    # If hand landmarks are found, draw them on the frame
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Iterate through each landmark of the hand and draw a circle at each landmark
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Draw circle at each landmark

    # Display the processed frame with hand and face detections
    cv2.imshow("Hand and Face Detection", frame)

    # Exit condition: Press 'q' to quit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
