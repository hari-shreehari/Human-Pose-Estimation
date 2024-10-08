import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, 
                    min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Ignoring empty camera frame.")
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False

    results = pose.process(rgb_frame)

    rgb_frame.flags.writeable = True
    frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

    side_panel = np.zeros((frame.shape[0], 300, 3), dtype=np.uint8)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))

        h, w, _ = frame.shape

        labels = ['Nose', 'Left Eye (inner)', 'Left Eye', 'Left Eye (outer)', 
                  'Right Eye (inner)', 'Right Eye', 'Right Eye (outer)', 
                  'Left Ear', 'Right Ear', 'Mouth (left)', 'Mouth (right)', 
                  'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow', 
                  'Left Wrist', 'Right Wrist', 'Left Pinky', 'Right Pinky', 
                  'Left Index', 'Right Index', 'Left Thumb', 'Right Thumb', 
                  'Left Hip', 'Right Hip', 'Left Knee', 'Right Knee', 
                  'Left Ankle', 'Right Ankle', 'Left Heel', 'Right Heel', 
                  'Left Foot Index', 'Right Foot Index']

        y_offset = 30

        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            visibility = landmark.visibility

            if visibility < 0.5:
                coord_text = f'{labels[idx]}: Not visible'
            else:
                coord_text = f'{labels[idx]}: ({cx}, {cy})'
            cv2.putText(side_panel, coord_text, (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            y_offset += 20

    combined_frame = np.hstack((frame, side_panel))

    cv2.imshow('Pose Detection with Coordinates', combined_frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()
