import cv2 as cv
import numpy as np
import streamlit as st

# Streamlit UI components
st.title("Pose Estimation with OpenPose (Live Feed)")
st.sidebar.header("Settings")

# Sidebar for input
threshold = st.sidebar.slider("Threshold for Pose Parts Heatmap", 0.0, 1.0, 0.2)
inWidth = st.sidebar.number_input("Input Width", value=368)
inHeight = st.sidebar.number_input("Input Height", value=368)

# Load pre-trained OpenPose model
net = cv.dnn.readNetFromTensorflow("pose_estimation_model.pb")

# Define body parts and pose pairs
BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

def process_frame(frame, width, height, threshold):
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    
    # Input frame preprocessing
    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (width, height), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]  # Keep only the relevant 19 parts

    assert(len(BODY_PARTS) == out.shape[1])

    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > threshold else None)

    return points

def draw_skeleton(frame, points):
    for pair in POSE_PAIRS:
        partFrom, partTo = pair
        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

# Start capturing live webcam feed
run_live_feed = st.checkbox("Start Live Feed", value=False)

if run_live_feed:
    cap = cv.VideoCapture(0)  # Use webcam

    stframe = st.empty()  # A placeholder to display the frame
    
    while run_live_feed:
        ret, frame = cap.read()
        if not ret:
            st.write("Error accessing webcam.")
            break

        # Process the frame for pose estimation
        points = process_frame(frame, inWidth, inHeight, threshold)
        draw_skeleton(frame, points)

        # Convert BGR to RGB for displaying in Streamlit
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        
        # Display frame
        stframe.image(rgb_frame, channels="RGB", use_column_width=True)
    
    cap.release()
else:
    st.write("Click on 'Start Live Feed' to capture the live camera stream.")

