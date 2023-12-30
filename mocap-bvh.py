import cv2
import numpy as np
import mediapipe as mp
from collections import namedtuple

# Load the 3D posenet model
mp_pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Define a named tuple for storing the joint positions
Joint = namedtuple('Joint', ['x', 'y', 'z'])

# Initialize variables
frame_width = 640
frame_height = 480
capture = cv2.VideoCapture(0)
joints_3d = []

while True:
    # Capture frame-by-frame
    ret, frame = capture.read()

    # Preprocess the frame for pose estimation
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = mp_pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Extract the joint positions from the pose estimation results
    if results.pose_landmarks:
        landmarks = np.array([[lmk.x, lmk.y] for lmk in results.pose_landmarks.landmark]).reshape(-1, 2)
        _, rvec, tvec = cv2.solvePnP(np.float32([[0, 0, 0], [0, 1, 0], [1, 0, 0]]), landmarks, np.eye(3), np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
        joints_3d = []
        for i in range(len(landmarks)):
            x, y = rvec @ landmarks[i] + tvec
            z = 0 # Assuming the camera is looking straight down at the subject
            joints_3d.append(Joint(x, y, z))

    # Draw the skeleton on the frame
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp.solutions.Pose.pose_landmarks)

    # Display the resulting frame
    cv2.imshow('Video', image)

    # Convert the 3D joint positions to BVH format and save them as a file
    if len(joints_3d) > 0:
        with open('output.bvh', 'w') as f:
            # Write the header information
            f.write("""
HIP
{0} 0.0 -1.0 0.0
0.0 1.0 0.0
0 0 0 1
""".format(len(joints_3d)))

            # Write the joint information
            for i in range(len(joints_3d)):
                x, y, z = joints_3d[i]
                f.write("""
{0} {1} {2} 0.0 -1.0 0.0
0.0 1.0 0.0
0 0 0 1
""".format(i, x, y, z))

            # Write the end of file marker
            f.write("""
MOT {0} 1
{0} 0.0 -1.0 0.0
0.0 1.0 0.0
0 0 0 1
""".format(len(joints_3d)))

    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
capture.release()
cv2.destroyAllWindows()
