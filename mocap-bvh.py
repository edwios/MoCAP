import cv2
import numpy as np
import mediapipe as mp
from collections import namedtuple

# Load the 3D posenet model
mp_holistic = mp.solutions.holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Define a named tuple for storing the joint positions
Joint = namedtuple('Joint', ['x', 'y', 'z'])

# Initialize variables
frame_width = 640
frame_height = 480
capture = cv2.VideoCapture(1)
joints_3d = []

while True:
    # Capture frame-by-frame
    ret, frame = capture.read()

    # Preprocess the frame for pose estimation
    frame.flags.writeable = False
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_holistic.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Extract the joint positions from the pose estimation results
    if results.pose_world_landmarks:
        landmarks = np.array([[lmk.x, lmk.y, lmk.z] for lmk in results.pose_world_landmarks.landmark]).reshape(-1, 3)
        if len(landmarks) >= 4:
            joints_3d = []
            for i in range(len(landmarks)):
                x, y, z = landmarks[i]
                joints_3d.append(Joint(x, y, z))

    # Draw the skeleton on the frame
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        # mp_drawing.plot_landmarks(results.pose_world_landmarks, mp.solutions.holistic.POSE_CONNECTIONS)
        cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
    
        # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
capture.release()
cv2.destroyAllWindows()
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
    f.close()
