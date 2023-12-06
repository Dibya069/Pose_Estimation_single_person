import cv2
import time
import math as m
import mediapipe as mp
import winsound
import time
import threading
import numpy as np

# Calculate distance
def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist


# Calculate angle.
def findAngle(x1, y1, x2, y2):
    theta = m.acos((y2 - y1) * (-y1) / (m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
    degree = int(180 / m.pi) * theta
    return degree

#CALCULATE ANGLE
def findAngle_bet_3_points(a0, a1, b0, b1, c0, c1):
    rediance = np.arctan2(c1 - b1, c0 - b0) - np.arctan2(a1 - b1, a0 - b0)
    angle = np.abs(rediance * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle

# Function to send alert
def sendWarning():
    winsound.Beep(800, 200)

# For push up count
def delay_function():
    global flag
    time.sleep(2)
    flag = False


# =============================CONSTANTS and INITIALIZATIONS=====================================#
# Initilize frame counters.
good_frames = 0
bad_frames = 0

# Font type.
font = cv2.FONT_HERSHEY_SIMPLEX

# Colors.
blue = (255, 127, 0)
red = (50, 50, 255)
green = (127, 255, 0)
dark_blue = (127, 20, 0)
light_green = (127, 233, 100)
yellow = (0, 255, 255)
pink = (255, 0, 255)

# Initialize mediapipe pose class.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
# ===============================================================================================#

if __name__ == "__main__":
    file_name = 'test2.mp4'
    cap = cv2.VideoCapture(1)

    # Meta.
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Video writer.
    video_output = cv2.VideoWriter('output.mp4', fourcc, fps, frame_size)
    tom = 0
    flag = None

    while cap.isOpened():
        # Capture frames.
        success, image = cap.read()
        if not success:
            print("Null.Frames")
            break
        # Get fps.
        fps = cap.get(cv2.CAP_PROP_FPS)
        # Get height and width.
        h, w = image.shape[:2]

        # Convert the BGR image to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image.
        keypoints = pose.process(image)

        # Convert the image back to BGR.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Use lm and lmPose as representative of the following methods.
        lm = keypoints.pose_landmarks
        lmPose = mp_pose.PoseLandmark

        if lm is not None:
            # Acquire the landmark coordinates.
            # Once aligned properly, left or right should not be a concern.      
            # Left shoulder.
            l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
            l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
            #left hip
            l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
            l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)
            #left knee
            l_KNEE_x = int(lm.landmark[lmPose.LEFT_KNEE].x * w)
            l_KNEE_y = int(lm.landmark[lmPose.LEFT_KNEE].y * h)
            #left knee
            l_ANKLE_x = int(lm.landmark[lmPose.LEFT_ANKLE].x * w)
            l_ANKLE_y = int(lm.landmark[lmPose.LEFT_ANKLE].y * h)


            # Calculate angles.
            hip_angle = findAngle_bet_3_points(l_shldr_x, l_shldr_y, l_hip_x, l_hip_y, l_KNEE_x, l_KNEE_y)
            knee_angle = findAngle_bet_3_points(l_hip_x, l_hip_y, l_KNEE_x, l_KNEE_y, l_ANKLE_x, l_ANKLE_y)

            # Draw landmarks.
            cv2.circle(image, (l_shldr_x, l_shldr_y), 7, yellow, -1)
            cv2.circle(image, (l_hip_x, l_hip_y), 7, yellow, -10)
            cv2.circle(image, (l_KNEE_x, l_KNEE_y), 7, yellow, -10)
            cv2.circle(image, (l_ANKLE_x, l_ANKLE_y), 7, yellow, -10)

            if knee_angle < 60:
                if hip_angle > 70:   #add other end condition also like elbow > 10
                    flag = "Down"
                if hip_angle < 50 and flag == "Down":
                    flag = "Up"
                    tom += 1
                    sendWarning()

                cv2.putText(image, str(int(tom)), (20, 300), font, 0.9, green, 2)
                cv2.putText(image, str(flag), (20, 350), font, 0.9, green, 2)

                
                cv2.putText(image, str(int(hip_angle)) + " deg", (l_hip_x + 10, l_hip_y), font, 0.9, green, 2)
                cv2.putText(image, str(int(knee_angle)) + " deg", (l_KNEE_x + 10, l_KNEE_y), font, 0.9, green, 2)

                # Join landmarks.
                cv2.line(image, (l_shldr_x, l_shldr_y), (l_hip_x, l_hip_y), green, 4)
                cv2.line(image, (l_hip_x, l_hip_y), (l_KNEE_x, l_KNEE_y), green, 4)
                cv2.line(image, (l_KNEE_x, l_KNEE_y), (l_ANKLE_x, l_ANKLE_y), green, 4)
            
            else:
                if hip_angle > 70:   #add other end condition also like elbow > 10
                    flag = "Down"
                if hip_angle < 50 and flag == "Down":
                    flag = "Up"
                    tom += 0

                cv2.putText(image, str(int(tom)), (20, 300), font, 0.9, red, 2)
                cv2.putText(image, str(flag), (20, 350), font, 0.9, red, 2)
                
                cv2.putText(image, str(int(hip_angle)) + " deg", (l_hip_x + 10, l_hip_y), font, 0.9, red, 2)
                cv2.putText(image, str(int(knee_angle)) + " deg", (l_KNEE_x + 10, l_KNEE_y), font, 0.9, red, 2)

                # Join landmarks.
                cv2.line(image, (l_shldr_x, l_shldr_y), (l_hip_x, l_hip_y), red, 4)
                cv2.line(image, (l_hip_x, l_hip_y), (l_KNEE_x, l_KNEE_y), red, 4)
                cv2.line(image, (l_KNEE_x, l_KNEE_y), (l_ANKLE_x, l_ANKLE_y), red, 4)
            

        else:
            print("body for predictioin is not shown yet...")
        # Write frames.
        video_output.write(image)

        image = cv2.resize(image, (w, h))

        # Display.
        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    video_output.release()
    cv2.destroyAllWindows()