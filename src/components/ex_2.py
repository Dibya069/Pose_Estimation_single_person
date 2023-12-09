import cv2
import time
import math as m
import mediapipe as mp
import winsound
import time
import threading
import numpy as np

from dataclasses import dataclass
from src.utils import *

@dataclass
class ex_2:
    def __init__(self):
        self.tom = 0
        self.flag = None
        self.processed_frame = None
        self.video_saved = cv2.VideoWriter('output2.mp4', SaveVideo.fourcc, SaveVideo.fps, SaveVideo.frame_size)
        self.stopLoop = False

    def Squard(self, cap):
        try:
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
                keypoints = CONST.pose.process(image)

                # Convert the image back to BGR.
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Use lm and lmPose as representative of the following methods.
                lm = keypoints.pose_landmarks
                lmPose = CONST.mp_pose.PoseLandmark

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

                    cv2.putText(image, str("Reverse_Crunches"), (140, 30), CONST.font, 1, CONST.blue, 5)

                    # Calculate angles.
                    hip_angle = findAngle_bet_3_points(l_shldr_x, l_shldr_y, l_hip_x, l_hip_y, l_KNEE_x, l_KNEE_y)
                    knee_angle = findAngle_bet_3_points(l_hip_x, l_hip_y, l_KNEE_x, l_KNEE_y, l_ANKLE_x, l_ANKLE_y)

                    # Draw landmarks.
                    cv2.circle(image, (l_shldr_x, l_shldr_y), 7, CONST.yellow, -1)
                    cv2.circle(image, (l_hip_x, l_hip_y), 7, CONST.yellow, -10)
                    cv2.circle(image, (l_KNEE_x, l_KNEE_y), 7, CONST.yellow, -10)
                    cv2.circle(image, (l_ANKLE_x, l_ANKLE_y), 7, CONST.yellow, -10)

                    cv2.putText(image, str("Push Up: 4 (Done)"), (450, 50), CONST.font, 0.5, CONST.green, 2)

                    if knee_angle < 60:
                        if hip_angle > 70:   #add other end condition also like elbow > 10
                            self.flag = "Down"
                        if hip_angle < 50 and self.flag == "Down":
                            self.flag = "Up"
                            self.tom += 1
                            sendWarning()
                        if self.tom == 3:
                            cv2.putText(image, str("Reverse Crunches: 3 (Done)"), (380, 90), CONST.font, 0.5, CONST.green, 2)

                        cv2.putText(image, str(int(self.tom)), (20, 180), CONST.font, 0.9, CONST.green, 2)
                        cv2.putText(image, str(self.flag), (20, 130), CONST.font, 1, CONST.green, 2)
                        cv2.putText(image, str(int(hip_angle)) + " deg", (l_hip_x + 10, l_hip_y), CONST.font, 0.9, CONST.green, 2)
                        cv2.putText(image, str(int(knee_angle)) + " deg", (l_KNEE_x + 10, l_KNEE_y), CONST.font, 0.9, CONST.green, 2)

                        # Join landmarks.
                        cv2.line(image, (l_shldr_x, l_shldr_y), (l_hip_x, l_hip_y), CONST.green, 4)
                        cv2.line(image, (l_hip_x, l_hip_y), (l_KNEE_x, l_KNEE_y), CONST.green, 4)
                        cv2.line(image, (l_KNEE_x, l_KNEE_y), (l_ANKLE_x, l_ANKLE_y), CONST.green, 4)
                    
                    else:
                        if hip_angle > 70:   #add other end condition also like elbow > 10
                            self.flag = "Down"
                        if hip_angle < 50 and self.flag == "Down":
                            self.flag = "Up"
                            self.tom += 0
                        if self.tom == 3:
                            cv2.putText(image, str("Reverse Crunches: 3 (Done)"), (380, 90), CONST.font, 0.5, CONST.green, 2)


                        cv2.putText(image, str(int(self.tom)), (20, 180), CONST.font, 0.9, CONST.red, 2)
                        cv2.putText(image, str(self.flag), (20, 130), CONST.font, 1, CONST.red, 2)
                        cv2.putText(image, str(int(hip_angle)) + " deg", (l_hip_x + 10, l_hip_y), CONST.font, 0.9, CONST.red, 2)
                        cv2.putText(image, str(int(knee_angle)) + " deg", (l_KNEE_x + 10, l_KNEE_y), CONST.font, 0.9, CONST.red, 2)

                        # Join landmarks.
                        cv2.line(image, (l_shldr_x, l_shldr_y), (l_hip_x, l_hip_y), CONST.red, 4)
                        cv2.line(image, (l_hip_x, l_hip_y), (l_KNEE_x, l_KNEE_y), CONST.red, 4)
                        cv2.line(image, (l_KNEE_x, l_KNEE_y), (l_ANKLE_x, l_ANKLE_y), CONST.red, 4)
                    

                else:
                    print("body for predictioin is not shown yet...")
                # Write frames.
                #video_output.write(image)

                self.processed_frame = self.video_saved.write(image)
                image = cv2.resize(image, (w, h))

                # Display.
                cv2.imshow('MediaPipe Pose', image)
                if cv2.waitKey(5) & self.tom > 3 or 0xFF == ord('q'):
                    self.stopLoop = True
                    break

        except Exception as e:
            print(e)