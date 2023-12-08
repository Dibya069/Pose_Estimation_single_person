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
class ex_1:
    def __init__(self):
        self.tom = 0
        self.flag = None
        self.processed_frame = None
        self.video_saved = cv2.VideoWriter('output1.mp4', SaveVideo.fourcc, SaveVideo.fps, SaveVideo.frame_size)

    def PushUp(self, cap):
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
                # Right shoulder
                r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
                r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
                # Left ear.
                l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
                l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)
                #left elbow
                l_elb_x = int(lm.landmark[lmPose.LEFT_ELBOW].x * w)
                l_elb_y = int(lm.landmark[lmPose.LEFT_ELBOW].y * h)
                #left hand
                l_wri_x = int(lm.landmark[lmPose.LEFT_WRIST].x * w)
                l_wri_y = int(lm.landmark[lmPose.LEFT_WRIST].y * h)
                #left hip
                l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
                l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)
                #left knee
                l_KNEE_x = int(lm.landmark[lmPose.LEFT_KNEE].x * w)
                l_KNEE_y = int(lm.landmark[lmPose.LEFT_KNEE].y * h)

                # Calculate distance between left shoulder and right shoulder points.
                offset = findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)

                # Assist to align the camera to point at the side view of the person.
                # Offset threshold 30 is based on results obtained from analysis over 100 samples.
                if offset < 400:
                    cv2.putText(image, str(int(offset)) + ' Aligned', (w - 200, 30), CONST.font, 0.9, CONST.green, 2)
                else:
                    cv2.putText(image, str(int(offset)) + ' Not Aligned', (w - 240, 30), CONST.font, 0.9, CONST.red, 2)

                cv2.putText(image, str("Push_Up"), (600, 50), CONST.font, 1.5, CONST.blue, 5)

                # Calculate angles.
                neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
                elbow_angle = findAngle(l_elb_x, l_elb_y, l_shldr_x, l_shldr_y)
                wrist_angle = findAngle(l_wri_x, l_wri_y, l_elb_x, l_elb_y)
                hip_angle = findAngle_bet_3_points(l_elb_x, l_elb_y, l_hip_x, l_hip_y, l_KNEE_x, l_KNEE_y)

                # Draw landmarks.
                cv2.circle(image, (l_shldr_x, l_shldr_y), 7, CONST.yellow, -1)
                cv2.circle(image, (l_ear_x, l_ear_y), 7, CONST.yellow, -1)

                cv2.circle(image, (l_shldr_x, l_shldr_y - 100), 7, CONST.yellow, -1)
                cv2.circle(image, (r_shldr_x, r_shldr_y), 7, CONST.pink, -1)
                cv2.circle(image, (l_elb_x, l_elb_y), 7, CONST.yellow, -10)
                cv2.circle(image, (l_wri_x, l_wri_y), 7, CONST.yellow, -10)
                cv2.circle(image, (l_hip_x, l_hip_y), 7, CONST.yellow, -10)
                cv2.circle(image, (l_KNEE_x, l_KNEE_y), 7, CONST.yellow, -10)

                angle_text_string = 'Neck : ' + str(int(neck_inclination))              

                if hip_angle > 100:
                    cv2.putText(image, angle_text_string, (10, 30), CONST.font, 0.9, CONST.green, 2)
                    cv2.putText(image, str(int(neck_inclination)) + " deg", (l_shldr_x + 10, l_shldr_y), CONST.font, 0.9, CONST.green, 2)
                    cv2.putText(image, str(int(elbow_angle)) + " deg", (l_elb_x + 10, l_elb_y), CONST.font, 0.9, CONST.green, 2)
                    cv2.putText(image, str(int(wrist_angle)) + " deg", (l_wri_x + 10, l_wri_y), CONST.font, 0.9, CONST.green, 2)
                    cv2.putText(image, str(int(hip_angle)) + "deg", (l_hip_x + 10, l_hip_y), CONST.font, 0.9, CONST.green, 2)

                    if elbow_angle > 115:   #add other end condition also like elbow > 10
                        self.flag = "Down"
                    if elbow_angle < 19 and self.flag == "Down":
                        self.flag = "Up"
                        self.tom += 1
                        sendWarning()

                    cv2.putText(image, str(int(self.tom)), (20, 300), CONST.font, 1, CONST.green, 2)
                    cv2.putText(image, str(self.flag), (20, 250), CONST.font, 1, CONST.green, 2)

                    # Join landmarks.
                    cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), CONST.green, 4)
                    cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 50), CONST.green, 4)
                    
                    cv2.line(image, (l_shldr_x, l_shldr_y), (l_elb_x, l_elb_y), CONST.green, 4)
                    cv2.line(image, (l_elb_x, l_elb_y), (l_elb_x, l_elb_y - 50), CONST.green, 4)

                    cv2.line(image, (l_elb_x, l_elb_y), (l_wri_x, l_wri_y), CONST.green, 4)
                    cv2.line(image, (l_wri_x, l_wri_y), (l_wri_x, l_wri_y - 50), CONST.green, 4)

                    cv2.line(image, (l_shldr_x, l_shldr_y), (l_hip_x, l_hip_y), CONST.green, 4)
                    cv2.line(image, (l_hip_x, l_hip_y), (l_KNEE_x, l_KNEE_y), CONST.green, 4)
                
                else:
                    cv2.putText(image, angle_text_string, (10, 30), CONST.font, 0.9, CONST.red, 2)
                    cv2.putText(image, str(int(neck_inclination)) + " deg", (l_shldr_x + 10, l_shldr_y), CONST.font, 0.9, CONST.red, 2)
                    cv2.putText(image, str(int(elbow_angle)) + " deg", (l_elb_x + 10, l_elb_y), CONST.font, 0.9, CONST.red, 2)
                    cv2.putText(image, str(int(wrist_angle)) + " deg", (l_wri_x + 10, l_wri_y), CONST.font, 0.9, CONST.red, 2)
                    cv2.putText(image, str(int(hip_angle)) + "deg", (l_hip_x + 10, l_hip_y), CONST.font, 0.9, CONST.red, 2)

                    if elbow_angle > 100:   #add other end condition also like elbow > 10
                        self.flag = "Down"
                    if elbow_angle < 19 and self.flag == "Down":
                        self.flag = "Up"
                        self.tom += 0
                        sendWarning()

                    cv2.putText(image, str(int(self.tom)), (20, 300), CONST.font, 1, CONST.red, 2)
                    cv2.putText(image, str(self.flag), (20, 250), CONST.font, 1, CONST.red, 2)
                    cv2.putText(image, str("Bad posture"), (20, 350), CONST.font, 1, CONST.red, 2)

                    # Join landmarks.
                    cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), CONST.red, 4)
                    cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 50), CONST.red, 4)
                    
                    cv2.line(image, (l_shldr_x, l_shldr_y), (l_elb_x, l_elb_y), CONST.red, 4)
                    cv2.line(image, (l_elb_x, l_elb_y), (l_elb_x, l_elb_y - 50), CONST.red, 4)

                    cv2.line(image, (l_elb_x, l_elb_y), (l_wri_x, l_wri_y), CONST.red, 4)
                    cv2.line(image, (l_wri_x, l_wri_y), (l_wri_x, l_wri_y - 50), CONST.red, 4)

                    cv2.line(image, (l_shldr_x, l_shldr_y), (l_hip_x, l_hip_y), CONST.red, 4)
                    cv2.line(image, (l_hip_x, l_hip_y), (l_KNEE_x, l_KNEE_y), CONST.red, 4)
                

            else:
                print("body for predictioin is not shown yet...")


            self.processed_frame = self.video_saved.write(image)
            image = cv2.resize(image, (w // 2, h // 2))

            # Display.
            cv2.imshow('MediaPipe Pose', image)
            if cv2.waitKey(5) & self.tom > 1 or 0xFF == ord("q"):
                break

        return self.processed_frame