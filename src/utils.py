import cv2
import time
import math as m
import mediapipe as mp
import winsound
import time
import threading
import numpy as np
from dataclasses import dataclass

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
@dataclass
class CONST:
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

@dataclass
class SaveVideo:
    file = "E:/data science/PoseEstimation/test2.mp4"
    cap = cv2.VideoCapture(1)
    
    # Meta.
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

@dataclass
class Results:
    ex_out1 = "E:/data science/PoseEstimation/output1.mp4"
    ex_out2 = "E:/data science/PoseEstimation/output2.mp4"