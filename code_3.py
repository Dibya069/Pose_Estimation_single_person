import cv2
import time
import math as m
import mediapipe as mp
import winsound
import time
import threading

# Calculate distance
def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist


# Calculate angle.
def findAngle(x1, y1, x2, y2):
    theta = m.acos((y2 - y1) * (-y1) / (m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
    degree = int(180 / m.pi) * theta
    return degree


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
    #video_output = cv2.VideoWriter('output.mp4', fourcc, fps, frame_size)
    tom = 0
    flag = False

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

            # Calculate distance between left shoulder and right shoulder points.
            offset = findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)

            # Assist to align the camera to point at the side view of the person.
            # Offset threshold 30 is based on results obtained from analysis over 100 samples.
            if offset < 400:
                cv2.putText(image, str(int(offset)) + ' Aligned', (w - 200, 30), font, 0.9, green, 2)
            else:
                cv2.putText(image, str(int(offset)) + ' Not Aligned', (w - 240, 30), font, 0.9, red, 2)

            # Calculate angles.
            neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
            elbow_angle = findAngle(l_elb_x, l_elb_y, l_shldr_x, l_shldr_y)
            wrist_angle = findAngle(l_wri_x, l_wri_y, l_elb_x, l_elb_y)

            # Draw landmarks.
            cv2.circle(image, (l_shldr_x, l_shldr_y), 7, yellow, -1)
            cv2.circle(image, (l_ear_x, l_ear_y), 7, yellow, -1)

            # Let's take y - coordinate of P3 100px above x1,  for display elegance.
            # Although we are taking y = 0 while calculating angle between P1,P2,P3.
            cv2.circle(image, (l_shldr_x, l_shldr_y - 100), 7, yellow, -1)
            cv2.circle(image, (r_shldr_x, r_shldr_y), 7, pink, -1)
            cv2.circle(image, (l_elb_x, l_elb_y), 7, yellow, -10)
            cv2.circle(image, (l_wri_x, l_wri_y), 7, yellow, -10)


            # Similarly, here we are taking y - coordinate 100px above x1. Note that
            # you can take any value for y, not necessarily 100 or 200 pixels.

            # Put text, Posture and angle inclination.
            # Text string for display.
            angle_text_string = 'Neck : ' + str(int(neck_inclination))

            # Determine whether good posture or bad posture.
            # The threshold angles have been set based on intuition.               

            cv2.putText(image, angle_text_string, (10, 30), font, 0.9, red, 2)
            cv2.putText(image, str(int(neck_inclination)) + " deg", (l_shldr_x + 10, l_shldr_y), font, 0.9, red, 2)
            cv2.putText(image, str(int(elbow_angle)) + " deg", (l_elb_x + 10, l_elb_y), font, 0.9, red, 2)
            cv2.putText(image, str(int(wrist_angle)) + " deg", (l_wri_x + 10, l_wri_y), font, 0.9, red, 2)

            if elbow_angle < 19 and not flag:
                tom += 1
                sendWarning()
                flag = True
                delay_thread = threading.Thread(target=delay_function)
                delay_thread.start()

            cv2.putText(image, str(int(tom)), (20, 300), font, 0.9, red, 2)

            # Join landmarks.
            cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), red, 4)
            cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 50), red, 4)
            
            cv2.line(image, (l_shldr_x, l_shldr_y), (l_elb_x, l_elb_y), green, 4)
            cv2.line(image, (l_elb_x, l_elb_y), (l_elb_x, l_elb_y - 50), green, 4)

            cv2.line(image, (l_elb_x, l_elb_y), (l_wri_x, l_wri_y), green, 4)
            cv2.line(image, (l_wri_x, l_wri_y), (l_wri_x, l_wri_y - 50), green, 4)

            # If you stay in bad posture for more than 3 minutes (180s) send an alert.
            #if bad_time > 5:
                #sendWarning()

        else:
            print("body for predictioin is not shown yet...")
        # Write frames.
        #video_output.write(image)

        #image = cv2.resize(image, (w // 2, h // 2))

        # Display.
        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()