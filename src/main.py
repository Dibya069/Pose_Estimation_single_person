import cv2
import time
import sys, os

from src.components.ex_1 import ex_1
from src.components.ex_2 import ex_2
from src.utils import *

def main():
    file_name = 'test2.mp4'
    cap = cv2.VideoCapture(file_name)

    # Meta.
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Video writer.
    #video_output = cv2.VideoWriter('output.mp4', fourcc, fps, frame_size)
    obj1 = ex_1()
    obj2 = ex_2()

    obj1.PushUp(cap)

    time.sleep(5)

    obj2.Squard(cap)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()