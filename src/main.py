import cv2
from src.components.ex_1 import ex_1
from src.components.ex_2 import ex_2
from src.utils import *
import time

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
    video_output = cv2.VideoWriter('output.mp4', fourcc, fps, frame_size)
    obj1 = ex_1()
    obj2 = ex_2()

    try:
        while True:
            obj1.PushUp(cap)
            time.sleep(1)
            obj2.Squard(cap)

            process_frame_1 = obj1.processed_frame
            process_frame_2 = obj2.processed_frame

            if process_frame_1 is not None:
                video_output.write(process_frame_1)
            if process_frame_2 is not None:
                video_output.write(process_frame_2)

            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):  # 27 is the ASCII code for the Esc key
                break

    except KeyboardInterrupt:
        pass
    finally:
        # Release the resources
        cap.release()
        video_output.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()