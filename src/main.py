import cv2
from src.components.ex_1 import ex_1
from src.components.ex_2 import ex_2
from src.utils import *
import time

def main():
    # Video writer.
    video_output = cv2.VideoWriter('output.mp4', SaveVideo.fourcc, SaveVideo.fps, SaveVideo.frame_size)
    obj1 = ex_1()
    obj2 = ex_2()

    try:
        while True:
            obj1.PushUp(SaveVideo.cap)
            time.sleep(0.3)
            obj2.Squard(SaveVideo.cap)

            process_frame_1 = obj1.processed_frame
            process_frame_2 = obj2.processed_frame

            video_output.write(process_frame_1)
            video_output.write(process_frame_2)

            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):  # 27 is the ASCII code for the Esc key
                break

    except KeyboardInterrupt:
        pass
    finally:
        # Release the resources
        SaveVideo.cap.release()
        video_output.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()