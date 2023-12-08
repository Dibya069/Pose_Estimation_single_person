import cv2
from src.components.ex_1 import ex_1
from src.components.ex_2 import ex_2
from src.utils import *
import time
import numpy as np

def main():
    # Video writer.
    video_output = cv2.VideoWriter('output.mp4', SaveVideo.fourcc, SaveVideo.fps, SaveVideo.frame_size)
    obj1 = ex_1()
    obj2 = ex_2()

    try:
        while True:
            process_frame_1 = obj1.PushUp(SaveVideo.cap)
            time.sleep(0.3)
            process_frame_2 = obj2.Squard(SaveVideo.cap)


            combined_frame = cv2.hconcat([process_frame_1, process_frame_2])

            # Normalize the combined frame to uint8
            normalized_frame = cv2.normalize(combined_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            # Write the normalized frame to the video
            video_output.write(normalized_frame)

            key = cv2.waitKey(1)    
            if key == 27 or key == ord('c'):  # 27 is the ASCII code for the Esc key
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