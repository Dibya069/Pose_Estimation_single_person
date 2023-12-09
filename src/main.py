import cv2
from src.components.ex_1 import ex_1
from src.components.ex_2 import ex_2
from src.components.merge import Merge_ex_vid
from src.utils import *
import time
import numpy as np

Merge = Merge_ex_vid()

def main():
    obj1 = ex_1()
    obj2 = ex_2()
    try:
        while True:
            obj1.PushUp(SaveVideo.cap)
            time.sleep(0.3)
            obj2.Squard(SaveVideo.cap)
            if obj2.stopLoop:
                break

    except KeyboardInterrupt:
        pass
    finally:
        # Release the resources
        SaveVideo.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    Merge.Merging_op()