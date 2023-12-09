import cv2
from src.utils import *
from tqdm import tqdm as tqdm_progress
from dataclasses import dataclass

@dataclass
class Merge_ex_vid:
    def __init__(self):
        self.output1 = Results.ex_out1
        self.output2 = Results.ex_out2
    def Merging_op(self):
        # A list of the paths of your videos
        videos = [self.output1, self.output2]

        # Create a new video
        video = cv2.VideoWriter("Output.mp4", SaveVideo.fourcc, SaveVideo.fps, SaveVideo.frame_size)

        # Get the total number of frames in all videos
        total_frames = sum(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT) for video_path in videos)


        # Write all the frames sequentially to the new video
        with tqdm_progress(total=total_frames, desc="Merging Videos", unit="frames") as pbar:
            for v in videos:
                curr_v = cv2.VideoCapture(v)
                while curr_v.isOpened():
                    # Get return value and curr frame of curr video
                    r, frame = curr_v.read()
                    if not r:
                        break
                    # Write the frame
                    video.write(frame)
                    pbar.update(1)  # Update progress bar

        # Save the video
        video.release()
