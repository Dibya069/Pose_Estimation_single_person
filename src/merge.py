import cv2
from src.utils import *
from tqdm import tqdm as tqdm_progress

# A list of the paths of your videos
videos = ["E:/data science/PoseEstimation/output1.mp4", "E:/data science/PoseEstimation/output2.mp4"]

# Create a new video
video = cv2.VideoWriter("new_video.mp4", cv2.VideoWriter_fourcc(*"MPEG"), SaveVideo.fps, SaveVideo.frame_size)

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