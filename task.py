import matplotlib.pyplot as plt
import numpy as np

from yuv_read_one_frame import read_frame
from predict_frame import predict_frame

#  use these test sequences
input_video = 'sequences/BasketballPass_416x240_50_10frames.yuv';
# input_video = 'sequences/BlowingBubbles_416x240_50_10frames.yuv';
# input_video = 'sequences/BQSquare_416x240_60_10frames.yuv';
# input_video = 'sequences/RaceHorses_416x240_30_10frames.yuv';
num_frames = 10 # all test sequences consist of 10 frames

# all test sequences have this spatial resolution
width = 416
height = 240

# Example code to demonstrate how to read and visualize a frame
# we only use the luma component Y
frame_number = 0
y, u, v = read_frame(input_video, frame_number, width, height)

plt.figure()
plt.imshow(y, cmap='gray')
plt.title('Luminance component Y')
plt.show()


# For each task use the function predict_frame() to perform the prediction and extend it if new parameters are required.
# Finally, executing the different scripts/functions for each task should create the respective results (e.g.  plots),
# which you may then include into the report.

# Task 1
# TODO: extend these code fragments
blocksize = 4
search_range = 6  #  modify this as appropriate
cur_frame = None # TODO: replace
ref_frame = None
pred = predict_frame(cur_frame, ref_frame, blocksize, search_range)


## Task 2
#  ...
