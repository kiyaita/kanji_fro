import os 
import sys
import glob
import itertools

path = os.path.join("video_dir","*")
video_list = glob.glob(path)
video_playing = [True for i in range(len(video_list))]

print(video_list)
print(video_playing)
video_list_iter = itertools.cycle(iter(video_list))

for i in range(100):
    print(next(video_list_iter))