import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
import random
import collections
import os
import glob
import itertools
from motpy import Detection, ModelPreset, MultiObjectTracker
import numpy as np


path = os.path.join("video_dir","*")
video_list = glob.glob(path)
video_playing = [True for i in range(len(video_list))]

print(video_list)
print(video_playing)
video_list_iter = itertools.cycle(iter(video_list))
current_id_video = []
videos = []
video_images=[]


for path_i in video_list:
    videos.append(cv2.VideoCapture(str(path_i)))

videos[1].set(cv2.CAP_PROP_POS_FRAMES, 50)
while True:
    video_images=[]
    for video_i in videos:
        video_i_ret, video_i_frame = video_i.read()
        if not video_i_ret:
            print(video_i_ret)
            video_i.set(cv2.CAP_PROP_POS_FRAMES, 0)
            video_i_frame = video_i.read()[1]
        video_images.append(video_i_frame)
        
    

    # 取得したフレームを表示
    cv2.imshow('frame', video_images[2])
    
    
    #print("2:",videos[2].get(cv2.CAP_PROP_POS_FRAMES))
    #print("1:",videos[1].get(cv2.CAP_PROP_POS_FRAMES))
    
    
    # qキーが押された場合はループを抜ける
    if cv2.waitKey(25) == ord('q'):
        break

