#https://github.com/wmuron/motpy
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
import copy

Track = collections.namedtuple('Track', 'id box score class_id')

def video_show(cap_frame, video_frame, bbox, hand_17, hand_5 , leftORright):
    
    video1_frame = cv2.flip(video_frame, 0)
    
    if insideORoutside(hand_17, hand_5 ,leftORright):#手の甲ならうらかえす
        video1_frame = cv2.flip(video1_frame, -1)
    
    if leftORright == 'Left':#左右の手合わせ、
        video1_frame = cv2.flip(video1_frame, -1)
    
    center = bbox_2_center(bbox)
    angle = bbox_2_angle(hand_17, hand_5)
    scale = bbox_2_scale(hand_17, hand_5, video1_frame)
    
    video1_frame = cv2.resize(video1_frame, None, None, scale ,scale)
    cap_h, cap_w, _= cap_frame.shape
    video_h_new, video_w_new,_= video1_frame.shape
    center_set = (center[0] - (video_w_new/2), center[1] - (video_h_new/2)) #videoの縦横の分中心補正
    
    M = cv2.getRotationMatrix2D(center = (video_w_new/2, video_h_new/2), angle=angle , scale=1)
    
    M = M + np.array([[0, 0, center[0] - video_w_new/2],
                     [0, 0, center[1] - video_h_new/2]])
    
    cap_frame = cv2.warpAffine(video1_frame, M, (cap_frame.shape[1], cap_frame.shape[0]), cap_frame, borderMode=cv2.BORDER_TRANSPARENT)
    return cap_frame

def insideORoutside(hand_17, hand_5 ,leftORright):
    if hand_17.x > hand_5.x and leftORright=='Left':
        return True
    if hand_17.x < hand_5.x and leftORright=='Right':
        return True
    return False
    
    
def bbox_2_center(bbox):
    # bboxから中心座標を計算
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    center = (center_x, center_y)
    
    return center

def bbox_2_angle(hand_17, hand_5):
    # bboxの角度を計算
    angle = np.degrees(np.arctan2((hand_17.y-hand_5.y), - (hand_17.x-hand_5.x)))
    
    return angle

def bbox_2_scale(hand_17, hand_5, video_frame):
    # bboxのスケールを計算
    scale = np.sqrt(((hand_17.x - hand_5.x)*video_frame.shape[1])**2 + ((hand_17.y - hand_5.y)*video_frame.shape[0])**2) / video_frame.shape[1]
    return scale


def draw_track(img, track: Track, random_color: bool = True, fallback_color=(200, 20, 20), thickness: int = 5, text_at_bottom: bool = False, text_verbose: int = 1):
    color = [ord(c) * ord(c) % 256 for c in track.id[:3]] if random_color else fallback_color
    draw_rectangle(img, track.box, color=color, thickness=thickness)
    pos = (track.box[0], track.box[3]) if text_at_bottom else (track.box[0], track.box[1])

    if text_verbose > 0:
        text = track_to_string(track) if text_verbose == 2 else track.id[:8]
        draw_text(img, text, pos=pos)

    return img
def draw_rectangle(img, box, color, thickness: int = 3) -> None:
    cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, thickness)


def draw_detection(img, detection: Detection) -> None:
    draw_rectangle(img, detection.box, color=(0, 220, 0), thickness=1)

def track_to_string(track: Track) -> str:
    score = track.score if track.score is not None else -1
    return f'ID: {track.id[:8]} | S: {score:.1f} | C: {track.class_id}'


def draw_text(img, text, pos, color=(255, 255, 255)) -> None:
    tl_pt = (int(pos[0]), int(pos[1]) - 7)
    cv2.putText(img, text, tl_pt,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5, color=color)

# Create an instance of the MultiObjectTracker class
tracker = MultiObjectTracker(dt=1 / 30,
                             model_spec=ModelPreset.constant_acceleration_and_static_box_size_2d.value,
                            active_tracks_kwargs={'min_steps_alive': 2, 'max_staleness': 4},
                            tracker_kwargs={'max_staleness': 6}
                             )

################################################################################

path = os.path.join("video_dir","*")
video_list = glob.glob(path)
videos = []
current_id_video = {}
pre_id_video = {}

for path_i in video_list:
    videos.append(cv2.VideoCapture(str(path_i)))


print(video_list)


#for webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    prev_bbox = None # initialize previous bbox to None
    moves_detected = 0  # initialize moves_detected to 0
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
        
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        video_frames=[]
        for video_i in videos:#画像フレーム読み込み
            video_i_ret, video_i_frame = video_i.read()
            if not video_i_ret:#動画を再生しきったら最初に戻す
                video_i.set(cv2.CAP_PROP_POS_FRAMES, 1)
                video_i_frame = video_i.read()[1]
            video_frames.append(video_i_frame)#フレームを動画軍から取得
            
        image_first = image#謎の画像全部同じになる対策
        
        if results.multi_hand_landmarks:
            #idと動画番号対応部
            
            current_id_video = {}
            
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand_landmarks_select = (hand_landmarks.landmark[0],hand_landmarks.landmark[1],hand_landmarks.landmark[5],hand_landmarks.landmark[17])
                # Extract bounding box coordinates
                x_min = int(min(l.x for l in hand_landmarks_select) * image.shape[1])
                y_min = int(min(l.y for l in hand_landmarks_select) * image.shape[0])
                x_max = int(max(l.x for l in hand_landmarks_select) * image.shape[1])
                y_max = int(max(l.y for l in hand_landmarks_select) * image.shape[0])
                
                out_detections = []
                out_detections.append(Detection(box=[x_min, y_min, x_max, y_max] ))
                tracker.step(detections=out_detections)
                
                tracks = tracker.active_tracks(min_steps_alive=6)
                
                
                for track in tracks:
                    if not track.id in current_id_video: #辞書にidが乗ってなかったらランダムな動画番号と共に追加・・動画をトラックさせる
                        current_id_video[track.id] = random.randrange(len(video_frames))
                
                for key, value in pre_id_video.items():
                            if key in current_id_video: #後の辞書のキーが前の辞書に存在する場合、後の辞書の値を優先
                                current_id_video[key] = value
                            else: #後の辞書のキーが前の辞書に存在しない場合、新しいキーとして追加しない
                                pass
                pre_id_video = current_id_video
                
                
                for track in tracks:
                    #print(track.id, track.box)
                    #print(results.multi_handedness[idx].classification[0].label)
                    draw_track(image, track)
                    print(video_list[current_id_video[track.id]])
                    imagecopy = copy.copy(image)
                    image = video_show(imagecopy, video_frames[current_id_video[track.id]], [x_min, y_min, x_max, y_max], hand_landmarks.landmark[17], hand_landmarks.landmark[5], results.multi_handedness[idx].classification[0].label)
                    if track == tracks[0]:#謎の画像全部同じになる対策
                        image_first = copy.copy(image)
                
                
                
                
                """mp_drawing.draw_landmarks(
                    image_first,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())"""
                
        else:
            tracker.step(detections=[])
            tracks = tracker.active_tracks(min_steps_alive=1)
                
        
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Hands', cv2.flip(image_first, 1))
        
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()



