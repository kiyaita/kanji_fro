import cv2
import numpy as np

cap = cv2.VideoCapture(0)

video1_path = 'output_video.mp4'
video1 = cv2.VideoCapture(video1_path)


dx = 50
dy = 50
resize_rate = 5
# 動画が再生中かどうかを示すフラグ
playing = True

def video_show(cap_frame, video1, dx, dy, playing):
    video1_ret, video1_frame = video1.read()
    if not video1_ret:
            playing = False
        
    if playing:
        video1_frame = cv2.resize(video1_frame, (int(video1_frame.shape[1] / resize_rate) ,int(video1_frame.shape[0] / resize_rate )))
        M = np.array([[1, 0, dx], [0, 1, dy]], dtype=float)
        
        cap_frame = cv2.warpAffine(video1_frame, M, (cap_frame.shape[1], cap_frame.shape[0]), cap_frame, borderMode=cv2.BORDER_TRANSPARENT)
    
    return cap_frame
    




while True:
    ret, cap_frame = cap.read()
    
    if ret:
        
        cap_frame = video_show(cap_frame, video1, dx , dy, playing)

        cv2.imshow('cap', cap_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) & 0xFF == 27:
            break
        
        
        
        # Wキーが押されたら、再生を再開
        if cv2.waitKey(1) == ord('w'):
            video1 = cv2.VideoCapture(video1_path)
            video1.set(cv2.CAP_PROP_POS_FRAMES, 0)
            playing = True

        # Eキーが押されたら、再生を停止
        if cv2.waitKey(1) == ord('e'):
            playing = False
        
        
    

cap.release()
video1.release()
cv2.destroyAllWindows()
