#https://github.com/wmuron/motpy
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
import random
import collections
from motpy import Detection, ModelPreset, MultiObjectTracker

Track = collections.namedtuple('Track', 'id box score class_id')


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
        
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                
                # Extract bounding box coordinates
                x_min = int(min(l.x for l in hand_landmarks.landmark) * image.shape[1])
                y_min = int(min(l.y for l in hand_landmarks.landmark) * image.shape[0])
                x_max = int(max(l.x for l in hand_landmarks.landmark) * image.shape[1])
                y_max = int(max(l.y for l in hand_landmarks.landmark) * image.shape[0])
                bbox_center_x = int((x_min + x_max) / 2)
                bbox_center_y = int((y_min + y_max) / 2)
                
                out_detections = []
                out_detections.append(Detection(box=[x_min, y_min, x_max, y_max] ))
                tracker.step(detections=out_detections)
                
                tracks = tracker.active_tracks(min_steps_alive=6)
                for track in tracks:
                    draw_track(image, track)
                    
                    
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 220, 0), 2)
                cv2.circle(image, (bbox_center_x, bbox_center_y), 5, (0, 220, 0), -1)
                
                
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
        else:
            tracker.step(detections=[])
            tracks = tracker.active_tracks(min_steps_alive=1)
                

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        
        if cv2.waitKey(5) & 0xFF == 27:
            break


cap.release()



