import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=5, min_detection_confidence=0.7)

def get_finger_status(hand_landmarks):
    finger_tips = [8, 12, 16, 20]
    finger_mcp = [5, 9, 13, 17]
    status = []
    
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        status.append(1)
    else:
        status.append(0)
    
    for tip, mcp in zip(finger_tips, finger_mcp):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[mcp].y:
            status.append(1)
        else:
            status.append(0)
    return status

def recognize_gesture(finger_status):
    gestures = {
        tuple([0,1,0,0,0]): "Peace ✌️",
        tuple([1,1,1,1,1]): "Open Hand 🖐️",
        tuple([0,0,0,0,0]): "Thumbs up ✊",
        tuple([1,0,0,0,0]): "Fist 👍"
    }
    return gestures.get(tuple(finger_status), "Unknown")

def process_frame(image):
    img_rgb_np = np.frombuffer(image, dtype=np.uint8)
    results = hands.process(img_rgb_np)
    return results
