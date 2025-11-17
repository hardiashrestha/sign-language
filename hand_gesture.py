import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe hand model
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

def get_finger_status(hand_landmarks):
    finger_tips = [8, 12, 16, 20]  # Index, middle, ring, pinky fingertips
    finger_mcp = [5, 9, 13, 17]    # Corresponding MCP joint landmarks for fingers
    status = []
    
    # Thumb: Compare tip and IP joint x-position for open/closed
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        status.append(1)
    else:
        status.append(0)
    
    # Fingers: tip y less than mcp y means finger is open
    for tip, mcp in zip(finger_tips, finger_mcp):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[mcp].y:
            status.append(1)
        else:
            status.append(0)
    return status

def recognize_gesture(finger_status):
    # Simple rules for 3 gestures
    if finger_status == [0,1,0,0,0]:
        return "Peace ✌️"
    elif finger_status == [1,1,1,1,1]:
        return "Open Hand 🖐️"
    elif finger_status == [0,0,0,0,0]:
        return "Thumbs up✊"
    elif finger_status == [1,0,0,0,0]:
        return "Fist👍"
    else:
        return "Unknown"

def main():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty frame.")
            continue

        image = cv2.flip(image, 1)  # Mirror image
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                finger_status = get_finger_status(hand_landmarks)
                gesture = recognize_gesture(finger_status)
                cv2.putText(image, gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)
                
        cv2.imshow("Hand Gesture Recognition", image)
        if cv2.waitKey(5) & 0xFF == 27:  # ESC key to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
