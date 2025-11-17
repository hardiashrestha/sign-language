import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.7)

# Load OpenCV face detector (fallback)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def get_finger_status(hand_landmarks):
    finger_tips = [8, 12, 16, 20]  # Index, middle, ring, pinky fingertips
    finger_mcp = [5, 9, 13, 17]    # Corresponding MCP joints
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

def analyze_emotion(face_img):
    try:
        result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
        return result['dominant_emotion']
    except Exception:
        return "Unknown"

def main():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty frame.")
            continue

        image = cv2.flip(image, 1)  # Mirror
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Hands
        hand_results = hands.process(img_rgb)

        # Face Mesh
        face_results = face_mesh.process(img_rgb)

        # Hand gesture
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                finger_status = get_finger_status(hand_landmarks)
                gesture = recognize_gesture(finger_status)
                cv2.putText(image, gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)

        h, w, _ = image.shape
        face_img = None

        # Face Mesh detection
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            x_coords = [lm.x for lm in face_landmarks.landmark]
            y_coords = [lm.y for lm in face_landmarks.landmark]
            x_min = max(int(min(x_coords)*w), 0)
            y_min = max(int(min(y_coords)*h), 0)
            x_max = min(int(max(x_coords)*w), w)
            y_max = min(int(max(y_coords)*h), h)
            face_img = image[y_min:y_max, x_min:x_max]

            emotion = analyze_emotion(face_img)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255,0,0), 2)
            cv2.putText(image, emotion, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

        # If Face Mesh fails, fallback to OpenCV Haar cascade
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            for (x, y, fw, fh) in faces:
                face_img = image[y:y+fh, x:x+fw]
                emotion = analyze_emotion(face_img)
                cv2.rectangle(image, (x, y), (x+fw, y+fh), (0,0,255), 2)
                cv2.putText(image, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0,0,255), 2)
                break  # analyze only first detected face

        cv2.imshow("Hand and Facial Expression Recognition", image)

        if cv2.waitKey(5) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
