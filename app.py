from flask import Flask, render_template, Response, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
from utils.gesture_recognition import get_finger_status, recognize_gesture
import threading
import base64

app = Flask(__name__)


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=5, min_detection_confidence=0.7)


camera = None
frame_lock = threading.Lock()
current_frame = None
current_gesture = "No hands detected"

def gen_frames():
    global current_frame, current_gesture, camera
    
    camera = cv2.VideoCapture(0)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Flip frame
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = hands.process(frame_rgb)
        
        gesture_text = "No hands detected"
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get gesture
                finger_status = get_finger_status(hand_landmarks)
                gesture_text = recognize_gesture(finger_status)
        
        # Update global variables
        with frame_lock:
            current_gesture = gesture_text
            current_frame = frame.copy()
        
        # Add text to frame
        cv2.putText(frame, gesture_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (0, 255, 0), 3)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/gesture')
def get_gesture():
    with frame_lock:
        gesture = current_gesture
    return jsonify({'gesture': gesture})

@app.route('/api/health')
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)