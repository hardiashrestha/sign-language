from flask import Flask, render_template, Response, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
from utils.gesture_recognition import get_finger_status, recognize_gesture
import threading
import time

app = Flask(__name__)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# Global variables
camera = None
frame_lock = threading.Lock()
current_frame = None
current_gesture = "No hands detected"
is_streaming = False

def init_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return camera

def gen_frames():
    global current_frame, current_gesture, is_streaming, camera
    
    is_streaming = True
    camera = init_camera()
    
    while is_streaming:
        success, frame = camera.read()
        if not success:
            print("Failed to read frame from camera")
            time.sleep(0.1)
            continue
        
        try:
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
                        2, (0, 255, 0), 3)
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.03)  # ~30 FPS
            
        except Exception as e:
            print(f"Error in frame processing: {e}")
            time.sleep(0.1)
            continue

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

@app.teardown_appcontext
def cleanup(error):
    global camera, is_streaming
    is_streaming = False
    if camera is not None:
        camera.release()

if __name__ == '__main__':
    print("Starting Hand Gesture Recognition App...")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
