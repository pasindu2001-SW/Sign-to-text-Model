import cv2
import numpy as np
import os
import tensorflow as keras
from keras.models import load_model
import mediapipe as mp
import time

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    if results.face_landmarks:
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=1, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=1, circle_radius=2))
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(80,22,76), thickness=1, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=1, circle_radius=2))
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=1, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=1, circle_radius=2))

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

try:
    model = load_model('action.h5')  # Replace with your model path
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

actions = np.array(['hello', 'thanks', 'ayubowan','alright','how are you'])  # Replace with your action labels

sequence = []
sentence = []
predictions = []
threshold = 0.7
last_detection_time = 0
display_duration = 2  # seconds to keep displaying the detected sign

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Allow camera to initialize
time.sleep(2.0)

with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            continue  # or break to exit
            
        image, results = mediapipe_detection(frame, holistic)
        #draw_styled_landmarks(image, results)
        
        # Check if hands are visible (sign is being made)
        hands_visible = results.left_hand_landmarks is not None or results.right_hand_landmarks is not None
        
        # Prediction logic only when hands are visible
        if hands_visible:
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]
            
            if len(sequence) == 30:
                try:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    predictions.append(np.argmax(res))
                    
                    if len(predictions) >= 10 and np.unique(predictions[-10:])[0] == np.argmax(res):
                        if res[np.argmax(res)] > threshold:
                            sentence = [actions[np.argmax(res)]]
                            last_detection_time = time.time()
                except Exception as e:
                    print(f"Prediction error: {e}")
        
        # Clear the sentence if no detection for display_duration seconds
        if time.time() - last_detection_time > display_duration:
            sentence = []
        
        # Only show the rectangle and text if there's something to display
        if sentence:
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('Sign Language Detection', image)
        
        key = cv2.waitKey(10)
        if key & 0xFF == ord('q') or key & 0xFF == 27:  # ESC or 'q'
            break
    
    cap.release()
    cv2.destroyAllWindows()
