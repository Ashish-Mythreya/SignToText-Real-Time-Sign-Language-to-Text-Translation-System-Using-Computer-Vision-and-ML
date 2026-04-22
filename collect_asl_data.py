import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

DATA_DIR = "asl_dataset"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

labels = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
num_samples = 100 

def extract_landmarks(hand_landmarks):
    """
    Normalized Landmark Extraction:
    1. Translation: Wrist becomes (0,0)
    2. Scaling: All distances are relative to the palm size
    """
    # Use the wrist (landmark 0) as the base point
    base_x = hand_landmarks.landmark[0].x
    base_y = hand_landmarks.landmark[0].y
    
    # Calculate palm size (distance from wrist to middle finger base) for scaling
    palm_size = ((hand_landmarks.landmark[9].x - base_x)**2 + 
                 (hand_landmarks.landmark[9].y - base_y)**2)**0.5
    
    if palm_size == 0: palm_size = 1 # Avoid division by zero

    landmarks = []
    for lm in hand_landmarks.landmark:
        # Normalize coordinates: (Value - Base) / Scale
        norm_x = (lm.x - base_x) / palm_size
        norm_y = (lm.y - base_y) / palm_size
        # Z is already depth-relative, but we'll include it for completeness
        landmarks.extend([norm_x, norm_y, lm.z]) 
        
    return np.array(landmarks)

cap = cv2.VideoCapture(0)
data = []
label_count = {label: 0 for label in labels}

print("Instructions: Press 'n' to capture, 'q' to quit.")
current_label = labels[0]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, f"Collecting: {current_label} ({label_count[current_label]}/{num_samples})",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Data Collection', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('n') and results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            features = extract_landmarks(hand_landmarks)
            data.append(np.append(features, current_label))  
            label_count[current_label] += 1
        
        if label_count[current_label] >= num_samples:
            idx = labels.index(current_label)
            if idx < len(labels) - 1:
                current_label = labels[idx + 1]
            else:
                print("Data collection complete!")
                break

    if key == ord('q'): break

# Save Data
columns = [f"lm_{i}_{coord}" for i in range(21) for coord in ['x', 'y', 'z']] + ['label']
df = pd.DataFrame(data, columns=columns)
df.to_csv(os.path.join(DATA_DIR, "asl_dataset_a_z.csv"), index=False)
cap.release()
cv2.destroyAllWindows()