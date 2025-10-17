import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import copy
import itertools

# --- 1. CONFIGURATION ---
actions = [
    'open_palm', 'fist', 'thumbs_up', 'thumbs_down', 'pointing_left', 
    'pointing_right', 'l_shape', 'ok_sign', 'peace_sign',
    'call_me', 'three_fingers_up', 'bull_sign'
]
DATA_PATH = "gesture_dataset"
SAMPLES_PER_GESTURE = 300
action_key_map = {
    'p': 'open_palm', 'f': 'fist', 'u': 'thumbs_up', 'd': 'thumbs_down',
    'l': 'pointing_left', 'r': 'pointing_right', 'n': 'l_shape',
    'c': 'ok_sign', 's': 'peace_sign',
    'y': 'call_me', '3': 'three_fingers_up', 'b': 'bull_sign'
}

# --- 2. SETUP ---
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

dataset = []
action_counts = {action: 0 for action in actions}

# --- 3. ADVANCED PREPROCESSING FUNCTION ---
def preprocess_landmarks(landmark_list):
    """
    This function takes the raw landmark list and converts it into a 
    preprocessed format suitable for the model.
    1. Makes coordinates relative to the wrist.
    2. Normalizes the coordinates to be scale-invariant.
    """
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0: # Wrist landmark
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))
    def normalize_(n):
        return n / max_value if max_value > 0 else 0

    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list

# --- 4. MAIN COLLECTION LOOP ---
while True:
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    
    y_pos = 20
    for i, (action, count) in enumerate(action_counts.items()):
        col = 220 if i >= 6 else 20
        row = y_pos + (i % 6) * 30
        
        if count >= SAMPLES_PER_GESTURE:
            color = (0, 255, 0) # Green for complete
            text = f"{action.replace('_', ' ').title()}: DONE ({count})"
        else:
            color = (255, 255, 255) # White for in-progress
            text = f"{action.replace('_', ' ').title()}: {count}"
        
        cv2.putText(frame, text, (col, row), cv2.FONT_HERSHEY_PLAIN, 1.5, color, 2)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
    
    key = cv2.waitKey(10)
    if key & 0xFF == ord('q'):
        break
    
    key_char = chr(key & 0xFF)
    if key_char in action_key_map:
        action = action_key_map[key_char]
        
        if action_counts[action] >= SAMPLES_PER_GESTURE:
            print(f"'{action}' is already complete. Not collecting more samples.")
        elif results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmark_list = [[lm.x, lm.y] for lm in hand_landmarks.landmark]
            preprocessed_landmarks = preprocess_landmarks(landmark_list)
            
            full_row = [action] + preprocessed_landmarks
            dataset.append(full_row)
            action_counts[action] += 1
            print(f"Collected sample for gesture: {action} (Total: {action_counts[action]})")
        else:
            print("!!! NO HAND DETECTED. Sample NOT collected. Try again. !!!")

    if all(count >= SAMPLES_PER_GESTURE for count in action_counts.values()):
        cv2.putText(frame, "COLLECTION COMPLETE!", (50, 240), 
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        cv2.imshow("Data Collection", frame)
        cv2.waitKey(2000)
        break

    cv2.imshow("Data Collection", frame)

# --- 5. SAVE THE FINAL DATASET ---
cap.release()
cv2.destroyAllWindows()

if dataset:
    csv_file_path = os.path.join(DATA_PATH, 'dataset.csv')
    print(f"Saving dataset with {len(dataset)} samples to {csv_file_path}...")

    header = ['label'] + [f'coord_{i}' for i in range(42)]

    with open(csv_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(dataset)
    print("Dataset saved successfully!")
else:
    print("No data was collected.")

