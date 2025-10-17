import cv2
import mediapipe as mp
import numpy as np
import pickle
import copy
import itertools

# --- 1. LOAD THE TRAINED MODEL ---
model_filename = 'gesture_model.pkl'
print(f"Loading model from {model_filename}...")
with open(model_filename, 'rb') as f:
    model = pickle.load(f)
print("Model loaded successfully.")

# --- 2. SETUP MEDIAPIPE AND OPENCV ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# --- 3. COPY THE PREPROCESSING FUNCTION ---
# This must be the EXACT same function from your data collection script
def preprocess_landmarks(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    max_value = max(list(map(abs, temp_landmark_list)))
    def normalize_(n):
        return n / max_value if max_value > 0 else 0

    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list

# --- 4. VARIABLES FOR STABILITY ---
prediction_history = []
HISTORY_LENGTH = 10 # Check the last 10 frames
predicted_gesture = "Waiting..."
confidence_score = 0.0 # Variable to hold the confidence

# --- 5. MAIN LOOP ---
while True:
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    debug_frame = copy.deepcopy(frame) # Create a copy for drawing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(rgb_frame)

    current_prediction = None
    current_confidence = 0.0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Preprocess the landmarks for the model
            landmark_list = [[lm.x, lm.y] for lm in hand_landmarks.landmark]
            preprocessed_landmarks = preprocess_landmarks(landmark_list)

            # --- NEW: Get probabilities instead of just the prediction ---
            probabilities = model.predict_proba([preprocessed_landmarks])
            
            # Find the highest probability and its corresponding gesture
            current_confidence = np.max(probabilities)
            predicted_index = np.argmax(probabilities)
            current_prediction = model.classes_[predicted_index]

            prediction_history.append(current_prediction)

            # Keep history to a certain length
            if len(prediction_history) > HISTORY_LENGTH:
                prediction_history.pop(0)

            # Check if the last N predictions are the same for a stable result
            if len(prediction_history) == HISTORY_LENGTH and len(set(prediction_history)) == 1:
                predicted_gesture = prediction_history[0]
                confidence_score = current_confidence # Update confidence of the stable gesture

            # Draw the hand skeleton
            mp_drawing.draw_landmarks(debug_frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
    else:
        # If no hand is detected, clear the history
        prediction_history.clear()
        predicted_gesture = "No Hand Detected"
        confidence_score = 0.0

    # --- NEW: Display the gesture and confidence score ---
    display_text = f"{predicted_gesture.replace('_', ' ').title()}"
    if confidence_score > 0:
        display_text += f" ({confidence_score:.0%})" # Format as percentage

    cv2.putText(debug_frame, f"Gesture: {display_text}", 
                (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cv2.imshow("Gesture Recognition Test", debug_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- CLEANUP ---
cap.release()
cv2.destroyAllWindows()

