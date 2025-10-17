from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import cv2
import mediapipe as mp
import numpy as np
import pickle
import itertools
import copy

app = Flask(__name__)
CORS(app)

# ---- Load Model ----
try:
    with open("gesture_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("ERROR: gesture_model2.pkl not found. Please train the model first.")
    exit()


# ---- Mediapipe Hands ----
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# ---- Global State ----
camera_enabled = False
gestures_enabled = True

# ---- Preprocessing Function (must match training) ----
def preprocess_landmarks(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]
    for i in range(len(temp_landmark_list)):
        temp_landmark_list[i][0] -= base_x
        temp_landmark_list[i][1] -= base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))
    if max_value > 0:
        temp_landmark_list = [n / max_value for n in temp_landmark_list]
    return temp_landmark_list

# ---- API Endpoints ----
@app.route('/toggle', methods=['POST'])
def toggle_camera():
    global camera_enabled
    data = request.get_json()
    camera_enabled = data.get("enabled", False)
    return jsonify({"success": True, "camera_enabled": camera_enabled})

@app.route('/toggle_gestures', methods=['POST'])
def toggle_gestures():
    global gestures_enabled
    # This logic allows the peace sign to re-activate the system
    data = request.get_json()
    prediction = data.get("prediction", "") if data else ""
    
    if not gestures_enabled and prediction == "peace_sign":
        gestures_enabled = True
    elif gestures_enabled and prediction == "peace_sign":
        gestures_enabled = False
    # This part handles the button click from the popup
    elif not data: 
        gestures_enabled = not gestures_enabled
        
    print(f"Gestures Toggled. New state: {gestures_enabled}")
    return jsonify({"success": True, "gestures_enabled": gestures_enabled})


@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        "camera_enabled": camera_enabled,
        "gestures_enabled": gestures_enabled
    })


@app.route('/frame', methods=['POST'])
def frame():
    data = request.get_json()
    image_data = data.get("image", "")
    if not image_data.startswith("data:image/jpeg;base64,"):
        return jsonify({"success": False, "error": "Invalid image"}), 400

    img_bytes = base64.b64decode(image_data.split(",")[1])
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    frame = cv2.flip(frame, 1) # Mirror the image

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        landmark_list = [[lm.x, lm.y] for lm in hand_landmarks.landmark]
        features = preprocess_landmarks(landmark_list)
        
        # Use the simple model.predict()
        prediction = model.predict([features])[0]
        
        # Only process the peace sign if gestures are disabled
        if not gestures_enabled and prediction != 'peace_sign':
            return jsonify({"success": False, "command": "gestures_disabled"})

        print(f"Prediction: {prediction}")
        # Send back the simple prediction
        return jsonify({"success": True, "command": prediction})

    return jsonify({"success": False, "command": "no_hand"})

if __name__ == "__main__":
    app.run(debug=True)

