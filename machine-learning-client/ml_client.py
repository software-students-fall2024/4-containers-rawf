import numpy as np
import os
import base64
import binascii
import cv2

from datetime import datetime
from pymongo import MongoClient
from cvzone.HandTrackingModule import HandDetector

# Initialize
MODEL_PATH = "model/rps_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)  # pylint: disable=no-member
app = Flask(__name__)
detector = HandDetector(max_hands=1) # setup hand detector module

print(f"Model input shape: {model.input_shape}")

# Analyze image to detect Rock, Paper, or Scissors
@app.route("/classify", methods=["POST"])
def classify(image_data):
    try:
        image_data = base64.b64decode(image_data.split(",")[1])
    except (IndexError, ValueError, binascii.Error):
        return None, "Invalid Image Data"
    
    # Convert image to NP array
    arr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    hands, image = detector.findHands(image, draw=False)

    # hand detected
    if hands:
        fingers = detector.fingersUp(hands[0])

        # ML Classification
        if fingers == [0,0,0,0,0]: return "Rock"
        elif fingers == [1,1,1,1,1]: return "Paper"
        elif fingers == [0,1,1,0,0]: return "Scissors"
        else: return "Unkown choice"

    # hand not detected
    else: return "No hand detected"

# API enpoint to store game results
# Expects JSON request with keys "user_choice", "computer_choice", and "result"
@app.route("/store", methods=["POST"])
def store():
    try:
        # Extract data from the JSON body
        data = request.json
        user_choice = data.get("user_choice")
        computer_choice = data.get("computer_choice")
        result = data.get("result")

        if not all([user_choice, computer_choice, result]):
            return jsonify({"error": "Missing required fields"}), 400

        # Here, you could implement MongoDB storage if needed
        # For now, we mock the storage response
        stored_data = {
            "timestamp": datetime.now().isoformat(),
            "user_choice": user_choice,
            "computer_choice": computer_choice,
            "result": result,
        }

        return jsonify({"status": "success", "data": stored_data})
    except Exception as e:  # pylint: disable=broad-except
        return jsonify({"error": str(e)}), 500

# API endpoint to preprocess uploaded image
@app.route("/preprocess", methods=["POST"])
def preprocess():
    try:
        image = request.files["image"]  # Get the uploaded file
        image_array = preprocess_image(image)  # Preprocess the image
        return jsonify({"image_array": image_array.tolist()})
    except Exception as e:  # pylint: disable=broad-except
        return jsonify({"error": f"Preprocessing failed: {str(e)}"}), 500

# Resize and normalize image for model
def preprocess_image(image):
    image = tf.image.resize(image, (300, 300))  # Resize to match the model input shape
    image = image / 255.0  # Normalize pixel values
    return image

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
