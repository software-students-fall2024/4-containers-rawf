"""
The Machine Learning Client Component of Our Web App.
This Module Loads A Trained Machine Learning Model, Preprocesses an Input Image,
Then Classifies the Image as Either Rock, Paper, or Scissors.
"""

import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import (  # pylint: disable=import-error,no-name-in-module
    load_model,  # pylint: disable=import-error,no-name-in-module
)  # pylint: disable=import-error,no-name-in-module


# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
MODEL_PATH = "model/rps_model.h5"
model = load_model(MODEL_PATH)

# Define the rock-paper-scissors categories
categories = {0: "rock", 1: "paper", 2: "scissors"}


def preprocess_image(image):
    """
    Resize and normalize the image for the model.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # pylint: disable=no-member
    image = cv2.resize(image, (300, 300))  # pylint: disable=no-member
    image = image / 255.0  # Normalize pixel values to [0, 1]
    return image


def mapper(prediction):
    """
    Map the model's numeric prediction to a category.
    """
    return categories[prediction]


@app.route("/classifyml", methods=["POST"])
def classifyml():
    """
    API endpoint to classify an image.
    Expects a single image uploaded via form-data with the key 'image'.
    """
    try:
        # Ensure an image is provided in the request

        # Parse the JSON payload
        request_data = request.get_json()
        if "image_data" not in request_data:
            return jsonify({"error": "No image data provided"}), 400

        # Decode the Base64 image data
        image_data = request_data["image_data"].split(",")[1]  # Remove the prefix
        image_bytes = base64.b64decode(image_data)

        file_bytes = np.frombuffer(image_bytes, np.uint8)  # pylint: disable=no-member
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # pylint: disable=no-member

        if image is None:
            return jsonify({"error": "Invalid image data"}), 400

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Add batch dimension and make the prediction
        prediction = model.predict(processed_image.reshape(1, 300, 300, 3))
        result = mapper(np.argmax(prediction[0]))

        return jsonify({"result": result})

    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error during classification: {e}")  # Log the error
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
