import base64
import pytest
import json
import numpy as np
from io import BytesIO
from PIL import Image
from ml_client import app, preprocess_image


@pytest.fixture
def client():
    """
    Fixture to set up the Flask test client.
    """
    with app.test_client() as client:
        yield client


def create_image_array():
    """
    Helper function to create a dummy image array for testing.
    """
    image = Image.new("RGB", (224, 224), color=(255, 0, 0))  # Create a red image
    image_array = np.array(image, dtype=np.uint8)  # Use uint8 for compatibility with OpenCV
    return image_array


def test_classify_success(client):
    """
    Test the /classifyml endpoint with a valid image array.
    """
    # Create a dummy image and encode it as base64
    image = Image.new("RGB", (300, 300), color=(255, 0, 0))  # Create a red image
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # Send the base64 image in the payload
    response = client.post(
        "/classifyml",
        data=json.dumps({"image_data": f"data:image/png;base64,{base64_image}"}),
        content_type="application/json",
    )
    assert response.status_code == 200
    assert "result" in response.json


def test_classify_missing_image_array(client):
    """
    Test the /classifyml endpoint with a missing image array.
    """
    response = client.post(
        "/classifyml", data=json.dumps({}), content_type="application/json"
    )
    assert response.status_code == 400
    assert "error" in response.json
    assert response.json["error"] == "No image data provided"


def test_preprocess_image():
    """
    Test the preprocess_image function to ensure resizing and normalization.
    """
    image_array = create_image_array()
    processed_image = preprocess_image(image_array)  # No .numpy() call; preprocess_image already returns NumPy array
    assert processed_image.shape == (300, 300, 3)  # Match model input shape
    assert (processed_image >= 0).all() and (processed_image <= 1).all()