"""
This module contains unit tests for the ml_client.py Flask application.
"""

import json
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import pytest
from ml_client import app, preprocess_image


@pytest.fixture
def client_fixture():
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
    image_array = np.array(image, dtype=np.uint8)  # Ensure dtype is uint8
    return image_array


def encode_image_to_base64(image_array):
    """
    Helper function to encode a NumPy image array to a Base64 string.
    """
    image = Image.fromarray(image_array)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def test_classify_success(client_fixture):  # pylint: disable=redefined-outer-name
    """
    Test the /classifyml endpoint with a valid image array.
    """
    image_array = create_image_array()
    base64_image = encode_image_to_base64(image_array)
    response = client_fixture.post(
        "/classifyml",
        data=json.dumps({"image_data": f"data:image/png;base64,{base64_image}"}),
        content_type="application/json",
    )
    assert response.status_code == 200
    assert "result" in response.json


def test_classify_missing_image_array(
    client_fixture,
):  # pylint: disable=redefined-outer-name
    """
    Test the /classifyml endpoint with a missing image array.
    """
    response = client_fixture.post(
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
    processed_image = preprocess_image(image_array)
    assert processed_image.shape == (300, 300, 3)  # Match the model's input shape
    assert (processed_image >= 0).all() and (processed_image <= 1).all()
