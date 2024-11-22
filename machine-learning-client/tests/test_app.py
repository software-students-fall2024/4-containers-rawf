import pytest
from app import process_image

def test_process_image_invalid_data():
    result, status = process_image("invalid_base64_data")
    assert result is None
    assert status == "Invalid Image Data"