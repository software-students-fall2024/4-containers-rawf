# app.py for ML Client
import numpy as np
import os
import base64
import binascii
import cv2

from datetime import datetime
from pymongo import MongoClient
from cvzone.HandTrackingModule import HandDetector

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI")
DATABASE_NAME = os.getenv("MONGO_DBNAME")
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
detector = HandDetector(max_hands=1) # setup hand detector module

# Get unprocessed image from MongoDB
def get_image():
    return db.images.find_one({"processed": False}, sort=[("_id", -1)])

# Analyze image to detect Rock, Paper, or Scissors
def process_image(image_data):
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

# Update database with processed result
def update_db(image_id, result):
    db.images.update_one(
        {"_id": image_id},
        {"$set": {"proceed":True, "result":result, "processed_at":datetime.utcnow()}}
    )

# Main Code - process images
def main():
    while True:
        image = get_image()
        if not image:
            print("No unprocessed images found. Waiting...")
            break
        
        result, status = process_image(image["data"])
        if result:
            update_db(image["_id"], result)
            print(f"Processed Image: {result}")
        else:
            print(f"Failed to process image: {status}")

if __name__ == "__main__":
    main()
