"""
A simple Flask application.
This module sets up a basic web server using Flask.
"""

import os
import random
import logging
import datetime
import requests
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv

# from pymongo.errors import PyMongoError

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongodb:27017/")
MONGO_DBNAME = os.getenv("MONGO_DBNAME", "rawf_database")

ML_CLIENT_URL = "http://ml-client:5001"  # Machine Learning Client's API endpoint


try:
    client = MongoClient(MONGO_URI)
    DATABASE = client[MONGO_DBNAME]
    logger.info("Connected to MongoDB at %s", MONGO_URI)
except ConnectionFailure as db_error:
    logger.error("Error connecting to MongoDB: %s", db_error)
    DATABASE = None

# Initialize the Flask app
app = Flask(__name__)
CORS(app)

# Secret key configuration
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "default_secret_key")

# Define a MongoDB collection for storing game results
GAME_RESULTS_COLLECTION = DATABASE["game_results"]


@app.route("/")
def home():
    """Render the index page."""
    return render_template("index.html")


@app.route("/tutorial")
def tutorial():
    """Render the tutorial page."""
    return render_template("tutorial.html")


@app.route("/game")
def game():
    """Render the game page."""
    return render_template("game.html")


@app.route("/stats")
def stats():
    """Render the stats page with game results."""
    try:
        user_wins = GAME_RESULTS_COLLECTION.count_documents({"winner": "User"})
        ties = GAME_RESULTS_COLLECTION.count_documents({"winner": "Nobody"})
        computer_wins = GAME_RESULTS_COLLECTION.count_documents({"winner": "Computer"})

        stats_data = {
            "user_wins": user_wins,
            "ties": ties,
            "computer_wins": computer_wins,
        }
        return render_template("stats.html", stats=stats_data)
    except ConnectionFailure as conn_error:
        logger.error("Error fetching stats data: %s", conn_error)
        return render_template(
            "stats.html", stats={"user_wins": 0, "ties": 0, "computer_wins": 0}
        )


@app.route("/data")
def data():
    """Return sample JSON data."""
    sample_data = {
        "name": "Dockerized Flask App",
        "description": "This is a Flask app running inside Docker.",
    }
    return jsonify({"status": "success", "data": sample_data})


@app.route("/test-db")
def test_db():
    """Test MongoDB connection and return database names."""
    try:
        GAME_RESULTS_COLLECTION.insert_one({"test_field": "test_value"})
        db_list = client.list_database_names()
        logger.info("Successfully retrieved databases: %s", db_list)
        return jsonify({"status": "success", "databases": db_list})
    except ConnectionFailure as conn_error:
        logger.error("Error retrieving databases: %s", conn_error)
        return jsonify({"status": "error", "message": str(conn_error)}), 500


@app.route("/save_game_result", methods=["POST"])
def save_game_result():
    """
    Save the result of a Rock-Paper-Scissors game into the database.
    """
    request_data = request.json  # 改名为更有意义的名字
    user_choice = request_data.get("user_choice")
    computer_choice = request_data.get("computer_choice")
    winner = request_data.get("winner")

    try:
        GAME_RESULTS_COLLECTION.insert_one(
            {
                "user_choice": user_choice,
                "computer_choice": computer_choice,
                "winner": winner,
                "timestamp": datetime.datetime.utcnow(),
            }
        )
        logger.info(
            "Game result saved: User=%s, Computer=%s, Winner=%s",
            user_choice,
            computer_choice,
            winner,
        )
        return jsonify({"status": "success", "message": "Game result saved"})
    except ValueError as value_error:
        logger.error("Value error: %s", value_error)
        return (
            jsonify({"status": "error", "message": "Failed to save game result"}),
            500,
        )


@app.route("/classify", methods=["POST"])
def classify():
    """
    Route to classify an image using the machine learning client.
    """
    try:
        # Ensure an image is provided in the request
        request_data = request.json
        if "image_data" not in request_data:
            logger.error("No image file provided in request")
            return (
                jsonify({"status": "error", "message": "No image data provided"}),
                400,
            )

        # Send the json payload to the ml_client for classification
        classify_response = requests.post(
            f"{ML_CLIENT_URL}/classifyml",
            json=request_data,
            timeout=10,
        )
        logger.info(
            "ML Client classify response: %s, %s",
            classify_response.status_code,
            classify_response.text,
        )

        if classify_response.status_code != 200:
            return jsonify({"status": "error", "message": "Classification failed"}), 500

        classify_data = classify_response.json()
        if "error" in classify_data:
            return jsonify({"status": "error", "message": classify_data["error"]}), 500

        # Extract classification result
        user_choice = classify_data["result"]

        # Simulate the computer's choice and determine the result
        computer_choice = simulate_computer_choice()
        result = determine_result(user_choice, computer_choice)

        # Save the game result to the database
        GAME_RESULTS_COLLECTION.insert_one(
            {
                "user_choice": user_choice,
                "computer_choice": computer_choice,
                "result": result,
                "timestamp": datetime.datetime.utcnow(),
            }
        )

        # Return the result to the client
        return jsonify(
            {
                "user_choice": user_choice,
                "computer_choice": computer_choice,
                "result": result,
            }
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Error during classification: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


def simulate_computer_choice():
    """
    Simulate the computer's choice for the game.
    Randomly chooses between 'rock', 'paper', and 'scissors'.
    """
    choices = ["rock", "paper", "scissors"]
    return random.choice(choices)


def determine_result(user_choice, computer_choice):
    """
    Determine the result of the game based on user and computer choices.
    """
    if user_choice == computer_choice:
        return "Nobody"

    win_conditions = {
        "rock": "scissors",
        "scissors": "paper",
        "paper": "rock",
    }
    return "User" if win_conditions[user_choice] == computer_choice else "Computer"


if __name__ == "__main__":
    app.run(
        host=os.getenv("FLASK_RUN_HOST", "0.0.0.0"),
        port=int(os.getenv("FLASK_PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "1") == "1",
    )
