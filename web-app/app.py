import random
import os
import logging
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv
import requests

# Configuration
logging.basicConfig(level=logging.INFO) # logger
logger = logging.getLogger(__name__)

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongodb:27017/") # MongoDB
MONGO_DBNAME = os.getenv("MONGO_DBNAME", "rawf_database")

try:
    client = MongoClient(MONGO_URI)
    DATABASE = client[MONGO_DBNAME]
    logger.info("Connected to MongoDB at %s", MONGO_URI)
except ConnectionFailure as db_error:
    logger.error("Error connecting to MongoDB: %s", db_error)
    DATABASE = None

app = Flask(__name__) # Flask
CORS(app)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "default_secret_key")


# ---------------HTLM FILES---------------
# index.html
@app.route("/")
def home():
    return render_template("index.html")

# tutorial.html
@app.route("/tutorial")
def tutorial():
    return render_template("tutorial.html")

# game.html
@app.route("/game")
def game():
    return render_template("game.html")

# stats.html
@app.route("/stats")
def stats():
    return render_template("stats.html")


# ---------------MONGODB GAME STATISTICS---------------

# Test MongoDB connection and return database names
@app.route("/test-db")
def test_db():
    try:
        db_list = client.list_database_names()
        logger.info("Successfully retrieved databases: %s", db_list)
        return jsonify({"status": "success", "databases": db_list})
    except ConnectionFailure as db_conn_error: # Renamed variable
        logger.error("Error retrieving databases: %s", db_conn_error)
        return jsonify({"status": "error", "message": str(db_conn_error)}), 500

# NEED FUNCTION TO ADD DATA (GAME STATS) TO DATABASE!!
@app.route("/stats")
def add_data():
    return 

# Function to generate a random computer choice
def random_computer_choice():
    choices = ["Rock", "Paper", "Scissors"]
    return random.choice(choices)

# Function to determine the winner
def determine_winner(user_choice, computer_choice):
    if user_choice == computer_choice:
        return "Nobody"
    elif (user_choice == "Rock" and computer_choice == "Scissors") or \
        (user_choice == "Paper" and computer_choice == "Rock") or \
        (user_choice == "Scissors" and computer_choice == "Paper"):
        return "User"
    else:
        return "Computer"

# Route to handle the game logic
@app.route("/play_game", methods=["POST"])
def play_game():
    data = request.json
    user_choice = data.get("user_choice")  # Get the user choice from the request
    if not user_choice:
        return {"error": "No user choice provided"}, 400

    computer_choice = random_computer_choice()  # Generate computer's choice
    winner = determine_winner(user_choice, computer_choice)  # Determine the winner

    # Return a JSON response
    return {
        "user_choice": user_choice,
        "computer_choice": computer_choice,
        "winner": winner
    }


# ---------------ML CLIENT---------------
ML_CLIENT_URL = "http://ml-client:5001"  # Machine Learning Client's API endpoint

# NEED A FUNCTION TO CLASSIFY IMAGE USING THE ML CLIENT
@app.route("/game")
def classify_image():
    return 


# ---------------RUN MAIN---------------
"""
# KEEP THIS FOR NOW
if __name__ == "__main__":
    # Use Flask configuration from environment variables
    app.run(
        host=os.getenv("FLASK_RUN_HOST", "0.0.0.0"),
        port=int(os.getenv("FLASK_PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "1") == "1",
    )
"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)