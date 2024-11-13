from flask import Flask, jsonify

# Initialize the Flask app
app = Flask(__name__)

# Define a route for the homepage
@app.route('/')
def home():
    return jsonify(message="Hello, Flask is running in Docker!")

# Define another sample route
@app.route('/data')
def data():
    sample_data = {
        "name": "Dockerized Flask App",
        "description": "This is a Flask app running inside Docker."
    }
    return jsonify(sample_data)

# Run the app only if this file is executed directly
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Host set to '0.0.0.0' for Docker compatibility
