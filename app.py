import os
import time
import threading
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests
import signal
import sys

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# Google Cloud Gemini API key from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
RETRY_INTERVAL = 120  # Retry every 2 minutes

# Flag to check model status
model_loaded = False

def generate_content(user_input):
    # Check if the API key is available
    if not GEMINI_API_KEY:
        print("ERROR: Google Cloud API key not set in env.")
        return "API key not found. Please set your key in the environment variables."

    try:
        # Define the Gemini API URL for content generation
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        
        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            "contents": [{
                "parts": [{"text": user_input}]
            }]
        }

        # Make the POST request to the Google Gemini API
        response = requests.post(url, headers=headers, params={"key": GEMINI_API_KEY}, json=data)

        if response.status_code == 200:
            result = response.json()
            return result.get("generated_content", "No content generated.")
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return "Something went wrong with the API request."

    except Exception as e:
        print(f"Error during content generation: {e}")
        return "An error occurred while generating the response."

def background_model_reload():
    while True:
        if not model_loaded:
            print("Retrying model load...")
            # Optionally retry content generation or API initialization logic here
        time.sleep(RETRY_INTERVAL)

# Graceful shutdown
def signal_handler(sig, frame):
    print("Shutting down...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "").strip()
    if not user_input:
        return jsonify({"reply": "Please enter a message."})

    try:
        response = generate_content(user_input)
        return jsonify({"reply": response})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"reply": "Something went wrong."}), 500

if __name__ == "__main__":
    threading.Thread(target=background_model_reload, daemon=True).start()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))