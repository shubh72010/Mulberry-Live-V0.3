import os
import time
import threading
import requests
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import signal
import sys

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# Models and API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Ensure this is set in your environment
RETRY_INTERVAL = 120  # Retry every 2 minutes

model_loaded = False

def generate_content(user_input):
    if not GEMINI_API_KEY:
        print("ERROR: Google Cloud API key not set in env.")
        return "API key not found. Please set your key in the environment variables."

    try:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
        headers = { "Content-Type": "application/json" }
        data = {
            "contents": [{
                "parts": [{ "text": user_input }]
            }]
        }

        response = requests.post(url, headers=headers, params={"key": GEMINI_API_KEY}, json=data)
        
        if response.status_code == 200:
            result = response.json()
            # Safely extract the response text from the JSON
            text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            return text if text else "No response from AI."
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return "Error talking to AI."

    except Exception as e:
        print(f"Error during content generation: {e}")
        return "An error occurred while generating the response."

def background_model_reload():
    while True:
        if not model_loaded:
            print("Retrying model load...")
            model_loaded = True  # For now, set to True immediately to skip background model loading
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

    if not model_loaded:
        return jsonify({"reply": "Model is loading or unavailable. Try again soon."})

    try:
        # Call the Gemini API with the user's message
        response_text = generate_content(user_input)
        return jsonify({"reply": response_text})
    except Exception as e:
        print(f"Error during chat generation: {e}")
        return jsonify({"reply": "Something went wrong."}), 500

if __name__ == "__main__":
    threading.Thread(target=background_model_reload, daemon=True).start()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))