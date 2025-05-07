import os
import requests
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import signal
import sys

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# Get your Gemini API key from Render environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

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
            text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            return text if text else "No response from AI."
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return "Error talking to AI."

    except Exception as e:
        print(f"Error during content generation: {e}")
        return "An error occurred while generating the response."

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
        response_text = generate_content(user_input)
        return jsonify({"reply": response_text})
    except Exception as e:
        print(f"Error during chat generation: {e}")
        return jsonify({"reply": "Something went wrong."}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))