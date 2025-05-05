from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from frontend

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")
    headers = {
        "Authorization": f"Bearer {os.getenv('HF_TOKEN')}"
    }
    data = {
        "inputs": user_message
    }

    try:
        response = requests.post(
            "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium",
            headers=headers,
            json=data
        )
        result = response.json()
        reply = result[0]['generated_text'] if isinstance(result, list) else "Sorry, no reply."
    except Exception as e:
        reply = "Error talking to AI."

    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)  # Production settings