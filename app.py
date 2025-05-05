import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

HF_API_KEY = os.getenv("HF_API_KEY")
API_URL = "https://api-inference.huggingface.co/models/EleutherAI/gpt-neo-1.3B"
headers = {"Authorization": f"Bearer {HF_API_KEY}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    if not user_input:
        return jsonify({"response": "No input provided."})

    data = query({
        "inputs": user_input,
        "parameters": {"max_new_tokens": 100}
    })

    # Handle possible errors
    if isinstance(data, dict) and data.get("error"):
        return jsonify({"response": f"HF API error: {data['error']}"})

    return jsonify({"response": data[0]["generated_text"]})

@app.route("/", methods=["GET"])
def home():
    return "Mulberry-Live is active"