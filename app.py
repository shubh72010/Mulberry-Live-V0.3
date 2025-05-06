import os
import requests
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

HF_API_KEY = os.getenv("HF_API_KEY")
API_URL = "https://api-inference.huggingface.co/models/EleutherAI/gpt-neo-1.3B"
headers = {"Authorization": f"Bearer {HF_API_KEY}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    if not user_input:
        return jsonify({"reply": "No input provided."})
    
    result = query({
        "inputs": user_input,
        "parameters": {"max_new_tokens": 100}
    })

    if isinstance(result, dict) and result.get("error"):
        return jsonify({"reply": f"HF API error: {result['error']}"})
    
    return jsonify({"reply": result[0]["generated_text"]})

if __name__ == "__main__":
    app.run(debug=True)