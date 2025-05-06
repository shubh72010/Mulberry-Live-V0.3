import os
import requests
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# Render will inject this from its env vars
HF_API_KEY = os.getenv("HF_API_KEY")

# Microsoft Phi model endpoint
API_URL = "https://api-inference.huggingface.co/models/microsoft/phi-1_5"
headers = {"Authorization": f"Bearer {HF_API_KEY}"}

def query(payload):
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}

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

    if isinstance(result, list) and result:
        return jsonify({"reply": result[0].get("generated_text", "No response from AI")})
    else:
        return jsonify({"reply": "Error: Invalid response from Hugging Face API"})

if __name__ == "__main__":
    app.run(debug=True)