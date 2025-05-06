import os
import requests
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# No need for load_dotenv, Render provides environment variables directly
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# Retrieve the Hugging Face API key from Render's environment variables
HF_API_KEY = os.getenv("HF_API_KEY")
API_URL = "https://api-inference.huggingface.co/models/EleutherAI/gpt-neo-1.3B"
headers = {"Authorization": f"Bearer {HF_API_KEY}"}

def query(payload):
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raises an HTTPError for bad responses
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

    # Check if the response contains an error
    if isinstance(result, dict) and result.get("error"):
        return jsonify({"reply": f"HF API error: {result['error']}"})

    # Ensure that the response is in the expected format
    if isinstance(result, list) and result:
        return jsonify({"reply": result[0].get("generated_text", "No response from AI")})
    else:
        return jsonify({"reply": "Error: Invalid response from Hugging Face API"})

if __name__ == "__main__":
    app.run(debug=True)