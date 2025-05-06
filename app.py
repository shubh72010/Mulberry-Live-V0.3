from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# Load a lightweight but decent model
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    if not user_input:
        return jsonify({"reply": "No input provided."})

    inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    output = model.generate(inputs, max_length=100, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output[:, inputs.shape[-1]:][0], skip_special_tokens=True)

    return jsonify({"reply": response})

if __name__ == "__main__":
    pass  # Keep empty for Render deployment