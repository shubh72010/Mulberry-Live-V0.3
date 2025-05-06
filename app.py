import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# Load a small but functional model
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

    try:
        inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
        outputs = model.generate(inputs, max_length=100, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(outputs[:, inputs.shape[-1]:][0], skip_special_tokens=True)
        return jsonify({"reply": response})
    except Exception as e:
        print("ERROR:", e)
        return jsonify({"reply": "Something went wrong on the server."}), 500

# Needed for local testing (ignored by Render if you have a Procfile)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))