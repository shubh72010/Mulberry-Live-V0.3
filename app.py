import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# Load DialoGPT model and tokenizer from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    if not user_input:
        return jsonify({"reply": "No input provided."})

    # Encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Append the new user input tokens to the chat history
    chat_history_ids = new_user_input_ids

    # Generate a response from the model
    bot_output = model.generate(chat_history_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Get the model's response, decode it, and add it to the chat history
    bot_output_text = tokenizer.decode(bot_output[:, chat_history_ids.shape[-1]:][0], skip_special_tokens=True)

    # Return the bot's response
    return jsonify({"reply": bot_output_text})

if __name__ == "__main__":
    app.run(debug=True)