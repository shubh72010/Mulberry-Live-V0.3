from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)
CORS(app)

tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    output = model.generate(inputs, max_length=100)
    response = tokenizer.decode(output[:, inputs.shape[-1]:][0], skip_special_tokens=True)
    return jsonify({"reply": response})

if __name__ == "__main__":
    app.run()