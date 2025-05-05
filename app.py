from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)
CORS(app)

# Load GPT-Neo
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
generate = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")

    prompt = f"User: {user_message}\nAI:"
    response = generate(prompt, max_length=100, do_sample=True, temperature=0.9, top_p=0.95, pad_token_id=tokenizer.eos_token_id)[0]["generated_text"]
    
    # Extract only AI part
    ai_reply = response.split("AI:")[-1].strip().split("\n")[0]

    return jsonify({"response": ai_reply})

if __name__ == "__main__":
    app.run(debug=True)