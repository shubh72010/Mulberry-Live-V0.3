import os
import time
import threading
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"
HF_TOKEN = os.getenv("HF_API_KEY")
RETRY_INTERVAL = 300  # seconds (5 minutes)

model = None
tokenizer = None
model_loaded = False

def load_model():
    global model, tokenizer, model_loaded
    if not HF_TOKEN:
        print("ERROR: HF_API_KEY environment variable is not set.")
        model_loaded = False
        return

    try:
        print("Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            token=HF_TOKEN,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model_loaded = True
        print("Model loaded successfully.")
    except Exception as e:
        print("Model load failed:", str(e))
        model_loaded = False

# Background thread to auto-retry model load every X seconds
def background_model_reload():
    while True:
        if not model_loaded:
            print("Retrying model load...")
            load_model()
        time.sleep(RETRY_INTERVAL)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "").strip()
    if not user_input:
        return jsonify({"reply": "Please enter a message."})

    if not model_loaded:
        return jsonify({"reply": "Model is currently unavailable. Please try again later."})

    try:
        inputs = tokenizer.encode(user_input, return_tensors="pt").to(model.device)
        outputs = model.generate(inputs, max_length=100, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return jsonify({"reply": response})
    except torch.cuda.OutOfMemoryError:
        print("ERROR: CUDA Out Of Memory")
        return jsonify({"reply": "The server ran out of memory. Try again in a bit."}), 500
    except Exception as e:
        print("ERROR during generation:", e)
        return jsonify({"reply": "Something went wrong during response generation."}), 500

if __name__ == "__main__":
    # Initial attempt to load model
    load_model()

    # Start background retry thread
    threading.Thread(target=background_model_reload, daemon=True).start()

    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))