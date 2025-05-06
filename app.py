import os
import time
import threading
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import signal
import sys

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# Models
MODEL_PRIMARY = "deepseek-ai/deepseek-coder-6.7b-instruct"
MODEL_FALLBACK = "mistralai/Mistral-7B-Instruct-v0.2"

HF_TOKEN = os.getenv("HF_API_KEY")
RETRY_INTERVAL = 120  # Retry every 2 minutes

model = None
tokenizer = None
model_loaded = False
current_model = None

def load_model(model_name):
    global model, tokenizer, model_loaded, current_model

    if not HF_TOKEN:
        print("ERROR: Hugging Face token not set in env.")
        model_loaded = False
        return

    try:
        print(f"Loading model: {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=HF_TOKEN,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        model_loaded = True
        current_model = model_name
        print(f"Model {model_name} loaded successfully.")
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        model_loaded = False

def background_model_reload():
    while True:
        if not model_loaded:
            print("Retrying model load...")
            load_model(MODEL_PRIMARY)
            if not model_loaded:
                print("Falling back to backup model...")
                load_model(MODEL_FALLBACK)
        time.sleep(RETRY_INTERVAL)

# Graceful shutdown
def signal_handler(sig, frame):
    print("Shutting down...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "").strip()
    if not user_input:
        return jsonify({"reply": "Please enter a message."})

    if not model_loaded:
        return jsonify({"reply": "Model is loading or unavailable. Try again soon."})

    try:
        inputs = tokenizer(user_input, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return jsonify({"reply": response})
    except torch.cuda.OutOfMemoryError:
        print("ERROR: CUDA OOM")
        return jsonify({"reply": "GPU memory ran out. Try again later."}), 500
    except Exception as e:
        print(f"Generation error: {e}")
        return jsonify({"reply": "Something went wrong."}), 500

if __name__ == "__main__":
    load_model(MODEL_PRIMARY)
    threading.Thread(target=background_model_reload, daemon=True).start()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))