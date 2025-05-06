import os
import time
import threading
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# Set model names
MODEL_PRIMARY = "distilbert-base-uncased"  # DistilBERT
MODEL_FALLBACK = "meta-llama/Llama-2-7b-chat-hf"  # TinyLlama

# Set HuggingFace token
HF_TOKEN = os.getenv("HF_API_KEY")
RETRY_INTERVAL = 120  # seconds (2 minutes)

model = None
tokenizer = None
model_loaded = False

def load_model(model_name):
    global model, tokenizer, model_loaded
    if not HF_TOKEN:
        print("ERROR: HF_API_KEY environment variable is not set.")
        model_loaded = False
        return

    try:
        print(f"Loading model: {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=HF_TOKEN,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model_loaded = True
        print(f"Model {model_name} loaded successfully.")
    except Exception as e:
        print(f"Model load failed for {model_name}: {str(e)}")
        model_loaded = False

# Background thread to auto-retry model load every X seconds
def background_model_reload():
    while True:
        if not model_loaded:
            print("Retrying model load...")
            load_model(MODEL_PRIMARY)  # Try to load the primary model
            if not model_loaded:
                print("Switching to fallback model...")
                load_model(MODEL_FALLBACK)  # Fallback to a heavier model
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
    # Initial attempt to load the primary model
    load_model(MODEL_PRIMARY)

    # Start background retry thread
    threading.Thread(target=background_model_reload, daemon=True).start()

    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))