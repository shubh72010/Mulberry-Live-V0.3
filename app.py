import os
import logging
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError, InvalidArgument

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# Load Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Missing Google Gemini API key in environment variable 'GEMINI_API_KEY'.")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-pro")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "").strip()

    if not user_input:
        logger.warning("Empty message received.")
        return jsonify({"reply": "Please enter a message."})

    try:
        logger.info(f"User input: {user_input}")
        response = model.generate_content(user_input)
        reply = getattr(response, "text", "").strip()
        if not reply:
            logger.warning("No content returned by Gemini.")
            reply = "The AI didn’t return anything useful. Try again."
        return jsonify({"reply": reply})

    except InvalidArgument as e:
        logger.error(f"Invalid input: {e}")
        return jsonify({"reply": "Invalid request format or content. Try rephrasing."}), 400

    except GoogleAPIError as e:
        logger.error(f"Google API error: {e}")
        if "403" in str(e):
            return jsonify({"reply": "API access denied. Check your key or quota."}), 403
        elif "429" in str(e):
            return jsonify({"reply": "Rate limit hit. Please slow down."}), 429
        else:
            return jsonify({"reply": "AI service error. Try again shortly."}), 502

    except Exception as e:
        logger.exception("Unexpected error:")
        return jsonify({"reply": "Unexpected error occurred. Try again later."}), 500

@app.route("/health")
def health():
    return "ok", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)