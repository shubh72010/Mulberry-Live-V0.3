import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError, InvalidArgument

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
        return jsonify({"reply": "Please enter a message."})

    try:
        response = model.generate_content(user_input)
        print("Full Gemini response:", response)

        # Try .text first
        if hasattr(response, "text") and response.text:
            reply = response.text.strip()
        # Try candidates fallback
        elif hasattr(response, "candidates") and response.candidates:
            parts = response.candidates[0].content.parts
            reply = "".join([p.text for p in parts if hasattr(p, "text")]).strip()
        else:
            reply = "No content generated."

        return jsonify({"reply": reply})

    except InvalidArgument as e:
        print(f"Invalid input: {e}")
        return jsonify({"reply": "Invalid request format or content. Try rephrasing."}), 400

    except GoogleAPIError as e:
        print(f"Google API error: {e}")
        if "403" in str(e):
            return jsonify({"reply": "API access denied. Check your key or quota."}), 403
        elif "429" in str(e):
            return jsonify({"reply": "Rate limit hit. Please slow down."}), 429
        return jsonify({"reply": "AI service error. Try again shortly."}), 502

    except Exception as e:
        print(f"Unexpected error: {e}")
        return jsonify({"reply": "Unexpected error occurred. Try again later."}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)