from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import os

app = Flask(__name__)

# EXPLICIT CORS ALLOWANCE
CORS(app, resources={r"/predict": {"origins": [
    "https://textdetectorai.online",
    "https://www.textdetectorai.online"
]}})

model = pipeline("text-classification", model="roberta-base-openai-detector")

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        # CORS preflight response
        response = jsonify({})
        response.headers.add("Access-Control-Allow-Origin", request.headers.get("Origin"))
        response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
        response.headers.add("Access-Control-Allow-Methods", "POST,OPTIONS")
        return response

    try:
        data = request.get_json()
        text = data.get("text", "")
        if not text.strip():
            return jsonify({"error": "No text provided"}), 400

        result = model(text)[0]
        label = result["label"]
        score = round(result["score"] * 100, 2)
        response = jsonify({"label": label, "confidence": score})
        response.headers.add("Access-Control-Allow-Origin", request.headers.get("Origin"))
        return response
    except Exception as e:
        response = jsonify({"error": str(e)})
        response.headers.add("Access-Control-Allow-Origin", request.headers.get("Origin"))
        return response, 500

@app.route("/", methods=["GET"])
def home():
    return "Text Detector API is running."

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
