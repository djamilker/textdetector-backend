from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from transformers import pipeline
import os

app = Flask(__name__)
CORS(app)  # Just in case, but the decorator is what really fixes it

model = pipeline("text-classification", model="roberta-base-openai-detector")

@app.route("/predict", methods=["POST", "OPTIONS"])
@cross_origin(origins=["https://textdetectorai.online", "https://www.textdetectorai.online"])
def predict():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    try:
        data = request.get_json()
        text = data.get("text", "")
        if not text.strip():
            return jsonify({"error": "No text provided"}), 400

        result = model(text)[0]
        label = result["label"]
        score = round(result["score"] * 100, 2)
        return jsonify({"label": label, "confidence": score})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "Text Detector API is running."

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
