from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)
model = pipeline("text-classification", model="roberta-base-openai-detector")

@app.route("/predict", methods=["POST"])
def predict():
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
    app.run(host="0.0.0.0", port=10000)
