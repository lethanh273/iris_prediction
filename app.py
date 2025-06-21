from flask import Flask, request, jsonify, render_template
import torch
import numpy as np
from train import IrisNet  # Make sure this matches your model class
import logging
logging.basicConfig(filename="requests.log", level=logging.INFO)
# 🧠 Setup
app = Flask(__name__)

# 🧠 Load model and scaler
model = IrisNet()
model.load_state_dict(torch.load("model.pt"))
model.eval()

scaler = torch.load("scaler.pt", weights_only=False)

# 🧠 Logging setup
logging.basicConfig(filename="requests.log", level=logging.INFO)

# 🌐 Home route: show form
@app.route("/")
def home():
    return render_template("index.html")

# 🔮 Predict route: accepts JSON, returns prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["features"]  # expects a list of 4 floats
        features = scaler.transform([data])  # scale input
        features_tensor = torch.tensor(features).float()

        with torch.no_grad():
            output = model(features_tensor)
            prediction = torch.argmax(output).item()

        logging.info(f"Input: {data}, Prediction: {prediction}")
        return jsonify({"prediction": prediction})

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 400

# 🏁 Run app (for local testing)
if __name__ == "__main__":
    app.run(debug=True)

