import os
import numpy as np
import pickle
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model + files
model = load_model("model_clean.keras")
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# Home
@app.route("/")
def home():
    return render_template("index.html")

# Prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form

        # Input values
        input_data = [
            float(data["river"]),
            float(data["water"]),
            float(data["elevation"]),
            float(data["pop"]),
            float(data["infra"]),
            float(data["history"])
        ]

        # Encode categorical
        land = data["land"]
        soil = data["soil"]

        land_map = {"Agricultural": 0, "Forest": 1, "Urban": 2}
        soil_map = {"Clay": 0, "Sandy": 1, "Loamy": 2}

        input_data.append(land_map[land])
        input_data.append(soil_map[soil])

        # Scale
        final_input = scaler.transform([input_data])

        # Predict
        prediction = model.predict(final_input)[0][0]

        # Convert to readable output
        if prediction > 0.7:
            result = f"HIGH RISK 🚨 ({round(prediction*100,2)}%)"
        elif prediction > 0.4:
            result = f"MEDIUM RISK ⚠️ ({round(prediction*100,2)}%)"
        else:
            result = f"LOW RISK ✅ ({round(prediction*100,2)}%)"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return str(e)

# Chatbot (basic AI upgrade)
@app.route("/chat", methods=["POST"])
def chat():
    msg = request.form["message"].lower()

    if "flood" in msg:
        reply = "Floods happen due to heavy rainfall, river overflow, or poor drainage."
    elif "safety" in msg:
        reply = "Move to higher ground, avoid water, and keep emergency supplies ready."
    elif "precaution" in msg:
        reply = "Monitor weather alerts and keep evacuation plans ready."
    else:
        reply = "I can help with flood risks, safety tips, and predictions."

    return jsonify({"reply": reply})

# ✅ IMPORTANT (Render Fix)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
