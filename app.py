import os
import numpy as np
import pickle
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model

app = Flask(__name__)

# -------------------------
# BASE PATH (Render safe)
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------
# LOAD MODEL + SCALER
# -------------------------
model = load_model(
    os.path.join(BASE_DIR, "model_fixed.h5"),
    compile=False
)

scaler = pickle.load(
    open(os.path.join(BASE_DIR, "scaler.pkl"), "rb")
)

# -------------------------
# HOME
# -------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -------------------------
# PREDICTION (FIXED & SAFE)
# -------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form

        land_map = {
            "Agricultural": 0,
            "Forest": 1,
            "Urban": 2,
            "Water Body": 3
        }

        soil_map = {
            "Clay": 0,
            "Sandy": 1,
            "Loamy": 2
        }

        # -------------------------
        # EXACT FEATURE ORDER (MUST MATCH TRAINING)
        # -------------------------
        input_data = [
            float(data.get("rainfall", 0)),
            float(data.get("temperature", 0)),
            float(data.get("humidity", 0)),
            float(data.get("river_discharge", 0)),
            float(data.get("water_level", 0)),
            float(data.get("elevation", 0)),
            float(data.get("population_density", 0)),
            float(data.get("infrastructure", 0)),
            float(data.get("historical_floods", 0)),
            land_map.get(data.get("land", "Urban"), 2),
            soil_map.get(data.get("soil", "Loamy"), 2)
        ]

        # Scale input
        final_input = scaler.transform([input_data])

        # Predict
        prediction = float(model.predict(final_input, verbose=0)[0][0])

        # -------------------------
        # RESULT LOGIC
        # -------------------------
        if prediction > 0.7:
            result = f"HIGH RISK 🚨 ({round(prediction * 100, 2)}%)"
        elif prediction > 0.4:
            result = f"MEDIUM RISK ⚠️ ({round(prediction * 100, 2)}%)"
        else:
            result = f"LOW RISK ✅ ({round(prediction * 100, 2)}%)"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return f"Error: {str(e)}"


# -------------------------
# CHATBOT
# -------------------------
@app.route("/chat", methods=["POST"])
def chat():
    msg = request.form.get("message", "").lower()

    if "flood" in msg:
        reply = "Floods happen due to heavy rainfall, river overflow, or poor drainage."
    elif "safety" in msg:
        reply = "Move to higher ground and avoid flooded areas."
    elif "precaution" in msg:
        reply = "Monitor weather alerts and keep emergency kits ready."
    else:
        reply = "Ask me about flood risk, safety tips, or predictions."

    return jsonify({"reply": reply})


# -------------------------
# RENDER ENTRY POINT
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
