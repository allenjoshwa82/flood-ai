import os
import numpy as np
import pickle
import pandas as pd   # ✅ ADDED

from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model

app = Flask(__name__)

# -------------------------
# BASE PATH (Render safe)
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------
# LOAD MODEL + SCALER (SAFE)
# -------------------------
try:
    model = load_model(
        os.path.join(BASE_DIR, "model_fixed.h5"),
        compile=False
    )
    print("✅ Model Loaded Successfully")
except Exception as e:
    print("❌ MODEL LOAD ERROR:", e)
    model = None

try:
    scaler = pickle.load(
        open(os.path.join(BASE_DIR, "scaler.pkl"), "rb")
    )
    print("✅ Scaler Loaded Successfully")
except Exception as e:
    print("❌ SCALER LOAD ERROR:", e)
    scaler = None

# -------------------------
# HOME
# -------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -------------------------
# PREDICTION
# -------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None or scaler is None:
            return "❌ Model or Scaler not loaded properly"

        data = request.form

        # -------------------------
        # CATEGORY MAPS
        # -------------------------
        land_map = {
            "Agricultural": 0,
            "Forest": 1,
            "Urban": 2,
            "Water Body": 3
        }

        soil_map = {
            "Clay": 0,
            "Sandy": 1,
            "Loamy": 2,
            "Peat": 3
        }

        # -------------------------
        # FEATURE ORDER (CRITICAL)
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

        # Debug log
        print("📊 INPUT DATA:", input_data)

        # -------------------------
        # FIXED: USE DATAFRAME (NO WARNING)
        # -------------------------
        columns = [
            "rainfall",
            "temperature",
            "humidity",
            "River Discharge (m³/s)",
            "Water Level (m)",
            "Elevation (m)",
            "Population Density",
            "Infrastructure",
            "Historical Floods",
            "Land Cover",
            "Soil Type"
        ]

        input_df = pd.DataFrame([input_data], columns=columns)

        final_input = scaler.transform(input_df)

        # -------------------------
        # PREDICT
        # -------------------------
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
        print("❌ PREDICTION ERROR:", e)
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
# ENTRY POINT
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
