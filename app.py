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
# PREDICTION
# -------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form

        # Categorical encoding
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
        # MUST MATCH TRAINING ORDER
        # -------------------------
        input_data = [
            float(data["rainfall"]),
            float(data["temperature"]),
            float(data["humidity"]),
            float(data["river_discharge"]),
            float(data["water_level"]),
            float(data["elevation"]),
            float(data["population_density"]),
            float(data["infrastructure"]),
            float(data["historical_floods"]),
            land_map[data["land"]],
            soil_map[data["soil"]]
        ]

        # Scale input
        final_input = scaler.transform([input_data])

        # Predict
        prediction = float(model.predict(final_input)[0][0])

        # Output logic
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
    msg = request.form["message"].lower()

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
