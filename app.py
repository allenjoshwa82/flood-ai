from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import requests
import smtplib
import pandas as pd
import os
from tensorflow.keras.models import load_model
import tensorflow as tf

# Reduce TF logs
tf.get_logger().setLevel('ERROR')

app = Flask(__name__)

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 🔥 Lazy loading variables
model = None
scaler = None
columns = None

# 🔥 Load model only when needed
def load_artifacts():
    global model, scaler, columns

    if model is None:
        print("🔄 Loading model...")
        model = load_model(os.path.join(BASE_DIR, "model_clean.keras"), compile=False)
        scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))
        columns = pickle.load(open(os.path.join(BASE_DIR, "columns.pkl"), "rb"))
        print("✅ Model loaded")


# 🌍 NASA DATA
def get_nasa_data(lat, lon):
    try:
        url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters=PRECTOT,T2M,RH2M&community=AG&longitude={lon}&latitude={lat}&format=JSON"

        response = requests.get(url, timeout=5)
        data = response.json()

        params = data['properties']['parameter']

        rainfall = list(params['PRECTOT'].values())[-1]
        temp = list(params['T2M'].values())[-1]
        humidity = list(params['RH2M'].values())[-1]

        return rainfall, temp, humidity

    except:
        return 10, 30, 70


# 📧 EMAIL ALERT
def send_email(msg):
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login("your_email@gmail.com", "your_app_password")
        server.sendmail("your_email@gmail.com", "receiver@gmail.com", msg)
        server.quit()
    except Exception as e:
        print("Email failed:", e)


# 🤖 CHATBOT
@app.route('/chat', methods=['POST'])
def chat():
    msg = request.form['message'].lower()

    if "flood" in msg:
        reply = "Floods happen due to heavy rainfall, river overflow, and drainage failure."
    elif "safe" in msg or "precaution" in msg:
        reply = "Move to higher ground, carry essentials, avoid waterlogged roads."
    else:
        reply = "Ask me about floods, risks, or safety tips!"

    return jsonify({'reply': reply})


# 🏠 HOME
@app.route('/')
def home():
    return "Flood Prediction App is Running ✅"


# 🔮 PREDICTION
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 🔥 Load model here
        load_artifacts()

        # 🌍 Live NASA data
        rainfall, temperature, humidity = get_nasa_data(13.08, 80.27)

        # Form inputs
        river = float(request.form['river'])
        water = float(request.form['water'])
        elevation = float(request.form['elevation'])
        pop = float(request.form['pop'])
        infra = int(request.form['infra'])
        history = int(request.form['history'])
        land = request.form['land']
        soil = request.form['soil']

        # Data dictionary
        input_dict = {
            'Latitude': 13.08,
            'Longitude': 80.27,
            'rainfall': rainfall,
            'temperature': temperature,
            'humidity': humidity,
            'River Discharge (m³/s)': river,
            'Water Level (m)': water,
            'Elevation (m)': elevation,
            'Land Cover': land,
            'Soil Type': soil,
            'Population Density': pop,
            'Infrastructure': infra,
            'Historical Floods': history
        }

        df = pd.DataFrame([input_dict])

        # Encoding
        df = pd.get_dummies(df)

        # Match columns
        for col in columns:
            if col not in df:
                df[col] = 0

        df = df[columns]

        # Scaling
        df = scaler.transform(df)

        # Prediction
        pred = model.predict(df)

        pred_value = float(pred[0][0]) if len(pred.shape) > 1 else float(pred[0])

        # Result
        if pred_value > 0.7:
            result = "HIGH RISK 🚨"
            send_email("⚠️ Flood Alert: HIGH RISK!")
        elif pred_value > 0.4:
            result = "MEDIUM RISK ⚠️"
        else:
            result = "LOW RISK ✅"

        return result

    except Exception as e:
        return f"Error: {str(e)}"
    if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
