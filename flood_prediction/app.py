from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import requests
import smtplib
import pandas as pd

from tensorflow.keras.models import load_model
import tensorflow as tf

app = Flask(__name__)

# Load model
model = load_model("../model_fixed.keras", compile=False)

scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# 🌍 NASA DATA (SAFE VERSION)
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
        # fallback values
        return 10, 30, 70


# 📧 EMAIL ALERT (SAFE)
def send_email(msg):
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()

        # ⚠️ Replace with your credentials
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
    return render_template('index.html')


# 🔮 PREDICTION
@app.route('/predict', methods=['POST'])
def predict():
    try:
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

        # Safe extraction
        pred_value = float(pred[0][0]) if len(pred.shape) > 1 else float(pred[0])

        # Result logic
        if pred_value > 0.7:
            result = "HIGH RISK 🚨"
            send_email("⚠️ Flood Alert: HIGH RISK!")
        elif pred_value > 0.4:
            result = "MEDIUM RISK ⚠️"
        else:
            result = "LOW RISK ✅"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return f"Error: {str(e)}"


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # IMPORTANT: use 10000 default
    app.run(host="0.0.0.0", port=port)
