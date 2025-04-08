from flask import Flask, request, jsonify
import pandas as pd
import xgboost as xgb
import joblib

app = Flask(__name__)

# Load all models
models = joblib.load("condition_models.pkl")

# Valid conditions
VALID_CONDITIONS = ['dehydration', 'overfatigue', 'heat_stroke_risk']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Validate input
        if not data or 'health_data' not in data or 'condition' not in data:
            return jsonify({"error": "Missing health_data or condition parameter"}), 400
        
        condition = data['condition'].lower()
        if condition not in VALID_CONDITIONS:
            return jsonify({
                "error": f"Invalid condition. Must be one of: {VALID_CONDITIONS}",
                "received": condition
            }), 400
        
        # Prepare features
        health_data = data['health_data']
        features = pd.DataFrame([{
            'heart_rate': health_data.get('heartRate', 0),
            'step_count': health_data.get('stepCount', 0),
            'distance': health_data.get('distanceWalkingRunning', 0),
            'energy_burned': health_data.get('activeEnergyBurned', 0),
            'resting_hr': health_data.get('restingHeartRate', 0),
            'walking_hr_avg': health_data.get('walkingHeartRateAverage', 0)
        }])
        
        # Predict
        model = models[condition]
        proba = float(model.predict(xgb.DMatrix(features))[0])
        prediction = "abnormal" if proba >= 0.5 else "normal"
        
        return jsonify({
            "condition": condition,
            "prediction": prediction,
            "probability": proba,
            "threshold": 0.5,
            "status": "success"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)