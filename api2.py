from flask import Flask, request, jsonify
import pandas as pd
import xgboost as xgb
import joblib

app = Flask(__name__)

# Load all models
heat_stroke_model = joblib.load("heat_stroke_model.pkl")
stress_model = joblib.load("stress_model.pkl")

# Load feature names
heat_stroke_features = joblib.load("heat_stroke_features.pkl")
stress_features = joblib.load("stress_features.pkl")

# Valid conditions
VALID_CONDITIONS = ['heat_stroke', 'stress']

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
        
        # Prepare features based on condition
        health_data = data['health_data']
        
        if condition == 'heat_stroke':
            # Map API input to heat stroke model features
            features = pd.DataFrame([{
                'Daily Ingested Water (L)': health_data.get('water_intake', 0),
                'Time of year (month)': health_data.get('month', 1),
                'Cardiovascular disease history': health_data.get('has_cvd', 0),
                'Dehydration': health_data.get('is_dehydrated', 0),
                'Heat Index (HI)': health_data.get('heat_index', 0),
                'Diastolic BP': health_data.get('diastolic_bp', 0),
                'Environmental temperature (C)': health_data.get('env_temp', 0),
                'Systolic BP': health_data.get('systolic_bp', 0),
                'Weight (kg)': health_data.get('weight', 0),
                'Patient temperature': health_data.get('body_temp', 37),
                'Relative Humidity': health_data.get('humidity', 0),
                'Exposure to sun': health_data.get('sun_exposure', 0),
                'BMI': health_data.get('bmi', 0),
                'Heart / Pulse rate (b/min)': health_data.get('heart_rate', 0),
                'Age': health_data.get('age', 30),
                'Sweating': health_data.get('is_sweating', 0),
                'Strenuous exercise': health_data.get('strenuous_exercise', 0),
                'Sex': health_data.get('sex', 0),
                'Time of day': health_data.get('hour_of_day', 12)
            }])
            
            # Predict
            proba = float(heat_stroke_model.predict(xgb.DMatrix(features))[0])
            prediction = "high_risk" if proba >= 0.5 else "low_risk"
            
            return jsonify({
                "condition": condition,
                "prediction": prediction,
                "probability": proba,
                "threshold": 0.5,
                "status": "success"
            })
            
        elif condition == 'stress':
            # Map API input to stress model features
            features = pd.DataFrame([{
                'Humidity': health_data.get('humidity', 0),
                'Temperature': health_data.get('temperature', 0),
                'Step count': health_data.get('step_count', 0)
            }])
            
            # Predict (multi-class)
            proba = stress_model.predict(xgb.DMatrix(features))[0]
            prediction = int(proba.argmax())
            confidence = float(proba.max())
            
            stress_levels = {
                0: "low_stress",
                1: "medium_stress",
                2: "high_stress"
            }
            
            return jsonify({
                "condition": condition,
                "prediction": stress_levels[prediction],
                "confidence": confidence,
                "probabilities": {
                    "low_stress": float(proba[0]),
                    "medium_stress": float(proba[1]),
                    "high_stress": float(proba[2])
                },
                "status": "success"
            })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)