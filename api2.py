from flask import Flask, request, jsonify
import pandas as pd
import xgboost as xgb
import pickle

app = Flask(__name__)

# Load model and label encoder
model = xgb.Booster()
model.load_model("health_condition_model.json")
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# Expected features (must match training data)
EXPECTED_FEATURES = [
    "heart_rate",
    "step_count",
    "distance",
    "energy_burned",
    "resting_hr",
    "walking_hr_avg"
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Validate input
        if not data:
            return jsonify({"error": "No input data provided"}), 400
        
        # Create feature vector
        features = {
            "heart_rate": data.get("heartRate", 0),
            "step_count": data.get("stepCount", 0),
            "distance": data.get("distanceWalkingRunning", 0),
            "energy_burned": data.get("activeEnergyBurned", 0),
            "resting_hr": data.get("restingHeartRate", 0),
            "walking_hr_avg": data.get("walkingHeartRateAverage", 0)
        }
        
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # Predict
        dmatrix = xgb.DMatrix(df)
        prediction = model.predict(dmatrix)
        condition = label_encoder.inverse_transform([int(prediction[0])])[0]
        
        return jsonify({
            "status": "success",
            "prediction": condition,
            "confidence": float(max(model.predict(dmatrix, output_margin=True)[0]))
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)