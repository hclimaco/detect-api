import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load and format raw health data with condition labels
def prepare_training_data(input_json, output_csv):
    with open(input_json, "r") as f:
        raw_data = json.load(f)
    
    processed_data = []
    
    for entry in raw_data:
        # Each entry should contain health metrics AND a 'condition' field
        processed_entry = {
            "heart_rate": entry.get("heartRate", 0),
            "step_count": entry.get("stepCount", 0),
            "distance": entry.get("distanceWalkingRunning", 0),
            "energy_burned": entry.get("activeEnergyBurned", 0),
            "resting_hr": entry.get("restingHeartRate", 0),
            "walking_hr_avg": entry.get("walkingHeartRateAverage", 0),
            "condition": entry.get("condition")  # This should be 'dehydration', 'overfatigue', or 'heat_stroke_risk'
        }
        processed_data.append(processed_entry)
    
    df = pd.DataFrame(processed_data)
    df.to_csv(output_csv, index=False)
    print(f"Data saved to {output_csv}")

if __name__ == "__main__":
    prepare_training_data("raw_health_data.json", "training_data.csv")