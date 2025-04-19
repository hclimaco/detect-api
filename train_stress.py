import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load stress data
df_stress = pd.read_csv("Stress-Lysis.csv")

# Prepare features and target
X = df_stress.drop('Stress Level', axis=1)
y = df_stress['Stress Level']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model (multi-class classification)
params = {
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'num_class': 3,
    'max_depth': 5,
    'eta': 0.1,
    'subsample': 0.8
}

model = xgb.train(
    params,
    xgb.DMatrix(X_train, label=y_train),
    num_boost_round=200
)

# Evaluate
preds = model.predict(xgb.DMatrix(X_test)).argmax(axis=1)
print(f"Stress Level Detection Accuracy: {accuracy_score(y_test, preds):.2f}")
print(classification_report(y_test, preds))

# Save model
joblib.dump(model, "stress_model.pkl")

# Save feature names for API
feature_names = list(X.columns)
joblib.dump(feature_names, "stress_features.pkl")