import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load heat stroke data
df_heat = pd.read_csv("HeatStroke.csv")

# Prepare features and target
X = df_heat.drop('Heat stroke', axis=1)
y = df_heat['Heat stroke']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 6,
    'eta': 0.1,
    'subsample': 0.8
}

model = xgb.train(
    params,
    xgb.DMatrix(X_train, label=y_train),
    num_boost_round=150
)

# Evaluate
preds = (model.predict(xgb.DMatrix(X_test))) > 0.5
print(f"Heat Stroke Detection Accuracy: {accuracy_score(y_test, preds):.2f}")
print(classification_report(y_test, preds))

# Save model
joblib.dump(model, "heat_stroke_model.pkl")

# Save feature names for API
feature_names = list(X.columns)
joblib.dump(feature_names, "heat_stroke_features.pkl")