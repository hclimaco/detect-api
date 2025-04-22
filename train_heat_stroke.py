import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load heat stroke data
df_heat = pd.read_csv("HeatStroke.csv")

# Check for missing values
print("Missing values:\n", df_heat.isnull().sum())

# Prepare features and target
X = df_heat.drop('Heat stroke', axis=1)
y = df_heat['Heat stroke']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model with hyperparameter tuning (optional)
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 6,
    'eta': 0.1,
    'subsample': 0.8,
    'scale_pos_weight': len(y_train[y_train == 0]) / len(y_train[y_train == 1])  # Handle class imbalance
}

model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
print(f"Heat Stroke Detection Accuracy: {accuracy_score(y_test, preds):.2f}")
print(classification_report(y_test, preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

# Feature importance
print("Feature Importances:")
for name, importance in zip(X.columns, model.feature_importances_):
    print(f"{name}: {importance:.4f}")

# Save model
joblib.dump(model, "heat_stroke_model.pkl")