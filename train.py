import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load data
df = pd.read_csv("training_data.csv")

# Create binary targets for each condition
conditions = ['dehydration', 'overfatigue', 'heat_stroke_risk']
models = {}

for condition in conditions:
    print(f"\nTraining {condition} classifier...")
    
    # Create binary target (1 = has condition, 0 = normal)
    y = (df['condition'] == condition).astype(int)
    X = df.drop('condition', axis=1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 4,
        'eta': 0.1,
        'subsample': 0.8
    }
    model = xgb.train(
        params,
        xgb.DMatrix(X_train, label=y_train),
        num_boost_round=100
    )
    
    # Evaluate
    preds = (model.predict(xgb.DMatrix(X_test))) > 0.5
    print(f"Accuracy for {condition}: {accuracy_score(y_test, preds):.2f}")
    print(classification_report(y_test, preds))
    
    # Save model
    models[condition] = model

# Save all models
joblib.dump(models, "condition_models.pkl")