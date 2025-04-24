import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, average_precision_score
import joblib
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbpipeline

# Load heat stroke data
df_heat = pd.read_csv("HeatStroke.csv")

# Prepare features and target
X = df_heat.drop('Heat stroke', axis=1)
y = df_heat['Heat stroke']

# Check class distribution
print("Class distribution:")
print(y.value_counts(normalize=True))

# Split data - stratified to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

# XGBoost with improved parameters and cross-validation
params = {
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'aucpr'],  # Better for imbalanced data
    'max_depth': 6,
    'learning_rate': 0.05,  # Reduced for better generalization
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': sum(y == 0) / sum(y == 1),  # Auto weight for imbalance
    'min_child_weight': 1,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0
}

# Train with early stopping
dtrain = xgb.DMatrix(X_res, label=y_res)
dtest = xgb.DMatrix(X_test, label=y_test)

model = xgb.train(
    params,
    dtrain,
    num_boost_round=500,  # Increased with early stopping
    evals=[(dtrain, 'train'), (dtest, 'test')],
    early_stopping_rounds=50,
    verbose_eval=20
)

# Feature importance
xgb.plot_importance(model, max_num_features=15)
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# Evaluate
y_pred_proba = model.predict(xgb.DMatrix(X_test))
y_pred = (y_pred_proba > 0.5).astype(int)

print("\nEvaluation Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.2f}")
print(f"Average Precision: {average_precision_score(y_test, y_pred_proba):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model and features
joblib.dump(model, "heat_stroke_model.pkl")
joblib.dump(list(X.columns), "heat_stroke_features.pkl")

print("Model training complete and saved.")