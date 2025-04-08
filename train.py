import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load and prepare data
df = pd.read_csv("training_data.csv")

# Encode the target variable
label_encoder = LabelEncoder()
df['condition'] = label_encoder.fit_transform(df['condition'])

# Features and target
X = df.drop("condition", axis=1)
y = df["condition"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Parameters for classification
params = {
    'objective': 'multi:softmax',
    'num_class': 3,
    'eval_metric': 'mlogloss',
    'eta': 0.1,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}

# Train model
model = xgb.train(params, dtrain, num_boost_round=100)

# Evaluate
preds = model.predict(dtest)
print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds, target_names=label_encoder.classes_))

# Save model and label encoder
model.save_model("health_condition_model.json")
pd.to_pickle(label_encoder, "label_encoder.pkl")