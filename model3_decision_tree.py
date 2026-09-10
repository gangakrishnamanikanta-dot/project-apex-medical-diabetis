"""
Healthcare Capstone Project - Diabetes Prediction
Model 3: Decision Tree Classifier (Tuned with GridSearchCV)
Team Member 3

This script implements a Decision Tree Classifier. Decision trees are highly
interpretable and mimic clinical flowcharts (e.g., IF Glucose > 127 AND BMI > 26).
However, unpruned trees overfit by memorizing noise.
We use 5-fold GridSearchCV to find the optimal tree depth and split criteria.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import joblib

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 60)
print("  MODEL 3: DECISION TREE CLASSIFIER (WITH GRIDSEARCHCV)")
print("=" * 60)

# Step 1: Load raw data
print("\n[1] Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"    Loaded {df.shape[0]} patient rows and {df.shape[1]} columns.")

# Step 2: Handle biologically impossible zeros
zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_columns:
    df[col] = df[col].replace(0, np.nan)

# Step 3: Split into train (80%) and test (20%) sets
X = df.drop(columns=['Outcome'])
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"    Training samples: {len(X_train)} | Test samples: {len(X_test)}")

# Step 4: Impute missing values using training medians
for col in zero_columns:
    med = X_train[col].median()
    X_train[col] = X_train[col].fillna(med)
    X_test[col] = X_test[col].fillna(med)

# Step 5: Feature engineering
def add_features(data):
    d = data.copy()
    bmi_conditions = [
        (d['BMI'] < 18.5),
        (d['BMI'] >= 18.5) & (d['BMI'] < 25.0),
        (d['BMI'] >= 25.0) & (d['BMI'] < 30.0),
        (d['BMI'] >= 30.0)
    ]
    d['BMI_Category'] = np.select(bmi_conditions, [0, 1, 2, 3], default=2)
    d['Glucose_to_Insulin_ratio'] = d['Glucose'] / (d['Insulin'] + 1e-5)
    return d

X_train_eng = add_features(X_train)
X_test_eng = add_features(X_test)

# Step 6: Hyperparameter Tuning via 5-Fold GridSearchCV
# Note: Decision trees do NOT require standard scaling because splits depend only on order.
print("\n[2] Hyperparameter Tuning via 5-Fold GridSearchCV...")
param_grid = {
    'max_depth': [3, 4, 5, 6, 8, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train_eng, y_train)
best_model = grid_search.best_estimator_

print(f"    Best Parameters Found : {grid_search.best_params_}")
print(f"    Best 5-Fold CV Score  : {grid_search.best_score_ * 100:.2f}%")

# Step 7: Test best model on unseen 20% test data
print("\n[3] Evaluating on Test Data:")
y_pred = best_model.predict(X_test_eng)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print(f"    Test Accuracy : {acc * 100:.2f}%")
print(f"    Precision     : {prec:.4f}")
print(f"    Recall        : {rec:.4f}")
print(f"    F1-Score      : {f1:.4f}")

print("\n    Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"    TN: {cm[0][0]} | FP: {cm[0][1]}")
print(f"    FN: {cm[1][0]} | TP: {cm[1][1]}")

print("\n    Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Non-Diabetic (0)", "Diabetic (1)"]))

# Step 8: Show Feature Importances
print("[4] Decision Tree Feature Importance (Gini Split Contribution):")
for feat, imp in sorted(zip(X_train_eng.columns, best_model.feature_importances_), key=lambda x: x[1], reverse=True):
    print(f"    {feat:<26}: {imp * 100:.2f}%")

# Step 9: Save best model to models directory
save_path = os.path.join(MODELS_DIR, "model_3_decision_tree.pkl")
joblib.dump(best_model, save_path)
print(f"\n[5] Model saved to: {save_path}")
print("=" * 60)
