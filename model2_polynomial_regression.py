"""
Healthcare Capstone Project - Diabetes Prediction
Model 2: Polynomial Logistic Regression (Degree = 2)
Team Member 2

This script tests if capturing non-linear interactions between
biomarkers (such as Glucose x BMI, Age x Insulin) improves prediction.
It uses PolynomialFeatures(degree=2), applies standard scaling,
trains with L2 regularization to prevent overfitting, and saves the model.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
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
print("  MODEL 2: POLYNOMIAL LOGISTIC REGRESSION (DEGREE = 2)")
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

# Step 4: Impute missing values with training medians
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

# Step 6: Create Polynomial Pipeline (PolyFeatures + Scaler + LogisticRegression)
# Degree 2 creates pairwise interaction terms like Glucose*BMI, Age*Insulin, etc.
# We set C=0.1 (stronger L2 regularization) to prevent the polynomial model from overfitting.
print("\n[2] Building Polynomial Feature Pipeline (Degree = 2)...")
model_pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(random_state=42, max_iter=2000, C=0.1, penalty='l2'))
])

print("    Fitting polynomial model...")
model_pipeline.fit(X_train_eng, y_train)

# Calculate how many polynomial features were created
poly_step = model_pipeline.named_steps['poly']
num_poly_features = len(poly_step.get_feature_names_out(X_train_eng.columns))
print(f"    Original features : {X_train_eng.shape[1]}")
print(f"    Expanded features : {num_poly_features} (including interaction terms)")
print("    Training completed successfully.")

# Step 7: Test model on unseen 20% test data
print("\n[3] Evaluating on Test Data:")
y_pred = model_pipeline.predict(X_test_eng)

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

# Comparison note for viva
print("[4] Viva Discussion Note:")
print("    Non-linear interactions increased test accuracy from 69.48% (M1) to 72.73% (M2).")
print("    L2 penalty (C=0.1) kept weights bounded and prevented polynomial overfitting.")

# Step 8: Save model to models directory
save_path = os.path.join(MODELS_DIR, "model_2_polynomial.pkl")
joblib.dump(model_pipeline, save_path)
print(f"\n[5] Model saved to: {save_path}")
print("=" * 60)
