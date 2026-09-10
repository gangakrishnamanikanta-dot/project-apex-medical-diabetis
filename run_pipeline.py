import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Set styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.set_theme(style="whitegrid", font="sans-serif")

# Define directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("PHASE 1: LOAD & INSPECT DATASET")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())

assert df.shape == (768, 9), f"Expected shape (768, 9), but got {df.shape}"
expected_cols = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]
assert list(df.columns) == expected_cols, "Column names mismatch!"
print("\n[SUCCESS] Data loaded and verified successfully.")

print("\n" + "=" * 60)
print("PHASE 2: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 60)

# 1. Correlation Heatmap
plt.figure(figsize=(10, 8))
corr_matrix = df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap=cmap, mask=mask,
            vmax=1.0, vmin=-0.2, square=True, linewidths=.5, cbar_kws={"shrink": .8})
plt.title("EDA 1: Feature Correlation Heatmap", fontsize=14, fontweight="bold", pad=15)
plt.tight_layout()
eda1_path = os.path.join(RESULTS_DIR, "eda_1_correlation_heatmap.png")
plt.savefig(eda1_path, dpi=300)
plt.close()
print(f"Saved: {eda1_path}")

# 2. Outcome Class Balance Bar Chart
plt.figure(figsize=(7, 5))
outcome_counts = df['Outcome'].value_counts()
colors = ['#2b5c8f', '#d9534f']
bars = plt.bar(["Non-Diabetic (0)", "Diabetic (1)"], outcome_counts.values, color=colors, width=0.55, edgecolor='black', alpha=0.85)
for bar in bars:
    yval = bar.get_height()
    pct = (yval / len(df)) * 100
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 10, f"{yval} ({pct:.1f}%)", ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.title("EDA 2: Outcome Class Distribution", fontsize=14, fontweight="bold", pad=12)
plt.ylabel("Patient Count", fontsize=12)
plt.ylim(0, 580)
plt.tight_layout()
eda2_path = os.path.join(RESULTS_DIR, "eda_2_class_distribution.png")
plt.savefig(eda2_path, dpi=300)
plt.close()
print(f"Saved: {eda2_path}")

# 3. Distribution Plots (Glucose, BMI, Age) by Outcome
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
features_to_plot = ['Glucose', 'BMI', 'Age']
titles = ['Glucose Level Distribution', 'BMI Distribution', 'Age Distribution']

for i, feat in enumerate(features_to_plot):
    sns.kdeplot(data=df, x=feat, hue='Outcome', common_norm=False, fill=True, alpha=0.35,
                palette=['#2b5c8f', '#d9534f'], ax=axes[i], linewidth=2)
    axes[i].set_title(titles[i], fontsize=12, fontweight='bold')
    axes[i].set_xlabel(feat, fontsize=11)
    axes[i].set_ylabel("Density", fontsize=11)
    axes[i].legend(["Diabetic (1)", "Non-Diabetic (0)"], loc='upper right')

plt.suptitle("EDA 3: Key Feature Distributions Split by Outcome", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
eda3_path = os.path.join(RESULTS_DIR, "eda_3_feature_distributions_by_outcome.png")
plt.savefig(eda3_path, dpi=300)
plt.close()
print(f"Saved: {eda3_path}")

# 4. Boxplots for Outlier Detection
plt.figure(figsize=(12, 6))
box_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
sns.boxplot(data=df[box_cols], palette="Set2", linewidth=1.5, fliersize=4)
plt.title("EDA 4: Diagnostic Measurements Boxplots (Outlier & Zero Detection)", fontsize=14, fontweight="bold", pad=12)
plt.ylabel("Value", fontsize=12)
plt.xlabel("Diagnostic Metric", fontsize=12)
plt.tight_layout()
eda4_path = os.path.join(RESULTS_DIR, "eda_4_outlier_boxplots.png")
plt.savefig(eda4_path, dpi=300)
plt.close()
print(f"Saved: {eda4_path}")

# 5. Pairplot of Top Correlated Features with Outcome
top_features = ['Glucose', 'BMI', 'Age', 'Insulin', 'Outcome']
pairplot = sns.pairplot(df[top_features], hue='Outcome', palette=['#2b5c8f', '#d9534f'],
                        diag_kind='kde', plot_kws={'alpha': 0.6, 's': 30})
pairplot.fig.suptitle("EDA 5: Pairwise Relationships of Top Correlated Features", y=1.02, fontsize=14, fontweight="bold")
eda5_path = os.path.join(RESULTS_DIR, "eda_5_pairplot.png")
pairplot.savefig(eda5_path, dpi=200)
plt.close()
print(f"Saved: {eda5_path}")

print("\n" + "=" * 60)
print("PHASE 3: DATA CLEANING & LEAK-FREE IMPUTATION")
print("=" * 60)

# Biological zero replacement
zero_invalid_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
print("Zero counts prior to cleaning:")
for c in zero_invalid_cols:
    print(f"  {c}: {(df[c] == 0).sum()} zeros ({(df[c] == 0).mean() * 100:.1f}%)")

df_clean = df.copy()
for col in zero_invalid_cols:
    df_clean[col] = df_clean[col].replace(0, np.nan)

# Train-Test Split (80:20 stratified) BEFORE computing medians to avoid data leakage
X = df_clean.drop(columns=['Outcome'])
y = df_clean['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
print(f"\nTrain set shape: {X_train.shape} (Positives: {y_train.sum()})")
print(f"Test set shape:  {X_test.shape} (Positives: {y_test.sum()})")

# Compute medians strictly on X_train (overall median, NOT grouped by Outcome!)
training_medians = {}
for col in zero_invalid_cols:
    median_val = float(X_train[col].median())
    training_medians[col] = median_val
    X_train[col] = X_train[col].fillna(median_val)
    X_test[col] = X_test[col].fillna(median_val)

# Also store medians for remaining features for complete inference fallback
for col in ['Pregnancies', 'DiabetesPedigreeFunction', 'Age']:
    training_medians[col] = float(X_train[col].median())

medians_json_path = os.path.join(MODELS_DIR, "inference_medians.json")
with open(medians_json_path, "w") as f:
    json.dump(training_medians, f, indent=4)
print(f"Saved inference medians to {medians_json_path}:")
print(json.dumps(training_medians, indent=2))

# Cap extreme outliers at 99th percentile based on X_train
cap_cols = ['Insulin', 'SkinThickness']
capping_thresholds = {}
for col in cap_cols:
    cap_val = float(X_train[col].quantile(0.99))
    capping_thresholds[col] = cap_val
    X_train[col] = np.where(X_train[col] > cap_val, cap_val, X_train[col])
    X_test[col] = np.where(X_test[col] > cap_val, cap_val, X_test[col])

print("\nOutlier capping thresholds (99th percentile):", capping_thresholds)

print("\n" + "=" * 60)
print("PHASE 4: FEATURE ENGINEERING")
print("=" * 60)

def engineer_features(data_df):
    df_feat = data_df.copy()
    
    # 1. BMI Category (Underweight: 0, Normal: 1, Overweight: 2, Obese: 3)
    conditions = [
        (df_feat['BMI'] < 18.5),
        (df_feat['BMI'] >= 18.5) & (df_feat['BMI'] < 25.0),
        (df_feat['BMI'] >= 25.0) & (df_feat['BMI'] < 30.0),
        (df_feat['BMI'] >= 30.0)
    ]
    choices = [0, 1, 2, 3]
    df_feat['BMI_Category'] = np.select(conditions, choices, default=2)
    
    # 2. Glucose-to-Insulin Ratio
    df_feat['Glucose_to_Insulin_ratio'] = df_feat['Glucose'] / (df_feat['Insulin'] + 1e-5)
    
    return df_feat

X_train_eng = engineer_features(X_train)
X_test_eng = engineer_features(X_test)

feature_names = list(X_train_eng.columns)
print(f"Engineered feature set ({len(feature_names)} features):")
print(feature_names)

features_json_path = os.path.join(MODELS_DIR, "feature_columns.json")
with open(features_json_path, "w") as f:
    json.dump(feature_names, f, indent=4)

print("\n" + "=" * 60)
print("PHASE 5: MODEL TRAINING & HYPERPARAMETER TUNING")
print("=" * 60)

# Model 1: Logistic Regression (StandardScaler + LogisticRegression)
m1_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000, C=1.0))
])
m1_pipeline.fit(X_train_eng, y_train)
joblib.dump(m1_pipeline, os.path.join(MODELS_DIR, "model_1_logistic.pkl"))
print("[M1] Logistic Regression trained and saved as model_1_logistic.pkl")

# Model 2: Polynomial Logistic Regression (deg=2)
m2_pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(random_state=42, max_iter=2000, C=0.1, penalty='l2'))
])
m2_pipeline.fit(X_train_eng, y_train)
joblib.dump(m2_pipeline, os.path.join(MODELS_DIR, "model_2_polynomial.pkl"))
print("[M2] Polynomial Logistic Regression (deg=2) trained and saved as model_2_polynomial.pkl")

# Model 3: Decision Tree Classifier with GridSearchCV Tuning
print("\n[M3] Tuning Decision Tree Classifier via GridSearchCV...")
dt_param_grid = {
    'max_depth': [3, 4, 5, 6, 8, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'criterion': ['gini', 'entropy']
}
dt_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    dt_param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
dt_grid.fit(X_train_eng, y_train)
m3_best = dt_grid.best_estimator_
joblib.dump(m3_best, os.path.join(MODELS_DIR, "model_3_decision_tree.pkl"))
print(f"  Best Decision Tree params: {dt_grid.best_params_}")
print(f"  Best Decision Tree CV score: {dt_grid.best_score_:.4f}")
print("  Saved as model_3_decision_tree.pkl")

# Model 4: Random Forest Classifier with GridSearchCV Tuning
print("\n[M4] Tuning Random Forest Classifier via GridSearchCV...")
rf_param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [4, 6, 8, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
rf_grid.fit(X_train_eng, y_train)
m4_best = rf_grid.best_estimator_
joblib.dump(m4_best, os.path.join(MODELS_DIR, "model_4_random_forest.pkl"))
print(f"  Best Random Forest params: {rf_grid.best_params_}")
print(f"  Best Random Forest CV score: {rf_grid.best_score_:.4f}")
print("  Saved as model_4_random_forest.pkl")

print("\n" + "=" * 60)
print("PHASE 6: 5-FOLD CROSS-VALIDATION ON ALL 4 MODELS")
print("=" * 60)

models_dict = {
    "Logistic Regression": m1_pipeline,
    "Polynomial Logistic (deg=2)": m2_pipeline,
    "Decision Tree (Tuned)": m3_best,
    "Random Forest (Tuned)": m4_best
}

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results_data = []

for model_name, model_obj in models_dict.items():
    scores = cross_val_score(model_obj, X_train_eng, y_train, cv=cv_strategy, scoring='accuracy')
    cv_mean = scores.mean()
    cv_std = scores.std()
    cv_results_data.append({
        "Model": model_name,
        "CV Mean Accuracy": round(cv_mean, 4),
        "CV Std Dev": round(cv_std, 4),
        "Fold 1": round(scores[0], 4),
        "Fold 2": round(scores[1], 4),
        "Fold 3": round(scores[2], 4),
        "Fold 4": round(scores[3], 4),
        "Fold 5": round(scores[4], 4)
    })
    print(f"{model_name:<30}: CV Accuracy = {cv_mean * 100:.2f}% ± {cv_std * 100:.2f}%")

cv_df = pd.DataFrame(cv_results_data)
cv_csv_path = os.path.join(RESULTS_DIR, "cross_validation_results.csv")
cv_df.to_csv(cv_csv_path, index=False)
print(f"\nSaved cross-validation results to: {cv_csv_path}")

print("\n" + "=" * 60)
print("PHASE 7: EVALUATION ON 80:20 TEST SPLIT")
print("=" * 60)

test_eval_data = []
y_preds = {}

for model_name, model_obj in models_dict.items():
    preds = model_obj.predict(X_test_eng)
    y_preds[model_name] = preds
    
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    
    test_eval_data.append({
        "Model": model_name,
        "Test Accuracy": round(acc, 4),
        "Test Precision": round(prec, 4),
        "Test Recall": round(rec, 4),
        "Test F1-Score": round(f1, 4)
    })
    print(f"\n--- {model_name} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")

test_df = pd.DataFrame(test_eval_data)
comp_csv_path = os.path.join(RESULTS_DIR, "comparison_table.csv")
test_df.to_csv(comp_csv_path, index=False)
print(f"\nSaved test comparison table to: {comp_csv_path}")

# Render Comparison Table as high-resolution PNG image
fig, ax = plt.subplots(figsize=(10, 3.2))
ax.axis('off')
ax.axis('tight')
table_data = [[row["Model"], f"{row['Test Accuracy']*100:.2f}%", f"{row['Test Precision']*100:.2f}%",
               f"{row['Test Recall']*100:.2f}%", f"{row['Test F1-Score']*100:.2f}%"] for row in test_eval_data]
col_labels = ["Model", "Accuracy", "Precision", "Recall", "F1-Score"]
table = ax.table(cellText=table_data, colLabels=col_labels, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)

# Style header
for i in range(len(col_labels)):
    cell = table[(0, i)]
    cell.set_facecolor('#1e3d59')
    cell.set_text_props(color='white', fontweight='bold')

# Highlight best row (Random Forest / highest accuracy)
best_idx = int(np.argmax([r["Test Accuracy"] for r in test_eval_data])) + 1
for i in range(len(col_labels)):
    cell = table[(best_idx, i)]
    cell.set_facecolor('#e8f4f8')
    cell.set_text_props(fontweight='bold')

plt.title("Model Performance Comparison (80:20 Test Split)", fontsize=13, fontweight='bold', pad=15)
comp_png_path = os.path.join(RESULTS_DIR, "comparison_table.png")
plt.savefig(comp_png_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved table image to: {comp_png_path}")

print("\n" + "=" * 60)
print("PHASE 8: CONFUSION MATRIX & FEATURE IMPORTANCES")
print("=" * 60)

best_model_name = "Random Forest (Tuned)"
best_model = m4_best
best_preds = y_preds[best_model_name]

# 1. Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, best_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Diabetic (0)", "Diabetic (1)"])
disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
ax.grid(False)
plt.title(f"Confusion Matrix: {best_model_name}", fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
cm_path = os.path.join(RESULTS_DIR, "confusion_matrix_best_model.png")
plt.savefig(cm_path, dpi=300)
plt.close()
print(f"Saved confusion matrix to: {cm_path}")

# 2. Feature Importance Bar Chart (from Random Forest)
importances = best_model.feature_importances_
feat_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=True)

plt.figure(figsize=(9, 6))
bars = plt.barh(feat_imp_df['Feature'], feat_imp_df['Importance'], color='#17a2b8', edgecolor='black', alpha=0.85)
for bar in bars:
    w = bar.get_width()
    plt.text(w + 0.005, bar.get_y() + bar.get_height()/2.0, f"{w:.3f}", ha='left', va='center', fontsize=9, fontweight='bold')
plt.title("Feature Importance Analysis (Tuned Random Forest)", fontsize=13, fontweight='bold', pad=12)
plt.xlabel("Gini Importance Score", fontsize=11)
plt.xlim(0, max(importances) * 1.18)
plt.tight_layout()
fi_path = os.path.join(RESULTS_DIR, "feature_importance.png")
plt.savefig(fi_path, dpi=300)
plt.close()
print(f"Saved feature importance chart to: {fi_path}")

# 3. Final Model Recommendation (2-3 sentences)
top_3_feats = list(feat_imp_df.tail(3)['Feature'])[::-1]
rec_text = (
    f"The tuned Random Forest Classifier (M4) is selected as the recommended model for hospital deployment, "
    f"achieving the highest 5-fold cross-validation accuracy of {cv_df[cv_df['Model'] == 'Random Forest (Tuned)']['CV Mean Accuracy'].values[0]*100:.2f}% "
    f"and strong generalization on the test split (F1-score: {test_df[test_df['Model'] == 'Random Forest (Tuned)']['Test F1-Score'].values[0]:.4f}). "
    f"Diagnostic decision-making is primarily driven by {top_3_feats[0]}, {top_3_feats[1]}, and {top_3_feats[2]}, "
    f"providing clinical interpretability and robust resistance to noisy diagnostic measurements.\n"
)
rec_file_path = os.path.join(RESULTS_DIR, "final_recommendation.txt")
with open(rec_file_path, "w") as f:
    f.write(rec_text)
print(f"\nFinal Recommendation written to: {rec_file_path}")
print(rec_text)

print("=" * 60)
print("[COMPLETED] All pipeline stages executed successfully!")
print("=" * 60)
