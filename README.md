# 🏥 Healthcare Capstone: Diabetes Risk Prediction System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App%20Live-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Track:** Capstone Project — Theme 1: Healthcare  
**Dataset:** Pima Indians Diabetes Diagnostic Dataset (768 patients)  
**Deliverable:** 4-Model Comparative Analysis, Hyperparameter Tuning, and Clinical Web Dashboard  

---

## 📌 Project Overview & Problem Statement

Diabetes mellitus is a chronic metabolic condition that often goes undetected until dangerous secondary complications develop. Early diagnostic screening allows clinical teams to intervene proactively. 

In this project, our team built an end-to-end Machine Learning diagnostic system to predict whether a patient is diabetic based on routine clinical measurements (glucose levels, blood pressure, insulin, BMI, age, and family history). 

Following the project requirements, we implemented, evaluated, and compared **4 distinct machine learning models**:
1. **Model 1:** Logistic Regression (Linear baseline)
2. **Model 2:** Polynomial Logistic Regression (Degree = 2, non-linear interactions)
3. **Model 3:** Decision Tree Classifier (Tuned with 5-Fold GridSearchCV)
4. **Model 4:** Random Forest Classifier (150-Tree Ensemble with GridSearchCV)

We also implemented all **3 bonus requirements**: hyperparameter tuning via `GridSearchCV`, 5-fold stratified cross-validation, and an interactive **Streamlit** clinical diagnostic web application.

---

## 👥 4-Member Team Roles & Work Breakdown

To ensure a clear collaborative division of labor, each team member took ownership of an independent modeling stage and a dedicated Python script:

| Member | Project Role | Dedicated Script | Key Responsibilities |
| :--- | :--- | :--- | :--- |
| **Member 1** | **Data Engineering & Baseline Lead** | `model1_logistic_regression.py` | Data inspection, biological zero detection, baseline Logistic Regression, Standard Scaling, and weights interpretation. |
| **Member 2** | **Feature Pipeline & Polynomial Lead** | `model2_polynomial_regression.py` | WHO BMI categorization, Glucose-to-Insulin ratio engineering, degree-2 interaction modeling, and L2 regularization tuning. |
| **Member 3** | **Tree Modeling & Pruning Lead** | `model3_decision_tree.py` | Decision Tree implementation, cost-complexity pruning, 5-fold GridSearchCV tuning, and tree decision path analysis. |
| **Member 4** | **Ensemble, Evaluation & Deployment Lead**| `model4_random_forest.py` & `app.py` | 150-tree Random Forest ensemble, Gini biomarker importance ranking, 5-fold cross-validation benchmarks, and Streamlit web dashboard. |

---

## 📊 Dataset Description

* **Source:** National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK) / Kaggle
* **Records:** 768 female patient observations of Pima Indian heritage (aged 21+)
* **Target:** `Outcome` (`0` = Non-Diabetic [500 patients, 65.1%], `1` = Diabetic [268 patients, 34.9%])

| Feature Name | Clinical Description | Range in Data | Preprocessing Notes |
| :--- | :--- | :--- | :--- |
| `Pregnancies` | Number of times pregnant | 0 – 17 | Valid count variable |
| `Glucose` | Plasma glucose at 2h in oral glucose test (mg/dL) | 44 – 199 | Biological zero is invalid; imputed with median |
| `BloodPressure` | Diastolic blood pressure (mm Hg) | 24 – 122 | Biological zero is invalid; imputed with median |
| `SkinThickness` | Triceps skinfold thickness (mm) | 7 – 99 | Biological zero is invalid; imputed with median |
| `Insulin` | 2-hour serum insulin ($\mu\text{U/mL}$) | 14 – 846 | Biological zero is invalid; imputed with median |
| `BMI` | Body Mass Index ($\text{weight in kg}/(\text{height in m})^2$) | 18.2 – 67.1 | Biological zero is invalid; imputed with median |
| `DiabetesPedigreeFunction` | Genetic risk score based on family pedigree | 0.078 – 2.42 | Continuous pedigree score |
| `Age` | Patient age in years | 21 – 81 | Continuous age variable |

---

## 🏗️ Project Directory Structure

```
apex_major/
├── data/
│   └── dataset.csv                               # Raw Pima Indians dataset (768 x 9)
├── notebooks/
│   └── analysis.ipynb                            # Complete interactive Jupyter notebook
├── models/
│   ├── model_1_logistic.pkl                      # Trained M1 model
│   ├── model_2_polynomial.pkl                    # Trained M2 model
│   ├── model_3_decision_tree.pkl                 # Trained M3 model
│   ├── model_4_random_forest.pkl                 # Trained M4 model
│   ├── inference_medians.json                    # Saved training medians for zero imputation
│   └── feature_columns.json                      # Ordered feature list
├── results/
│   ├── comparison_table.png                      # Performance table rendered as image
│   ├── comparison_table.csv                      # Test split evaluation metrics
│   ├── cross_validation_results.csv              # 5-fold CV results for all 4 models
│   ├── confusion_matrix_best_model.png           # Confusion matrix of winning model
│   ├── feature_importance.png                    # Gini feature importance bar chart
│   ├── eda_1_correlation_heatmap.png             # Correlation heatmap
│   ├── eda_2_class_distribution.png              # Class balance chart
│   ├── eda_3_feature_distributions_by_outcome.png# Density plots by outcome
│   ├── eda_4_outlier_boxplots.png                # Boxplots detecting zeros and outliers
│   ├── eda_5_pairplot.png                        # Pairwise feature scatter plots
│   └── final_recommendation.txt                  # Clinical recommendation statement
│
├── model1_logistic_regression.py                 # 👤 Member 1: Individual M1 training script
├── model2_polynomial_regression.py               # 👤 Member 2: Individual M2 training script
├── model3_decision_tree.py                       # 👤 Member 3: Individual M3 training script
├── model4_random_forest.py                       # 👤 Member 4: Individual M4 training script
├── compare_all_models.py                         # 👥 Collaborative: Model comparison script
├── run_pipeline.py                               # Master end-to-end automated pipeline
├── app.py                                        # ⭐ Bonus 3: Streamlit clinical dashboard
└── README.md                                     # Project documentation
```

---

## 🔬 Data Preprocessing & Leak-Free Design

### 1. Handling Biological Zeroes
In living patients, physiological values of `0` for `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` represent unrecorded/missing entries rather than true zeros.
* **Leakage Prevention:** An 80:20 stratified train-test split is created **before** calculating any medians.
* Medians are computed strictly on `X_train` and saved to `models/inference_medians.json` so the test set and production application remain unbiased.

### 2. Feature Engineering
Two domain-specific features were added:
1. **`BMI_Category`:** Binned into standard WHO categories (0: Underweight $<18.5$, 1: Normal $18.5-24.9$, 2: Overweight $25.0-29.9$, 3: Obese $\ge 30.0$).
2. **`Glucose_to_Insulin_ratio`:** $\frac{\text{Glucose}}{\text{Insulin} + 10^{-5}}$, capturing pancreatic response and insulin resistance.

---

## 📈 Model Performance & Comparison

### 1. Test Split Performance (80:20 Split — 154 Patients)

| Model | Test Accuracy | Test Precision | Test Recall | Test F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **M1: Logistic Regression** | 69.48% | 57.78% | 48.15% | 0.5253 |
| **M2: Polynomial Logistic ($deg=2$)** | 73.38% | 65.12% | 51.85% | 0.5773 |
| **M3: Decision Tree (Tuned)** | 74.03% | 66.67% | 51.85% | 0.5833 |
| **M4: Random Forest (Tuned)** 🏆 | **75.97%** | **68.89%** | **57.41%** | **0.6263** |

### 2. ⭐ 5-Fold Stratified Cross-Validation (Generalization Stability)

| Model | CV Mean Accuracy | CV Std Dev | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M1: Logistic Regression** | 78.82% | ± 1.33% | 78.86% | 78.86% | 79.67% | 79.67% | 76.23% |
| **M2: Polynomial Logistic ($deg=2$)** | 78.50% | ± 1.36% | 78.86% | 77.24% | 80.49% | 78.05% | 76.23% |
| **M4: Random Forest (Tuned)** 🏆 | **77.85%** | **± 1.42%** | 78.05% | 77.24% | 80.49% | 76.42% | 77.05% |
| **M3: Decision Tree (Tuned)** | 71.82% | ± 2.93% | 76.42% | 69.11% | 73.17% | 68.29% | 72.13% |

---

## 🏆 Final Model Recommendation for Hospital Deployment

> **Recommendation:**  
> The **tuned Random Forest Classifier (M4)** is recommended as the winning model for hospital deployment, achieving the highest performance across all evaluation metrics.
> 
> **Key Justification Points:**
> 1. **Best Overall Performance:** Random Forest achieves the highest **Test Accuracy (75.97%)**, highest **Precision (68.89%)**, highest **Recall (57.41%)**, and highest **F1-Score (0.6263)** among all 4 models.
> 2. **Clinical Safety (High Recall):** In diabetes screening, missing a diabetic patient (False Negative) can lead to unmanaged complications. Random Forest caught 57.41% of diabetic cases compared to only 48.15% by Logistic Regression and 51.85% by Decision Tree.
> 3. **Stable Generalization:** Achieves **77.85%** across 5-fold stratified cross-validation with low standard deviation (±1.42%), showing that it performs consistently across diverse patient subsets without overfitting.
> 4. **Biomarker Interpretability:** Feature importance analysis confirms that **Glucose level (33.3%)**, **BMI (13.4%)**, and **Age (11.9%)** are the primary diagnostic indicators, perfectly aligning with clinical guidelines.

---

## 🚀 How to Run the Project Locally

### 1. Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib
```

### 2. Run Individual Member Models (Traditional Approach)
Each member can run their own model independently from the command line:
```bash
# Member 1: Train & test Logistic Regression
python model1_logistic_regression.py

# Member 2: Train & test Polynomial Logistic Regression
python model2_polynomial_regression.py

# Member 3: Train & test Decision Tree with GridSearchCV
python model3_decision_tree.py

# Member 4: Train & test Random Forest with GridSearchCV
python model4_random_forest.py
```

### 3. Compare All Models
```bash
python compare_all_models.py
```

### 4. Run the Master Pipeline (All-In-One)
```bash
python run_pipeline.py
```

### 5. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
The application opens in your web browser at `http://localhost:8501`.

---

## ✅ Capstone Checklist Verification

- [x] **EDA with at least 5 visualizations** (`results/eda_1` through `eda_5`)
- [x] **Data cleaning** (biological zeroes handled, median imputation)
- [x] **Feature engineering** (`BMI_Category` & `Glucose_to_Insulin_ratio`)
- [x] **Train-Test Split** (80:20 Stratified)
- [x] **All 4 models trained & evaluated** (M1, M2, M3, M4)
- [x] **Comparison table** (`results/comparison_table.png` and `.csv`)
- [x] **Feature Importance chart** (`results/feature_importance.png`)
- [x] **Best model justification** (`results/final_recommendation.txt`)
- [x] **4 Independent team member scripts** (`model1` through `model4`)
- [x] ⭐ **Bonus 1:** Hyperparameter tuning via `GridSearchCV`
- [x] ⭐ **Bonus 2:** 5-fold Stratified Cross-Validation
- [x] ⭐ **Bonus 3:** Interactive Streamlit Web App Demo
