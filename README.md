# 🏥 Diabetes Risk Prediction System (Healthcare Capstone Project)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App%20Live-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

* **Course Track:** Capstone Project — Theme 1: Healthcare
* **Dataset:** Pima Indians Diabetes Diagnostic Dataset (768 patient records)
* **Goal:** Train and compare 4 Machine Learning models, tune hyperparameters, and deploy an interactive clinical web dashboard.
* **GitHub Repository:** [https://github.com/gangakrishnamanikanta-dot/project-apex-medical-diabetis](https://github.com/gangakrishnamanikanta-dot/project-apex-medical-diabetis)

---

## 📌 Project Overview

Diabetes is one of the most common chronic diseases in the world, and many people live with it for years without knowing until it causes serious damage to their heart, kidneys, or vision. 

In this capstone project, our team built a complete Machine Learning diagnostic system to help doctors screen patients early. Using routine clinical measurements (glucose levels, blood pressure, insulin, BMI, age, and family history), our system estimates whether a patient is at risk of diabetes.

As required by the project guidelines, we implemented, evaluated, and compared **4 different machine learning models**:
1. **Model 1: Logistic Regression** — A simple, fast linear baseline.
2. **Model 2: Polynomial Logistic Regression (Degree = 2)** — Captures non-linear interactions like Glucose × BMI.
3. **Model 3: Decision Tree Classifier** — A flowchart-like model tuned with 5-fold GridSearchCV to prevent overfitting.
4. **Model 4: Random Forest Classifier** — An ensemble of 150 diverse trees with GridSearchCV, which emerged as our **#1 winning model**.

We also completed all **3 bonus challenges**: hyperparameter tuning (`GridSearchCV`), 5-fold cross-validation, and an interactive **Streamlit web application** with smart gender options.

---

## 👥 4-Member Team Roles & Work Breakdown

To divide the work fairly among 4 team members, each person took complete ownership of one model and has their own independent Python script:

| Member | Project Role | Dedicated Script | What They Did |
| :--- | :--- | :--- | :--- |
| **Member 1** | **Data Engineering & Baseline Lead** | `model1_logistic_regression.py` | Inspected raw data, handled impossible zero values, built the baseline Logistic Regression model with Standard Scaling, and analyzed feature weights. |
| **Member 2** | **Feature Pipeline & Polynomial Lead** | `model2_polynomial_regression.py` | Engineered WHO BMI categories and Glucose-to-Insulin ratios, built polynomial degree-2 interaction terms, and tuned L2 regularization. |
| **Member 3** | **Tree Modeling & Pruning Lead** | `model3_decision_tree.py` | Built the Decision Tree model, analyzed tree pruning, ran 5-fold GridSearchCV to stop overfitting, and extracted decision rules. |
| **Member 4** | **Ensemble, Evaluation & Deployment Lead** | `model4_random_forest.py` & `app.py` | Built the 150-tree Random Forest ensemble, generated Gini feature importance rankings, ran 5-fold cross-validation, and developed the Streamlit web dashboard. |

---

## 📊 Dataset Overview

* **Source:** National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK) / Kaggle
* **Patients:** 768 female patient observations of Pima Indian heritage (aged 21+)
* **Outcome Target:** `0` = Healthy / Non-Diabetic (500 patients, 65.1%), `1` = Diabetic (268 patients, 34.9%)

| Feature Name | What It Measures | Range in Data | Preprocessing Notes |
| :--- | :--- | :--- | :--- |
| `Pregnancies` | Number of times pregnant | 0 – 17 | Valid count variable (automatically set to 0 for males) |
| `Glucose` | Plasma glucose at 2 hours in oral test (mg/dL) | 44 – 199 | Zero is medically impossible; imputed with median |
| `BloodPressure` | Diastolic blood pressure (mm Hg) | 24 – 122 | Zero is medically impossible; imputed with median |
| `SkinThickness` | Triceps skinfold thickness (mm) | 7 – 99 | Zero is medically impossible; imputed with median |
| `Insulin` | 2-hour serum insulin ($\mu\text{U/mL}$) | 14 – 846 | Zero is medically impossible; imputed with median |
| `BMI` | Body Mass Index ($\text{weight in kg}/(\text{height in m})^2$) | 18.2 – 67.1 | Zero is medically impossible; imputed with median |
| `DiabetesPedigreeFunction` | Genetic risk score based on family pedigree | 0.078 – 2.42 | Higher numbers mean stronger family history |
| `Age` | Patient age in years | 21 – 81 | Older age bands have higher diabetes risk |

---

## 📁 Repository Structure

```
apex_major/
├── data/
│   └── dataset.csv                               # Raw Pima Indians dataset (768 rows x 9 columns)
├── notebooks/
│   └── analysis.ipynb                            # Complete Jupyter notebook with all 15 executed cells and plots
├── models/
│   ├── model_1_logistic.pkl                      # Trained M1 model binary
│   ├── model_2_polynomial.pkl                    # Trained M2 model binary
│   ├── model_3_decision_tree.pkl                 # Trained M3 model binary
│   ├── model_4_random_forest.pkl                 # Trained M4 model binary (Winning Model)
│   ├── inference_medians.json                    # Saved training medians for missing data handling
│   └── feature_columns.json                      # Ordered list of features
├── results/
│   ├── comparison_table.png                      # Rendered 4-model performance comparison table
│   ├── comparison_table.csv                      # Test metrics across all models
│   ├── cross_validation_results.csv              # 5-fold cross-validation scores for all models
│   ├── confusion_matrix_best_model.png           # Confusion matrix of the winning model
│   ├── feature_importance.png                    # Gini feature importance bar chart
│   ├── eda_1_correlation_heatmap.png             # Correlation matrix heatmap
│   ├── eda_2_class_distribution.png              # Class balance chart (65% healthy vs 35% diabetic)
│   ├── eda_3_feature_distributions_by_outcome.png# Density distributions split by outcome
│   ├── eda_4_outlier_boxplots.png                # Boxplots detecting zero-anomalies and outliers
│   ├── eda_5_pairplot.png                        # Pairwise scatter plot grid
│   └── final_recommendation.txt                  # Clinical recommendation summary text
│
├── model1_logistic_regression.py                 # Member 1: M1 training script
├── model2_polynomial_regression.py               # Member 2: M2 training script
├── model3_decision_tree.py                       # Member 3: M3 training script
├── model4_random_forest.py                       # Member 4: M4 training script
├── compare_all_models.py                         # Team script: Evaluates all 4 models side by side
├── run_pipeline.py                               # Master all-in-one automation script
├── app.py                                        # Interactive Streamlit clinical web app
├── presentation.pdf                              # 6-slide presentation deck for viva/evaluation
├── HOW_TO_RUN_AND_TEST.md                        # Step-by-step test guide and viva cheat sheet
└── README.md                                     # Main documentation
```

---

## 🔬 How We Cleaned the Data (No Data Leakage)

1. **Fixing Medically Impossible Zeros:**  
   A living person cannot have zero blood glucose, zero blood pressure, or zero BMI. In this dataset, `0` was used for missing measurements. We replaced zeros in those 5 columns with `NaN`.
2. **Preventing Data Leakage:**  
   We performed our **80:20 stratified train-test split FIRST**. Then, we calculated replacement medians using **only the 80% training data**, saving them to `models/inference_medians.json`. The 20% test patients remained 100% unseen until final testing.
3. **Clinical Feature Engineering:**  
   We added two domain features:
   * **`BMI_Category`:** Binned into standard WHO categories (Underweight, Normal, Overweight, Obese).
   * **`Glucose_to_Insulin_ratio`:** $\frac{\text{Glucose}}{\text{Insulin} + 10^{-5}}$, an indicator of insulin resistance.

---

## 📈 Final Model Performance & Comparison

All 4 models were evaluated under the **exact same rules** on the **same 154 unseen test patients (20% split)**:

| Model Architecture | Test Accuracy | Precision | Recall (Medical Safety) | F1-Score | 5-Fold Cross-Validation | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **M1: Logistic Regression** | 69.48% | 0.5778 | 0.4815 | 0.5253 | 78.82% (±1.33%) | Linear Baseline |
| **M2: Polynomial Logistic ($deg=2$)** | 73.38% | 0.6512 | 0.5185 | 0.5773 | 78.50% (±1.36%) | Non-Linear Fit |
| **M3: Decision Tree (Tuned)** | 74.03% | 0.6667 | 0.5185 | 0.5833 | 71.82% (±2.93%) | Pruned Tree |
| **M4: Random Forest (Tuned)** 🏆 | **75.97%** | **0.6889** | **0.5741** | **0.6263** | **77.85% (±1.42%)** | **UNDISPUTED WINNER** |

---

## 🏆 Why Random Forest Was Selected for Hospital Deployment

We selected the **tuned Random Forest Classifier (M4)** as the recommended model for hospital use:

1. **Highest Overall Performance:** It achieves the highest **Test Accuracy (75.97%)**, highest **Precision (68.89%)**, highest **Recall (57.41%)**, and highest **F1-Score (0.6263)** among all 4 models.
2. **Clinical Safety (High Recall):** In healthcare, the worst mistake is sending a diabetic patient home undiagnosed (a False Negative). Random Forest caught 57.4% of diabetic patients, significantly outperforming Logistic Regression (48.1%) and Decision Tree (51.8%).
3. **Cross-Validation Stability:** While a single Decision Tree's score fluctuated widely across folds (dropping as low as 68.2%), Random Forest maintained a stable **77.85% (±1.42%)** across all 5 folds without overfitting.
4. **Physician Transparency:** Feature importance analysis proves that **Glucose (33.3%)**, **BMI (13.4%)**, and **Age (11.9%)** drive the decisions, which aligns directly with standard clinical diagnostic criteria.

---

## 🚀 How to Run and Test the Project

For a full step-by-step testing guide and sample patient values, see **[HOW_TO_RUN_AND_TEST.md](HOW_TO_RUN_AND_TEST.md)**.

### 1. Install Requirements
```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib
```

### 2. Run Any Member's Model Independently
```bash
python model1_logistic_regression.py   # Member 1 (M1)
python model2_polynomial_regression.py # Member 2 (M2)
python model3_decision_tree.py         # Member 3 (M3)
python model4_random_forest.py         # Member 4 (M4)
```

### 3. Run the Comparison Script
```bash
python compare_all_models.py
```

### 4. Launch the Interactive Web Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`. You can test sample patients, toggle Male/Female options, and view technical audit tabs.

---

## ✅ Capstone Submission Checklist

- [x] **EDA with at least 5 visualizations** (`results/eda_1` through `eda_5`)
- [x] **Data cleaning** (biological zeroes handled, median imputation)
- [x] **Feature engineering** (`BMI_Category` & `Glucose_to_Insulin_ratio`)
- [x] **Train-Test Split** (80:20 Stratified)
- [x] **All 4 models trained & evaluated** (M1, M2, M3, M4)
- [x] **Comparison table** (`results/comparison_table.png` and `.csv`)
- [x] **Feature Importance chart** (`results/feature_importance.png`)
- [x] **Best model justification** (`results/final_recommendation.txt`)
- [x] **4 Independent team member scripts** (`model1` through `model4`)
- [x] **GitHub Repository:** [https://github.com/gangakrishnamanikanta-dot/project-apex-medical-diabetis](https://github.com/gangakrishnamanikanta-dot/project-apex-medical-diabetis)
- [x] **Presentation slide deck:** `presentation.pdf`
- [x] ⭐ **Bonus 1:** Hyperparameter tuning via `GridSearchCV`
- [x] ⭐ **Bonus 2:** 5-fold Stratified Cross-Validation
- [x] ⭐ **Bonus 3:** Interactive Streamlit Clinical Web App Demo
