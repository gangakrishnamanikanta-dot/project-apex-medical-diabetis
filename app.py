import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Diabetes Risk Prediction System | Apex Capstone",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3d59;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid #17a2b8;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    .risk-high {
        background: linear-gradient(135deg, #ff4b4b15, #ff4b4b30);
        border: 2px solid #ff4b4b;
        border-radius: 10px;
        padding: 1.2rem;
        color: #990000;
        font-weight: 600;
    }
    .risk-low {
        background: linear-gradient(135deg, #28a74515, #28a74530);
        border: 2px solid #28a745;
        border-radius: 10px;
        padding: 1.2rem;
        color: #155724;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Load configuration, medians and models
@st.cache_resource
def load_assets():
    medians_path = os.path.join(MODELS_DIR, "inference_medians.json")
    with open(medians_path, "r") as f:
        medians = json.load(f)
        
    features_path = os.path.join(MODELS_DIR, "feature_columns.json")
    with open(features_path, "r") as f:
        feature_cols = json.load(f)
        
    models = {
        "Random Forest (Tuned) [Recommended]": joblib.load(os.path.join(MODELS_DIR, "model_4_random_forest.pkl")),
        "Decision Tree (Tuned)": joblib.load(os.path.join(MODELS_DIR, "model_3_decision_tree.pkl")),
        "Logistic Regression": joblib.load(os.path.join(MODELS_DIR, "model_1_logistic.pkl")),
        "Polynomial Logistic (deg=2)": joblib.load(os.path.join(MODELS_DIR, "model_2_polynomial.pkl"))
    }
    
    comp_df = pd.read_csv(os.path.join(RESULTS_DIR, "comparison_table.csv"))
    cv_df = pd.read_csv(os.path.join(RESULTS_DIR, "cross_validation_results.csv"))
    
    return medians, feature_cols, models, comp_df, cv_df

try:
    medians, feature_cols, models, comp_df, cv_df = load_assets()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}. Please ensure `run_pipeline.py` has completed.")
    st.stop()

# Header
st.markdown('<div class="main-header">🏥 Healthcare ML: Diabetes Risk Diagnostic System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Trained on the Pima Indians Diagnostic Dataset — 4-Model Comparative Analysis (Apex Capstone)</div>', unsafe_allow_html=True)

# Layout: Sidebar for Configuration & Inputs
with st.sidebar:
    st.header("⚙️ Model Configuration")
    selected_model_name = st.selectbox(
        "Select Active ML Model:",
        list(models.keys()),
        index=0,
        help="Select among the 4 trained and evaluated models."
    )
    active_model = models[selected_model_name]
    
    st.markdown("---")
    st.header("📋 Clinical Inputs")
    st.caption("Enter patient measurements below. Zeroes in biological fields will automatically fall back to training medians.")
    
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=2, step=1)
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=125, step=1,
                              help="Plasma glucose concentration at 2 hours in an oral glucose tolerance test.")
    blood_pressure = st.number_input("Diastolic Blood Pressure (mm Hg)", min_value=0, max_value=180, value=72, step=1)
    skin_thickness = st.number_input("Triceps Skinfold Thickness (mm)", min_value=0, max_value=99, value=23, step=1)
    insulin = st.number_input("2-Hour Serum Insulin (mu U/ml)", min_value=0, max_value=900, value=90, step=1)
    bmi = st.number_input("Body Mass Index (BMI kg/m²)", min_value=0.0, max_value=70.0, value=32.4, step=0.1, format="%.1f")
    pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.01, max_value=2.50, value=0.45, step=0.01, format="%.3f",
                               help="Genetic diabetes risk score based on family history.")
    age = st.number_input("Patient Age (years)", min_value=18, max_value=120, value=38, step=1)

# Preprocessing & Feature Engineering
patient_raw = {
    'Pregnancies': pregnancies,
    'Glucose': glucose,
    'BloodPressure': blood_pressure,
    'SkinThickness': skin_thickness,
    'Insulin': insulin,
    'BMI': bmi,
    'DiabetesPedigreeFunction': pedigree,
    'Age': age
}

# Replace biological zeroes with training medians (prevention of train-serve skew)
patient_cleaned = {}
imputed_flags = []
for key, val in patient_raw.items():
    if key in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'] and val == 0:
        patient_cleaned[key] = medians[key]
        imputed_flags.append(f"{key} (substituted median: {medians[key]:.1f})")
    else:
        patient_cleaned[key] = float(val)

# BMI Categorization
calc_bmi = patient_cleaned['BMI']
if calc_bmi < 18.5:
    bmi_cat = 0
    bmi_label = "Underweight (<18.5)"
elif calc_bmi < 25.0:
    bmi_cat = 1
    bmi_label = "Normal weight (18.5 - 24.9)"
elif calc_bmi < 30.0:
    bmi_cat = 2
    bmi_label = "Overweight (25.0 - 29.9)"
else:
    bmi_cat = 3
    bmi_label = "Obese (≥30.0)"

patient_cleaned['BMI_Category'] = bmi_cat

# Glucose to Insulin Ratio
patient_cleaned['Glucose_to_Insulin_ratio'] = patient_cleaned['Glucose'] / (patient_cleaned['Insulin'] + 1e-5)

# Build DataFrame in exact training order
input_df = pd.DataFrame([patient_cleaned])[feature_cols]

# Inference
prediction = active_model.predict(input_df)[0]
try:
    probabilities = active_model.predict_proba(input_df)[0]
    diabetic_prob = probabilities[1]
    non_diabetic_prob = probabilities[0]
except Exception:
    diabetic_prob = float(prediction)
    non_diabetic_prob = 1.0 - diabetic_prob

# Main UI Columns: Results & Clinical Insights
col_result, col_insights = st.columns([1.1, 1.2])

with col_result:
    st.subheader("🎯 Diagnostic Prediction")
    
    if prediction == 1:
        st.markdown(f"""
        <div class="risk-high">
            <h3 style="margin:0; color:#c00;">⚠️ HIGH RISK: DIABETIC (Positive)</h3>
            <p style="margin-top:8px; margin-bottom:4px; font-size:1.1rem;">
                Estimated Probability: <strong>{diabetic_prob*100:.1f}%</strong>
            </p>
            <p style="margin:0; font-size:0.95rem;">Recommendation: Immediate clinical follow-up and HbA1c screening advised.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="risk-low">
            <h3 style="margin:0; color:#007a22;">✅ LOW RISK: NON-DIABETIC (Negative)</h3>
            <p style="margin-top:8px; margin-bottom:4px; font-size:1.1rem;">
                Estimated Confidence: <strong>{non_diabetic_prob*100:.1f}%</strong>
            </p>
            <p style="margin:0; font-size:0.95rem;">Patient vitals and metabolic indicators align with normal parameters.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.write("")
    st.markdown("**Risk Probability Breakdown:**")
    st.progress(diabetic_prob)
    col_p1, col_p2 = st.columns(2)
    col_p1.metric("Diabetic Risk", f"{diabetic_prob*100:.1f}%")
    col_p2.metric("Non-Diabetic Confidence", f"{non_diabetic_prob*100:.1f}%")
    
    if imputed_flags:
        st.info("ℹ️ **Automated Imputation Notice:**\n" + "\n".join([f"- {item}" for item in imputed_flags]))

with col_insights:
    st.subheader("📊 Patient Feature Summary")
    st.markdown(f"""
    <div class="metric-card">
        <strong>Metabolic & Physical Metrics:</strong><br>
        • <strong>Glucose Level:</strong> {patient_cleaned['Glucose']} mg/dL {'(Elevated)' if patient_cleaned['Glucose'] >= 140 else '(Normal)'}<br>
        • <strong>BMI:</strong> {patient_cleaned['BMI']:.1f} kg/m² — <em>{bmi_label}</em><br>
        • <strong>Blood Pressure:</strong> {patient_cleaned['BloodPressure']:.0f} mm Hg<br>
        • <strong>Insulin:</strong> {patient_cleaned['Insulin']:.0f} μU/mL<br>
        • <strong>Glucose/Insulin Ratio:</strong> {patient_cleaned['Glucose_to_Insulin_ratio']:.2f}<br>
        • <strong>Age / Pedigree Score:</strong> {patient_cleaned['Age']:.0f} yrs / {patient_cleaned['DiabetesPedigreeFunction']:.3f}
    </div>
    """, unsafe_allow_html=True)
    
    st.caption("Active Model: **" + selected_model_name + "**")

st.markdown("---")

# Model Comparison & Architecture Tabs
tab_comp, tab_import, tab_eda = st.tabs([
    "📈 4-Model Comparison & Benchmarks",
    "🌲 Feature Importance Analysis",
    "🔬 Exploratory Visualizations"
])

with tab_comp:
    st.subheader("Model Performance on 80:20 Test Split")
    st.dataframe(
        comp_df.style.format({
            "Test Accuracy": "{:.2%}",
            "Test Precision": "{:.2%}",
            "Test Recall": "{:.2%}",
            "Test F1-Score": "{:.2%}"
        }).highlight_max(subset=["Test Accuracy", "Test F1-Score"], color="#d4edda"),
        width="stretch"
    )
    
    st.subheader("5-Fold Cross-Validation Generalization Scores")
    st.dataframe(
        cv_df.style.format({
            "CV Mean Accuracy": "{:.2%}",
            "CV Std Dev": "{:.2%}",
            "Fold 1": "{:.2%}",
            "Fold 2": "{:.2%}",
            "Fold 3": "{:.2%}",
            "Fold 4": "{:.2%}",
            "Fold 5": "{:.2%}"
        }).highlight_max(subset=["CV Mean Accuracy"], color="#d4edda"),
        width="stretch"
    )
    
    st.image(os.path.join(RESULTS_DIR, "confusion_matrix_best_model.png"), caption="Confusion Matrix: Best Performing Model", width=420)

with tab_import:
    st.subheader("Key Predictive Biomarkers")
    st.write("Feature importances derived from the tuned Random Forest model show which clinical metrics drive diabetes classification most strongly:")
    st.image(os.path.join(RESULTS_DIR, "feature_importance.png"), caption="Feature Importance Ranking (Gini Score)", width="stretch")

with tab_eda:
    st.subheader("Project Exploratory Visualizations")
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.image(os.path.join(RESULTS_DIR, "eda_1_correlation_heatmap.png"), caption="EDA 1: Feature Correlation Heatmap")
        st.image(os.path.join(RESULTS_DIR, "eda_3_feature_distributions_by_outcome.png"), caption="EDA 3: Feature Distributions by Outcome")
    with col_e2:
        st.image(os.path.join(RESULTS_DIR, "eda_2_class_distribution.png"), caption="EDA 2: Outcome Class Balance")
        st.image(os.path.join(RESULTS_DIR, "eda_4_outlier_boxplots.png"), caption="EDA 4: Diagnostic Metric Boxplots")
