import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="MediScan AI | Diabetes Clinical Diagnostic System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Industry-Grade CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .app-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 1.8rem 2.2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.8rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .app-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
        color: #ffffff;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .app-subtitle {
        font-size: 1.0rem;
        color: #94a3b8;
        margin-top: 0.4rem;
        margin-bottom: 0;
        font-weight: 500;
    }
    .hospital-badge {
        display: inline-block;
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    .card {
        background: #ffffff;
        border-radius: 14px;
        padding: 1.5rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
        margin-bottom: 1.2rem;
    }
    .risk-banner-high {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border: 1.5px solid #f87171;
        border-left: 6px solid #ef4444;
        border-radius: 14px;
        padding: 1.5rem;
        color: #991b1b;
        margin-bottom: 1.2rem;
    }
    .risk-banner-low {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 1.5px solid #4ade80;
        border-left: 6px solid #10b981;
        border-radius: 14px;
        padding: 1.5rem;
        color: #065f46;
        margin-bottom: 1.2rem;
    }
    .metric-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.4rem 0.8rem;
        font-size: 0.88rem;
        font-weight: 600;
        color: #334155;
        margin: 0.2rem;
    }
    .advice-box {
        background: #f8fafc;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #3b82f6;
        font-size: 0.92rem;
        color: #334155;
        line-height: 1.6;
        margin-top: 0.8rem;
    }
    .gender-lock {
        background: #f1f5f9;
        border: 1px dashed #cbd5e1;
        border-radius: 8px;
        padding: 0.6rem 0.9rem;
        color: #64748b;
        font-size: 0.88rem;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-top: 0.4rem;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Load configuration and models
@st.cache_resource
def load_system_assets():
    with open(os.path.join(MODELS_DIR, "inference_medians.json"), "r") as f:
        medians = json.load(f)
        
    with open(os.path.join(MODELS_DIR, "feature_columns.json"), "r") as f:
        feature_cols = json.load(f)
        
    models = {
        "Random Forest Classifier (Tuned) [Recommended]": joblib.load(os.path.join(MODELS_DIR, "model_4_random_forest.pkl")),
        "Decision Tree Classifier (Tuned)": joblib.load(os.path.join(MODELS_DIR, "model_3_decision_tree.pkl")),
        "Polynomial Logistic Regression (deg=2)": joblib.load(os.path.join(MODELS_DIR, "model_2_polynomial.pkl")),
        "Logistic Regression (Linear Baseline)": joblib.load(os.path.join(MODELS_DIR, "model_1_logistic.pkl"))
    }
    
    comp_df = pd.read_csv(os.path.join(RESULTS_DIR, "comparison_table.csv"))
    cv_df = pd.read_csv(os.path.join(RESULTS_DIR, "cross_validation_results.csv"))
    
    return medians, feature_cols, models, comp_df, cv_df

try:
    medians, feature_cols, models, comp_df, cv_df = load_system_assets()
except Exception as e:
    st.error(f"Error loading system models: {e}. Please ensure models are trained.")
    st.stop()

# Top Header
st.markdown("""
<div class="app-header">
    <div class="hospital-badge">Smart Healthcare Assistant</div>
    <div class="app-title">🏥 MediScan AI: Diabetes Risk Screening Portal</div>
    <div class="app-subtitle">A simple, reliable tool to help doctors and patients detect early signs of Type 2 Diabetes</div>
</div>
""", unsafe_allow_html=True)

# Portal Navigation: Clean Separation between Clinical Patient Use & Capstone Audit
portal_mode = st.radio(
    "Choose What You Want to Do:",
    ["🩺 Check a Patient (Diagnostic Portal)", "📊 View Model Results & Project Audit (For Evaluators)"],
    index=0,
    horizontal=True,
    help="Switch between checking an individual patient or reviewing the project's machine learning charts and accuracy scores."
)

st.write("")

# ==============================================================================
# MODE 1: PROFESSIONAL PATIENT DIAGNOSTIC PORTAL (CLEAN, NO CLUTTER)
# ==============================================================================
if portal_mode == "🩺 Check a Patient (Diagnostic Portal)":

    # Preset Patient Profiles for quick demo in viva/presentation
    st.sidebar.markdown("### ⚡ Quick Test Profiles")
    st.sidebar.caption("Click either button to automatically fill in sample patient values:")
    col_pre1, col_pre2 = st.sidebar.columns(2)
    load_healthy = col_pre1.button("🟢 Normal", help="Loads a healthy, non-diabetic patient profile")
    load_diabetic = col_pre2.button("🔴 High-Risk", help="Loads an elevated diabetic patient profile")

    # Handle preset state
    if "preset_data" not in st.session_state:
        st.session_state.preset_data = {
            "gender": "Female", "preg": 2, "gluc": 115, "bp": 72, "skin": 22,
            "ins": 85, "bmi": 24.5, "pedi": 0.35, "age": 32
        }

    if load_healthy:
        st.session_state.preset_data = {
            "gender": "Female", "preg": 0, "gluc": 85, "bp": 68, "skin": 18,
            "ins": 65, "bmi": 21.4, "pedi": 0.18, "age": 24
        }
        st.rerun()

    if load_diabetic:
        st.session_state.preset_data = {
            "gender": "Female", "preg": 5, "gluc": 178, "bp": 86, "skin": 34,
            "ins": 210, "bmi": 38.2, "pedi": 0.82, "age": 52
        }
        st.rerun()

    # Sidebar: Patient Demographics & Diagnostics
    with st.sidebar:
        st.markdown("### 📋 Patient Details")
        gender = st.radio(
            "Gender",
            ["Female", "Male"],
            index=0 if st.session_state.preset_data["gender"] == "Female" else 1,
            horizontal=True
        )

        # Smart Gender Handling: If Male, lock Pregnancies to 0
        if gender == "Male":
            pregnancies = 0
            st.markdown("""
            <div class="gender-lock">
                🔒 <strong>Pregnancies: 0</strong> (Automatically disabled for male patients)
            </div>
            """, unsafe_allow_html=True)
        else:
            pregnancies = st.number_input(
                "Pregnancies",
                min_value=0, max_value=20,
                value=int(st.session_state.preset_data["preg"]),
                step=1,
                help="Number of times pregnant (gestational history)."
            )

        age = st.number_input(
            "Age (years)",
            min_value=18, max_value=120,
            value=int(st.session_state.preset_data["age"]),
            step=1,
            help="Patient age in years."
        )

        st.markdown("---")
        st.markdown("### 🧪 Medical Tests & Vitals")
        st.caption("Enter patient lab test numbers below. If a test wasn't performed (e.g. Insulin), leaving it at 0 will automatically use the standard population average.")

        glucose = st.number_input(
            "Plasma Glucose (2h Oral Test, mg/dL)",
            min_value=0, max_value=300,
            value=int(st.session_state.preset_data["gluc"]),
            step=1,
            help="Normal fasting is <100 mg/dL, 2-hour tolerance <140 mg/dL."
        )

        bmi = st.number_input(
            "Body Mass Index (BMI, kg/m²)",
            min_value=0.0, max_value=70.0,
            value=float(st.session_state.preset_data["bmi"]),
            step=0.1, format="%.1f",
            help="Normal weight is 18.5 - 24.9 kg/m²."
        )

        blood_pressure = st.number_input(
            "Diastolic Blood Pressure (mm Hg)",
            min_value=0, max_value=180,
            value=int(st.session_state.preset_data["bp"]),
            step=1,
            help="Diastolic reading (bottom number). Normal is <80 mm Hg."
        )

        insulin = st.number_input(
            "Serum Insulin (2h post-test, μU/mL)",
            min_value=0, max_value=900,
            value=int(st.session_state.preset_data["ins"]),
            step=5,
            help="Serum insulin level. Enter 0 if test was not conducted."
        )

        skin_thickness = st.number_input(
            "Triceps Skinfold Thickness (mm)",
            min_value=0, max_value=99,
            value=int(st.session_state.preset_data["skin"]),
            step=1,
            help="Subcutaneous body fat marker. Enter 0 if unavailable."
        )

        pedigree = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.01, max_value=2.50,
            value=float(st.session_state.preset_data["pedi"]),
            step=0.01, format="%.2f",
            help="Family genetics and heritage risk score (typically 0.1 to 1.2)."
        )

        st.markdown("---")
        with st.expander("⚙️ Diagnostic Engine Settings", expanded=False):
            selected_model_name = st.selectbox(
                "Active Model Architecture:",
                list(models.keys()),
                index=0
            )
            active_model = models[selected_model_name]

    # Preprocessing and Feature Calculation
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

    patient_cleaned = {}
    imputed_fields = []
    for key, val in patient_raw.items():
        if key in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'] and val == 0:
            patient_cleaned[key] = medians[key]
            imputed_fields.append(f"{key} (defaulted to population median: {medians[key]:.1f})")
        else:
            patient_cleaned[key] = float(val)

    # WHO BMI category classification
    curr_bmi = patient_cleaned['BMI']
    if curr_bmi < 18.5:
        bmi_cat = 0
        bmi_label = "Underweight"
        bmi_color = "#3b82f6"
    elif curr_bmi < 25.0:
        bmi_cat = 1
        bmi_label = "Normal Weight"
        bmi_color = "#10b981"
    elif curr_bmi < 30.0:
        bmi_cat = 2
        bmi_label = "Overweight"
        bmi_color = "#f59e0b"
    else:
        bmi_cat = 3
        bmi_label = "Obese"
        bmi_color = "#ef4444"

    patient_cleaned['BMI_Category'] = bmi_cat
    patient_cleaned['Glucose_to_Insulin_ratio'] = patient_cleaned['Glucose'] / (patient_cleaned['Insulin'] + 1e-5)

    input_df = pd.DataFrame([patient_cleaned])[feature_cols]

    # Run Model Inference
    prediction = active_model.predict(input_df)[0]
    try:
        probabilities = active_model.predict_proba(input_df)[0]
        diabetic_risk = probabilities[1]
        healthy_confidence = probabilities[0]
    except Exception:
        diabetic_risk = float(prediction)
        healthy_confidence = 1.0 - diabetic_risk

    # Main Clinical Dashboard Presentation
    col_left, col_right = st.columns([1.15, 1.0], gap="large")

    with col_left:
        st.markdown("### 🩺 Screening Result")

        if prediction == 1:
            st.markdown(f"""
            <div class="risk-banner-high">
                <div style="font-size:0.85rem; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; color:#b91c1c;">
                    ⚠️ Screening Alert: Higher Risk Detected
                </div>
                <h2 style="margin: 0.3rem 0; font-size:1.8rem; font-weight:800; color:#991b1b;">
                    Elevated Risk of Type 2 Diabetes
                </h2>
                <div style="font-size:1.15rem; font-weight:600; margin-top:0.4rem; color:#7f1d1d;">
                    Calculated Risk Score: <strong>{diabetic_risk * 100:.1f}%</strong>
                </div>
                <p style="margin:0.5rem 0 0 0; font-size:0.95rem; color:#991b1b; line-height:1.5;">
                    The patient's blood glucose, BMI, or other vitals are noticeably higher than normal healthy ranges. We strongly recommend scheduling a confirmatory blood test (such as HbA1c or Fasting Blood Sugar) with a doctor soon.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="risk-banner-low">
                <div style="font-size:0.85rem; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; color:#047857;">
                    ✅ Good News: Normal Healthy Range
                </div>
                <h2 style="margin: 0.3rem 0; font-size:1.8rem; font-weight:800; color:#065f46;">
                    Low Risk / Non-Diabetic
                </h2>
                <div style="font-size:1.15rem; font-weight:600; margin-top:0.4rem; color:#064e3b;">
                    Healthy Likelihood: <strong>{healthy_confidence * 100:.1f}%</strong>
                </div>
                <p style="margin:0.5rem 0 0 0; font-size:0.95rem; color:#065f46; line-height:1.5;">
                    The patient's readings look healthy and are within the normal expected ranges. Continuing regular exercise, a balanced diet, and standard annual checkups is recommended.
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Visual Risk Probability Progress Bar
        st.markdown(f"**Risk Level Meter ({diabetic_risk*100:.1f}% Risk):**")
        st.progress(diabetic_risk)

        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Diabetes Risk", f"{diabetic_risk * 100:.1f}%")
        col_m2.metric("Healthy Likelihood", f"{healthy_confidence * 100:.1f}%")

        if imputed_fields:
            st.caption("ℹ️ **Note:** Since some tests were left as 0, standard population averages were safely used for: " + ", ".join(imputed_fields))

        # Actionable Clinical Recommendations in human language
        st.markdown("### 💡 Recommended Next Steps")
        if prediction == 1:
            st.markdown("""
            <div class="advice-box">
                <strong>What to do next:</strong><br>
                1. <strong>Confirm with a lab test:</strong> Get an HbA1c or Fasting Plasma Glucose test done at an authorized clinic within the next 2 weeks.<br>
                2. <strong>Dietary changes:</strong> Cut down on sugary drinks, refined carbs, and processed snacks. Focus on fresh vegetables, fiber, and whole foods.<br>
                3. <strong>Doctor consultation:</strong> Schedule an appointment with a general physician to review blood pressure, heart health, and personalized lifestyle advice.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="advice-box">
                <strong>Tips to stay healthy:</strong><br>
                1. <strong>Keep moving:</strong> Aim for at least 30 minutes of moderate activity (like brisk walking, jogging, or cycling) 5 days a week.<br>
                2. <strong>Balanced meals:</strong> Drink plenty of water and eat meals with a good mix of vegetables, lean proteins, and whole grains.<br>
                3. <strong>Yearly checkup:</strong> Get your blood sugar and vitals checked once a year during a regular health screening.
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        st.markdown("### 📊 Patient Metabolic Profile")
        
        # Summary Card
        st.markdown(f"""
        <div class="card">
            <h4 style="margin-top:0; color:#1e293b; font-size:1.1rem; border-bottom:1px solid #e2e8f0; padding-bottom:0.6rem;">
                Key Physiological Indicators
            </h4>
            <div style="margin-top:0.8rem;">
                <div class="metric-pill">👤 Gender: <strong>{gender}</strong></div>
                <div class="metric-pill">🎂 Age: <strong>{age} years</strong></div>
                <div class="metric-pill">👶 Pregnancies: <strong>{pregnancies}</strong></div>
            </div>
            <div style="margin-top:0.6rem;">
                <div class="metric-pill">🩸 Glucose: <strong>{patient_cleaned['Glucose']} mg/dL</strong> ({'Elevated' if patient_cleaned['Glucose'] >= 140 else 'Normal'})</div>
                <div class="metric-pill" style="border-color:{bmi_color};">⚖️ BMI: <strong>{patient_cleaned['BMI']:.1f} ({bmi_label})</strong></div>
                <div class="metric-pill">💓 Blood Pressure: <strong>{patient_cleaned['BloodPressure']:.0f} mm Hg</strong></div>
            </div>
            <div style="margin-top:0.6rem;">
                <div class="metric-pill">💉 Insulin: <strong>{patient_cleaned['Insulin']:.0f} μU/mL</strong></div>
                <div class="metric-pill">🧬 Pedigree Score: <strong>{patient_cleaned['DiabetesPedigreeFunction']:.2f}</strong></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Download Patient Report Option
        report_text = f"""==================================================
MEDISCAN AI - CLINICAL DIABETES RISK ASSESSMENT
==================================================
Date/Time: Current Session
Patient Gender: {gender}
Patient Age: {age} years
Pregnancies: {pregnancies}

DIAGNOSTIC MEASUREMENTS:
- Plasma Glucose: {patient_cleaned['Glucose']} mg/dL
- Diastolic Blood Pressure: {patient_cleaned['BloodPressure']} mm Hg
- Body Mass Index (BMI): {patient_cleaned['BMI']:.1f} kg/m² ({bmi_label})
- 2-Hour Serum Insulin: {patient_cleaned['Insulin']} μU/mL
- Diabetes Pedigree Function: {patient_cleaned['DiabetesPedigreeFunction']:.3f}

DIAGNOSTIC PREDICTION:
Status: {'HIGH RISK: DIABETIC' if prediction == 1 else 'LOW RISK: NON-DIABETIC'}
Estimated Diabetic Probability: {diabetic_risk * 100:.1f}%
Confidence Score: {healthy_confidence * 100:.1f}%
Model Utilized: {selected_model_name}
==================================================
This tool is for clinical screening assistance and does not replace certified physician diagnosis.
"""
        st.download_button(
            label="📥 Download Clinical Patient Summary (.txt)",
            data=report_text,
            file_name=f"MediScan_Report_{gender}_{age}yo.txt",
            mime="text/plain",
            width="stretch"
        )

# ==============================================================================
# MODE 2: PROJECT RESULTS & MODEL AUDIT (FOR PROFESSORS & EVALUATORS)
# ==============================================================================
else:
    st.markdown("""
    <div style="background:#f8fafc; padding:1.4rem; border-radius:12px; border:1px solid #cbd5e1; margin-bottom:1.5rem;">
        <h3 style="margin:0; color:#0f172a;">📊 Capstone Project Results & Model Evidence</h3>
        <p style="margin:0.4rem 0 0 0; color:#475569; font-size:0.95rem; line-height:1.5;">
            Here is the complete technical evidence from our capstone project. We trained and evaluated 4 different machine learning models 
            on the exact same 154 patients who were completely unseen during training. You can explore the accuracy comparisons, 
            the winning Random Forest confusion matrix, biomarker rankings, and our exploratory data analysis charts below.
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab_eval, tab_feat, tab_eda_view = st.tabs([
        "📈 How All 4 Models Compare",
        "🌲 What Features Matter Most? (Biomarker Importance)",
        "🔬 Exploratory Data Analysis (5 Key Charts)"
    ])

    with tab_eval:
        st.subheader("1. Test Results on 154 Unseen Patients (20% Test Split)")
        st.info("""
        **💡 How to read this table:**
        - **Test Accuracy:** How many total patients did the model diagnose correctly out of 100? **Model 4 (Random Forest) is the winner at 75.97%**.
        - **Test Recall (Most Important in Medicine):** Out of 100 people who actually have diabetes, how many did the model detect? Random Forest scored the highest (**57.41%**), meaning it misses the fewest sick patients.
        - **F1-Score:** The harmonic balance between precision and recall. Random Forest leads with **0.6263**.
        """)
        st.dataframe(
            comp_df.style.format({
                "Test Accuracy": "{:.2%}",
                "Test Precision": "{:.4f}",
                "Test Recall": "{:.4f}",
                "Test F1-Score": "{:.4f}"
            }).highlight_max(subset=["Test Accuracy", "Test Recall", "Test F1-Score"], color="#d4edda"),
            width="stretch"
        )

        st.subheader("2. 5-Fold Cross-Validation (Checking Consistency Across 5 Splits)")
        st.caption("Cross-validation splits the data 5 different ways to prove the models didn't just get lucky on one test set:")
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

        st.subheader("3. Confusion Matrix: Where Did the Winning Random Forest Model Get It Right?")
        col_cm_img, col_cm_desc = st.columns([1.1, 1.0], gap="medium")
        with col_cm_img:
            st.image(os.path.join(RESULTS_DIR, "confusion_matrix_best_model.png"), caption="Confusion Matrix on 154 Unseen Test Patients", width=440)
        with col_cm_desc:
            st.markdown("""
            **What this matrix tells us:**
            - **86 True Negatives (Top-Left):** 86 healthy patients were correctly diagnosed as healthy.
            - **31 True Positives (Bottom-Right):** 31 diabetic patients were correctly flagged as having diabetes.
            - **14 False Positives (Top-Right):** 14 healthy patients were flagged for a checkup (safe false alarm).
            - **23 False Negatives (Bottom-Left):** 23 diabetic patients were missed (this is the lowest missed rate among all models).
            
            *Overall, the Random Forest correctly diagnosed 117 out of 154 patients (75.97%).*
            """)

    with tab_feat:
        st.subheader("Which Biomarkers Drive the Diabetes Diagnosis?")
        st.info("""
        **💡 What this chart means:**
        This bar chart reveals which clinical measurements our Random Forest model relies on the most.
        - **Glucose is #1 (33.3% importance):** Blood sugar is the direct hallmark of diabetes, so it's reassuring that the algorithm naturally prioritizes it.
        - **BMI is #2 (13.4% importance):** Higher body weight and obesity directly contribute to insulin resistance.
        - **Age is #3 (11.9% importance):** Risk increases steadily as patients grow older.
        """)
        st.image(os.path.join(RESULTS_DIR, "feature_importance.png"), caption="Feature Importance Ranking (Glucose, BMI, and Age are the top 3 drivers)", width="stretch")

    with tab_eda_view:
        st.subheader("Exploratory Data Analysis: What Did We Learn from the Patient Data?")
        st.caption("Here are the 5 core charts we used to inspect patient patterns and clean the data before training:")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 1. Correlation Heatmap")
            st.caption("Shows how measurements relate to diabetes. Glucose has the strongest correlation (+0.49).")
            st.image(os.path.join(RESULTS_DIR, "eda_1_correlation_heatmap.png"), caption="Correlation Matrix Heatmap")
            
            st.markdown("#### 3. Patient Vitals Comparison")
            st.caption("Orange curves (diabetic) are clearly shifted to the right towards higher glucose, BMI, and age.")
            st.image(os.path.join(RESULTS_DIR, "eda_3_feature_distributions_by_outcome.png"), caption="Distributions by Outcome (Healthy vs Diabetic)")
            
            st.markdown("#### 5. Biomarker Pairplot")
            st.caption("Shows how pairs of biomarkers interact. Notice how high glucose and high BMI cluster together for diabetic patients.")
            st.image(os.path.join(RESULTS_DIR, "eda_5_pairplot.png"), caption="Pairplot of Key Biomarkers")
        
        with c2:
            st.markdown("#### 2. Class Balance in the Dataset")
            st.caption("65% of patients in this dataset are healthy (500) and 35% are diabetic (268).")
            st.image(os.path.join(RESULTS_DIR, "eda_2_class_distribution.png"), caption="Class Distribution: 500 Healthy vs 268 Diabetic")
            
            st.markdown("#### 4. Outlier & Zero-Value Detection")
            st.caption("Demonstrates the impossible 0 values (e.g. 0 blood pressure or 0 glucose) that we had to fix with medians.")
            st.image(os.path.join(RESULTS_DIR, "eda_4_outlier_boxplots.png"), caption="Outlier and Zero Detection Boxplots")
