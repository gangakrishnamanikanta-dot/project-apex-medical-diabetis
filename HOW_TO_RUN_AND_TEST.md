# 📖 How to Run & Test the Project (Complete Guide in Plain English)

Welcome to our Diabetes Prediction Capstone project! This guide explains how to install, run, test, and present every part of this project in simple, everyday human language. 

Whether you are testing the code on your own laptop, presenting to an examiner, or doing a team viva, this guide tells you exactly what to type, what to click, and what to say.

---

## ⚡ Quick Start: Test Everything in 2 Minutes

If you just want to verify that everything works right now, open your terminal (PowerShell or Command Prompt) in the project folder and run these two commands:

### 1. View all 4 models compared in the terminal
```powershell
python compare_all_models.py
```
* **What you will see:** A clean comparison table showing accuracy, precision, recall, and F1-score for all 4 models.
* **The Winner:** You will clearly see **Model 4 (Random Forest) in first place with 75.97% test accuracy and the highest recall (57.41%)**.

### 2. Open the interactive hospital dashboard
```powershell
streamlit run app.py
```
* **What happens:** Your default web browser will pop open automatically at `http://localhost:8501`.
* You will see our clean, professional clinical dashboard where you can test sample patients or view the project evaluation charts.

---

## 💻 Prerequisites & Setup (Only takes 1 minute)

Make sure you have Python 3.10+ installed. If you haven't installed the required libraries yet, run this one line in your terminal:

```powershell
pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib
```

That's it! No complex database or cloud setup is needed. Everything runs locally on your machine.

---

## 🖥️ How to Test the Web Dashboard (`app.py`)

Run:
```powershell
streamlit run app.py
```

At the top of the webpage, you will see two clear modes:
1. **🩺 Check a Patient (Diagnostic Portal):** Designed for everyday clinic use.
2. **📊 View Model Results & Project Audit:** Designed specifically for your project evaluator/professor.

Here are the 5 test cases you can try:

### 🧪 Test 1: Testing a Healthy Patient (Low Risk)
1. On the left sidebar, under **⚡ Quick Test Profiles**, click the **"🟢 Normal"** button.
2. Notice how all the boxes automatically fill in with healthy, normal numbers:
   * **Glucose:** 85 mg/dL (Normal fasting blood sugar)
   * **BMI:** 21.4 kg/m² (Normal healthy weight)
   * **Age:** 24 years old
3. Look at the main screen:
   * A bright green card appears saying: **✅ Good News: Normal Healthy Range (Low Risk / Non-Diabetic)**.
   * The risk meter stays very low (~2% risk, ~98% healthy confidence).
   * Friendly preventive tips appear at the bottom advising balanced meals and annual checkups.

### 🧪 Test 2: Testing an Elevated Patient (High Risk)
1. In the sidebar, click the **"🔴 High-Risk"** button.
2. The boxes will automatically change to high diabetic numbers:
   * **Glucose:** 178 mg/dL (High blood sugar)
   * **BMI:** 38.2 kg/m² (Obese category)
   * **Age:** 52 years old
3. Look at the main screen:
   * An urgent red card appears saying: **⚠️ Elevated Risk of Type 2 Diabetes**.
   * The risk meter jumps up to **~87% Risk**.
   * Clear medical action steps appear: booking an HbA1c lab test within 14 days and consulting a doctor.

### 🧪 Test 3: Testing the Smart Gender Selection (Male vs Female)
1. In the sidebar under **Patient Details**, change Gender from **Female** to **Male**.
2. **Watch what happens:** The "Pregnancies" input field disappears immediately and is replaced with a locked grey badge:  
   `🔒 Pregnancies: 0 (Automatically disabled for male patients)`.
3. Switch back to **Female**, and the pregnancy input box instantly reappears.
4. *Why this matters:* This solves a common real-world flaw where clinical tools accidentally ask male patients how many times they've been pregnant.

### 🧪 Test 4: Downloading a Clinical Patient Report
1. On the right side of the screen, click the button: **"📥 Download Clinical Patient Summary (.txt)"**.
2. A clean, formatted text file will be saved to your computer containing the patient's vitals, calculated risk percentage, and doctor recommendations ready for their medical file.

### 🧪 Test 5: The Examiner / Evaluator Mode
1. At the very top of the page, switch the radio button to:  
   **"📊 View Model Results & Project Audit (For Evaluators)"**.
2. This reveals the complete technical evidence for your project:
   * **Tab 1 (Model Comparison):** Shows the 4-model accuracy table, 5-fold cross-validation table, and the Confusion Matrix (showing 86 healthy and 31 diabetic patients correctly diagnosed).
   * **Tab 2 (Biomarker Importance):** Shows the bar chart proving Blood Glucose is the #1 factor (33.3%), followed by BMI (13.4%) and Age (11.9%).
   * **Tab 3 (Exploratory Data Analysis):** Displays all 5 core charts (correlation heatmap, class balance, vitals distributions, zero-detection boxplots, and pairplot).

---

## 👥 How Each Team Member Runs Their Script

In our project, the work is cleanly divided so that each of the 4 team members has their own independent script to talk about:

### 👤 Member 1: Model 1 — Logistic Regression (Linear Baseline)
```powershell
python model1_logistic_regression.py
```
* **What Member 1 explains:**  
  *"I handled the raw data cleaning, replaced impossible zeros with median values, applied standard feature scaling, and trained the linear Logistic Regression baseline. It gave us 69.48% test accuracy, which showed us that diabetes diagnosis requires non-linear models to capture complex biological interactions."*

### 👤 Member 2: Model 2 — Polynomial Logistic Regression
```powershell
python model2_polynomial_regression.py
```
* **What Member 2 explains:**  
  *"I engineered non-linear features by expanding the 10 inputs into 65 polynomial interaction terms (like Glucose multiplied by BMI). This improved test accuracy from 69.48% to 73.38%, proving that curved relationships exist between body mass and blood sugar."*

### 👤 Member 3: Model 3 — Decision Tree Classifier (Tuned)
```powershell
python model3_decision_tree.py
```
* **What Member 3 explains:**  
  *"I built our first tree-based classifier. Because deep decision trees tend to memorize training noise (overfitting), I used 5-fold GridSearchCV to prune the tree to depth 8. It reached 74.03% test accuracy and gave us human-readable if-then decision rules."*

### 👤 Member 4: Model 4 — Random Forest Classifier (The Winner 🏆) & Web App
```powershell
python model4_random_forest.py
```
* **What Member 4 explains:**  
  *"I built our ensemble model combining 150 diverse decision trees with random feature sub-sampling. It achieved the highest test accuracy (75.97%), highest precision (68.89%), and highest recall (57.41%), making it our clear winner. I also built the interactive Streamlit clinical web app for real-time doctor use."*

---

## 📊 Understanding the Numbers in Plain English

Examiners love asking what these evaluation metrics actually mean in the real world:

| Metric | Simple Everyday Definition | What Our Winning Model Scored | Why It Matters |
| :--- | :--- | :---: | :--- |
| **Accuracy** | Out of 100 random patients, how many did we diagnose correctly? | **75.97%** | High overall correctness across both healthy and sick patients. |
| **Precision** | When the model flags a patient as diabetic, how often is it right? | **68.89%** | Reduces false alarms so healthy people don't panic unnecessarily. |
| **Recall (Medical Safety)** | Out of 100 people who *actually* have diabetes, how many did we catch? | **57.41%** | **The most important medical metric.** A high recall means fewer sick people get sent home undiagnosed. Random Forest beat all other models here. |
| **F1-Score** | The fair balance between Precision and Recall. | **0.6263** | Prevents a model from cheating by just guessing everyone is healthy. |
| **5-Fold Cross-Validation** | Testing the model on 5 different splits to make sure it wasn't just lucky. | **77.85% (+/- 1.42%)** | Proves the model is stable and dependable on new hospital data. |

---

## 🎯 Viva & Oral Exam Cheat Sheet (10 Common Questions & Simple Human Answers)

### Q1: "Why did you select Random Forest as your final model instead of Logistic Regression?"
> **Human Answer:**  
> *"Because diabetes risk is not a straight line. High blood sugar is especially dangerous when a person is also older and has a high BMI. Logistic Regression treats everything linearly and only got 69.48% accuracy. Random Forest combines 150 different decision trees, catches those complex non-linear combinations, and scored the highest across every single test metric: 75.97% accuracy, 57.41% recall, and 0.6263 F1-score."*

### Q2: "Why did you replace 0s in Glucose, Blood Pressure, and BMI with medians instead of means?"
> **Human Answer:**  
> *"First, a living human cannot have zero blood glucose or zero blood pressure, so 0 clearly represented missing data that nurses didn't record. Second, we used medians instead of means because medical data often has extreme outliers (like a few patients with very high insulin levels). The mean gets distorted by extreme values, whereas the median stays solid and representative of typical patients."*

### Q3: "What is data leakage and how did your team prevent it?"
> **Human Answer:**  
> *"Data leakage happens when information from future test patients leaks into the training phase. If you calculate the median across the whole dataset before splitting, the model indirectly 'cheats'. We strictly prevented this: we did our 80:20 train-test split FIRST. We calculated our medians ONLY using the 80% training set. The 20% test patients remained 100% untouched until final testing."*

### Q4: "The Pima dataset only contains female patients. How does your app handle male patients?"
> **Human Answer:**  
> *"The core clinical biomarkers for diabetes—blood glucose, insulin, BMI, blood pressure, age, and genetics—work on identical biological principles for both men and women. However, men cannot have pregnancies. In our app, selecting 'Male' automatically locks pregnancies to 0 so the model never receives impossible biological inputs."*

### Q5: "How did you tune hyperparameters in Decision Tree and Random Forest?"
> **Human Answer:**  
> *"We used `GridSearchCV` with 5-fold cross-validation. Instead of guessing values, the computer tested dozens of combinations of tree depth, minimum samples per leaf, and split criteria. For example, in Decision Tree, unpruned trees overfit, but GridSearchCV found that a depth of 8 provided the best generalization."*

### Q6: "Why did you create the new features `BMI_Category` and `Glucose_to_Insulin_ratio`?"
> **Human Answer:**  
> *"Raw numbers don't always convey medical context. We binned BMI into standard World Health Organization categories (Underweight, Normal, Overweight, Obese) to help tree splits. We also created the Glucose-to-Insulin ratio because high blood sugar paired with low insulin points directly toward insulin deficiency or resistance."*

### Q7: "What is the difference between a Decision Tree and a Random Forest?"
> **Human Answer:**  
> *"A Decision Tree is like asking one single doctor for their opinion—it's fast and easy to follow, but that doctor might have personal biases or overfit to specific memories. A Random Forest is like consulting a panel of 150 diverse doctors who each look at different random clues and then vote. The majority vote smooths out errors and produces much more reliable predictions."*

### Q8: "What does the Confusion Matrix of your winning model show?"
> **Human Answer:**  
> *"On our 154 unseen test patients, our Random Forest correctly identified 86 healthy patients (True Negatives) and 31 diabetic patients (True Positives). It had 14 false alarms (False Positives) and missed 23 diabetic patients (False Negatives). In total, it got 117 out of 154 correct (75.97%)."*

### Q9: "Why didn't you use Deep Learning or Neural Networks?"
> **Human Answer:**  
> *"Deep neural networks need tens of thousands of records to train properly without overfitting. The Pima dataset has 768 rows. For tabular datasets of this size, ensemble tree methods like Random Forest consistently outperform neural networks, train in seconds, and allow physicians to see exactly which biomarkers drove the diagnosis."*

### Q10: "If this tool was used in a real hospital, what is its primary role?"
> **Human Answer:**  
> *"It is designed as an early screening support assistant, not an autonomous replacement for a physician. It flags patients during routine checkups who need confirmatory lab tests like an HbA1c test, helping doctors catch diabetes years before severe complications develop."*

---

## 📁 Summary of All Project Files

| File | What It Does In Plain Words |
| :--- | :--- |
| `app.py` | The interactive hospital web dashboard with patient inputs, presets, and charts. |
| `compare_all_models.py` | Terminal script that loads all 4 models and prints the final comparison table. |
| `run_pipeline.py` | Master script that cleans the data, trains all models, and saves results in one go. |
| `model1_logistic_regression.py` | Member 1's independent training script for the linear baseline. |
| `model2_polynomial_regression.py` | Member 2's script creating interaction features and polynomial curves. |
| `model3_decision_tree.py` | Member 3's script building the tuned, pruned decision tree. |
| `model4_random_forest.py` | Member 4's script building the 150-tree winning Random Forest. |
| `data/dataset.csv` | The cleaned Pima Indians dataset (768 patient records). |
| `models/` | The saved `.pkl` trained model files and saved median numbers. |
| `results/` | All 8 generated charts, confusion matrix, and CSV tables. |
| `presentation.pdf` | The 6-slide presentation deck matching project submission requirements. |
| `HOW_TO_RUN_AND_TEST.md` | This test guide and viva preparation manual. |
| `README.md` | Complete project documentation and GitHub overview. |
