# Healthcare — Diabetes Prediction Capstone: Execution Plan

**Scope:** Theme 1 (Healthcare) from the PDF. No PPT. All 3 bonus points included.
**Dataset:** Pima Indians Diabetes Dataset — 768 rows, 8 features + Outcome target.
**Budget:** ~5 hours available, ~4h20min planned work, rest as buffer.

---

## PHASE 1 — CODE (~2h15min)

Write one script/notebook covering, in this exact order:

1. **Load & verify data** (10 min) — load CSV, assert shape is (768, 9), confirm all 8 PDF-specified features are present.

2. **EDA — 5+ visualizations** (40 min) — mandatory minimum is 5:
   - Correlation heatmap (explicitly required by Theme 1)
   - Outcome class balance bar chart
   - Distribution plots (Glucose/BMI/Age) split by Outcome
   - Boxplots for outlier detection (Insulin, SkinThickness, BloodPressure, BMI)
   - Pairplot of top correlated features

3. **Data cleaning** (25 min) — Glucose, BloodPressure, SkinThickness, Insulin, and BMI use `0` as a disguised missing value (biologically impossible). Replace with NaN, then median-impute.
   - **Design decision to get right the first time:** impute using the *overall* column median, NOT grouped by Outcome. Grouping by the target leaks label information into the input features (data leakage) and can't be replicated at real prediction time, since a new patient's Outcome is exactly what you're trying to predict.
   - Cap extreme outliers (e.g. 99th percentile) on Insulin/SkinThickness.
   - Save the imputation medians to a small JSON file — needed later for the Streamlit app to stay consistent with training.

4. **Feature engineering** (20 min) — at least 1 new feature required; plan for 2:
   - `BMI_Category` (binned: Underweight/Normal/Overweight/Obese, one-hot encoded)
   - `Glucose_to_Insulin_ratio`

5. **Train-test split (80:20) + train all 4 models** (30 min) — stratify on Outcome since it's classification:
   - Logistic Regression (scaled)
   - Polynomial Logistic, degree=2 (scaled)
   - Decision Tree Classifier
   - Random Forest Classifier
   - Save each as `model_1_logistic.pkl` ... `model_4_random_forest.pkl`, matching the PDF's exact required filenames.

6. **Bonus: GridSearchCV tuning** (20 min) — tune Decision Tree (`max_depth`, `min_samples_split`) and Random Forest (`n_estimators`, `max_depth`). Use the tuned versions from this point onward, not the untuned base models.

7. **Evaluation on the 80:20 test split** (20 min) — comparison table (Accuracy, Precision, Recall, F1) for all 4 models, saved as both CSV and a rendered image.

8. **Bonus: 5-fold Cross-Validation** (15 min) — run on all 4 (tuned) models across the full dataset, report mean accuracy ± std per model.

9. **Final model selection — decide by CV, not just the single split** (15 min) — a single 80:20 split on ~154 test rows has real sampling variance; whichever model wins that one split isn't necessarily the most reliable. Plan to use 5-fold CV mean accuracy as the deciding metric, and explicitly write a sentence if it disagrees with the single-split "winner" — that disagreement is itself a legitimate, defensible insight for your guide, not a mistake to hide.

10. **Confusion matrix + feature importance** (included above) — confusion matrix for the final selected model; feature importance bar chart from Random Forest specifically (matches what the PDF's chart expects).

11. **Bonus: Streamlit demo app** (40 min) — input form for the 8 raw features → loads the saved Random Forest model → outputs Diabetic/Not Diabetic + probability.
    - **Design decision to get right the first time:** if a user leaves a field like Insulin at 0/unknown, the app must substitute the same median used during training cleaning (from the saved JSON) — not pass the raw 0 straight to the model. Otherwise the app silently feeds the model data far outside what it was trained on.

12. **Final written recommendation** (10 min) — 2-3 sentences naming the selected model, its CV accuracy, and top 3 predictive features, saved as a short text file.

---

## PHASE 2 — RUN (~15 min, folded into Phase 1 in practice)

Execute the full script top to bottom in one pass. Expect it to mostly work on a small, clean dataset like this — the goal here isn't a flawless first run, it's catching *real* issues before they become invisible mistakes in your final submission.

---

## PHASE 3 — DEBUG (budget ~30-45 min, don't skip this)

Things worth specifically checking, based on the kinds of issues that actually show up on a first pass:

- **Cosmetic:** deprecation warnings (e.g. seaborn `palette` without `hue`), tables/labels rendering with overflow or wrong axis labels — easy to miss if you don't actually open the generated images.
- **Correctness:** manually spot-check 1-2 known rows — take a real row from the dataset, run it through the *exact* pipeline the model was trained on, and confirm the prediction matches the recorded Outcome reasonably (not necessarily 100%, but reasonable).
- **Consistency between training and the app:** this is the one that's easy to get wrong silently. If cleaning/feature engineering logic in the notebook and the Streamlit app diverge even slightly (e.g. different imputation strategy), the app will quietly give worse predictions than your reported metrics — and you won't notice unless you deliberately test it with an edge case (like a 0/unknown value).
- **Actually launch the Streamlit app** (not just check it compiles) and confirm it serves a real prediction end-to-end, not just that the code has no syntax errors.

---

## PHASE 4 — PACKAGE (~20 min)

Assemble the exact folder structure required by the PDF:

```
your_project/
├── data/dataset.csv
├── notebooks/analysis.ipynb
├── models/
│   ├── model_1_logistic.pkl
│   ├── model_2_polynomial.pkl
│   ├── model_3_decision_tree.pkl
│   ├── model_4_random_forest.pkl
│   └── inference_medians.json      (needed by the Streamlit app)
├── results/
│   ├── comparison_table.png / .csv
│   ├── cross_validation_results.csv
│   ├── confusion_matrix_best_model.png
│   ├── feature_importance.png
│   ├── eda_1..5_*.png
│   └── final_recommendation.txt
├── app.py                           (Streamlit bonus)
└── README.md
```

Write the README: problem statement, dataset source, how to run the notebook, how to run `streamlit run app.py`, and the final model recommendation.

---

## PHASE 5 — SUBMIT (~10 min)

- Push to GitHub, get the repo link
- Walk the full mandatory checklist from the PDF one more time, ticking off each item against what's actually in the repo (don't just trust memory — open the folder and check):
  - [ ] EDA with 5+ visualizations
  - [ ] Data cleaning (nulls/outliers handled)
  - [ ] Feature engineering (≥1 new feature)
  - [ ] 80:20 train-test split
  - [ ] All 4 models trained & evaluated
  - [ ] Comparison table (Accuracy/Precision/Recall/F1)
  - [ ] Feature importance chart
  - [ ] Best model justification (2-3 sentences)
  - [ ] GitHub repo link
  - [ ] Bonus: GridSearchCV
  - [ ] Bonus: 5-fold CV
  - [ ] Bonus: Streamlit/Gradio demo

---

## Total: ~4h20min planned, ~40min buffer inside your 5-hour window.

When you're ready for Option B, this plan is the exact spec to execute against — nothing here is exploratory, it's already been thought through in detail.