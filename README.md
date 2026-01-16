# HorizonUp

A reproducible data-science pipeline for survival analysis and risk prediction.

This repository contains a Jupyter notebook and supporting code that sets up a complete data-science pipeline applied to survival analysis and risk prediction. It combines statistical survival modeling (e.g., Cox proportional hazards) and supervised machine learning (logistic regression, decision trees, Random Forest, Gradient Boosting, XGBoost) to generate interpretable risk scores and high-performance predictions for time-to-event (and binary) outcomes. This is especially useful for medical and decision-making contexts where both interpretability and predictive power matter.

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data expectations](#data-expectations)
- [Notebook / Pipeline overview](#notebook--pipeline-overview)
- [Modeling approaches](#modeling-approaches)
- [Evaluation metrics](#evaluation-metrics)
- [How to run](#how-to-run)
- [Results and interpretation](#results-and-interpretation)
- [Reproducibility and tips](#reproducibility-and-tips)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Features

- Data loading, cleaning, and missing-value handling.
- Variable selection and feature engineering.
- Statistical survival analysis using lifelines (CoxPH, Kaplan–Meier where appropriate).
- Supervised machine learning: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost.
- Train / test split and cross-validation, hyperparameter tuning.
- Evaluation using survival- and regression/classification-appropriate metrics.
- Generation of a risk/prioritization score to flag urgent/high-risk cases.
- Visualization of model results and feature importance.

---

## Requirements

- Python 3.8+ (3.10 recommended)
- Recommended packages (install via pip or conda):
  - numpy
  - pandas
  - scikit-learn
  - lifelines
  - xgboost
  - matplotlib
  - seaborn
  - jupyterlab / notebook
  - joblib (optional)

Example quick install:
```bash
python -m venv .venv
source .venv/bin/activate         # macOS / Linux
.venv\Scripts\activate            # Windows PowerShell

pip install -U pip
pip install numpy pandas scikit-learn lifelines xgboost matplotlib seaborn jupyterlab joblib
```

You can also create a requirements.txt with the exact versions you need, then:
```bash
pip install -r requirements.txt
```

---

## Data expectations

The notebook expects a tabular dataset with (at minimum) the following columns or equivalents:

- `time` — follow-up duration (time to event or censoring)
- `event` — binary indicator (1 if event observed, 0 if censored)
- Predictor columns — features used for modeling (demographics, lab results, vitals, derived features, etc.)

Notes:
- Time units must be consistent (e.g., days).
- Missing values are handled in the notebook (imputation options: median, mean, model-based).
- If you only want to build classifiers (risk at fixed horizon), create a binary label for the horizon of interest.

If your dataset uses different column names, adapt the reading/preprocessing cell to map them.

---

## Notebook / Pipeline overview

Typical steps implemented in the notebook:

1. Environment setup: install & import libraries.
2. Data loading: read CSV / Parquet and basic sanity checks.
3. Exploratory Data Analysis (EDA): distributions, missingness, correlations.
4. Preprocessing:
   - Imputation for missing values.
   - Encoding categorical variables (one-hot or ordinal).
   - Scaling numeric features if required.
   - Feature selection (filtering low-variance or highly correlated features).
5. Split data:
   - Train / validation / test split (time-aware splits optional).
6. Survival analysis:
   - Kaplan–Meier plots for group-level survival.
   - Cox Proportional Hazards model via lifelines: fit, interpret coefficients, check proportional hazards assumptions.
   - Concordance index (c-index) for model discrimination.
7. Supervised ML:
   - Train classifiers/regressors (logistic regression, decision tree, random forest, gradient boosting, XGBoost).
   - Hyperparameter tuning (GridSearchCV / RandomizedSearchCV) and cross-validation.
   - Calibration checks and probability outputs (when relevant).
8. Evaluation:
   - Survival-specific metrics (concordance index).
   - Classification/regression metrics: ROC AUC, precision, recall, F1, MAE, R² (as applicable).
   - Confusion matrix, ROC curves, feature importance plots.
9. Risk scoring & prioritization:
   - Combine survival/risk outputs into a prioritization list.
   - Thresholds and triage logic for urgent cases.
10. Reporting: summary tables and visualizations for stakeholders.

---

## Modeling approaches

- Survival modeling:
  - lifelines.CoxPHFitter for hazard ratios and interpretability.
  - Kaplan–Meier for non-parametric survival curves.
  - Check PH assumption (Schoenfeld residuals) and create stratified models if needed.

- Machine learning:
  - Binary risk prediction using logistic regression and tree-based models.
  - Ensemble methods (Random Forest, Gradient Boosting, XGBoost) for improved predictive performance.
  - Feature importance via permutation importance, SHAP (optional), or built-in model importances.

- Combined strategy:
  - Use Cox model coefficients for interpretable hazard insights.
  - Use ML model probabilities/risk scores to improve ranking and sensitivity.
  - Optionally ensemble/stack models or use ML predictions as features for survival models.

---

## Evaluation metrics

- Survival:
  - Concordance index (c-index): discrimination for time-to-event models.
  - Integrated Brier Score (optional) for calibration over time.

- Classification/regression (when framing as a binary prediction or risk score):
  - ROC AUC, Precision, Recall, F1-score
  - Calibration (reliability) plots
  - MAE and R² if predicting continuous risk scores

Choose metrics that reflect your clinical or operational priorities (e.g., sensitivity for urgent case detection).

---

## How to run

Open the Jupyter notebook included in this repository (e.g., `survival_analysis_pipeline.ipynb`) and run cells sequentially. Example:

```bash
git clone https://github.com/ladharimanel/HorizonUp.git
cd HorizonUp

# Optional: create venv and install dependencies as shown above
jupyter lab      # or: jupyter notebook
```

If you prefer scripted execution, create/run a script entrypoint (e.g., `python run_pipeline.py`) that performs data loading, training, and evaluation. The notebook is the recommended, interactive workflow.

---

## Results and interpretation

- Use Cox model outputs to extract interpretable hazard ratios and confidence intervals for predictors.
- Use ML models to obtain high-performing risk scores; inspect feature importances to validate domain plausibility.
- Combine both outputs for a prioritized list of cases:
  - Use survival probabilities at a clinically relevant horizon (e.g., 30/90 days).
  - Use ML risk probabilities for short-term event risk ranking.
- Visualizations in the notebook help communicate findings to clinicians or decision-makers.

---

## Reproducibility & tips

- Set a random seed for train/test splits and model training to make results reproducible.
- Save model artifacts (joblib, pickle) and use versioned datasets.
- Document data transformations; consider storing preprocessed datasets.
- If sharing or publishing, provide synthetic or de-identified sample data and a clear README (this file).

---

## Contributing

Contributions are welcome. Suggested workflow:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feat/your-feature`.
3. Make changes and add tests where appropriate.
4. Submit a pull request describing your changes.

Please follow best practices for data privacy if contributing real/clinical datasets.

---


## Contact

If you need help adapting the notebook to your dataset, generating a requirements.txt, or creating a runnable script, open an issue or contact the repository owner.

---

Acknowledgements
- Built using standard Python data-science libraries: NumPy, Pandas, Scikit-learn, Lifelines, XGBoost.
