# Aluminium Wire Rod Property Predictor

Predicting mechanical and electrical properties of aluminium wire rod directly from upstream process parameters, using a multi-output regression pipeline.

## Problem

In aluminium wire rod manufacturing, final properties — tensile strength, elongation, and electrical conductivity — are determined by process settings during casting and rolling. Being able to predict these properties from process parameters before physical testing can help process engineers flag out-of-spec batches earlier and reduce reliance on destructive testing.

This project models that relationship: given **casting temperature**, **rolling speed**, and **cooling rate**, predict **UTS (MPa)**, **elongation (%)**, and **electrical conductivity (% IACS)**.

## Dataset

- 10,000 synthetic samples generated to reflect realistic ranges of casting temperature, rolling speed, and cooling rate, and their effect on UTS, elongation, and conductivity.
- 3 input features, 3 target properties (multi-output regression).
- No missing values; features are uniformly distributed across their ranges (verified via EDA).

> **Note:** This dataset is synthetic, not plant/lab-measured data. It was used to validate the modeling pipeline and methodology. Results should be read as a proof of concept, not as production-ready predictions.

## Approach

1. **EDA** — distribution checks, correlation analysis, and a strength-vs-conductivity scatter plot colored by cooling rate to visualize the known metallurgical trade-off between the two properties.
2. **Baseline modeling** — Random Forest multi-output regressor as a baseline.
3. **Model benchmarking** — compared baseline Random Forest against HistGradientBoostingRegressor and XGBoost (both wrapped in `MultiOutputRegressor`).
4. **Hyperparameter tuning** — `RandomizedSearchCV` (50 candidate settings, 5-fold CV) on the HistGradientBoosting pipeline, tuning learning rate, max depth, max iterations, and min samples per leaf.
5. **Evaluation** — R², MAE, and RMSE computed independently per target property on a held-out 20% test set.
6. **Interpretability** — SHAP `TreeExplainer` applied per target to identify which process parameters drive each predicted property.

## Results

| Target Property | Baseline RF R² | Tuned Model R² | Tuned MAE | Tuned RMSE |
|---|---|---|---|---|
| UTS (MPa) | 0.736 | **0.766** | 4.03 | 5.02 |
| Elongation (%) | 0.635 | **0.669** | 0.82 | 1.02 |
| Conductivity (% IACS) | 0.040 | **0.122** | 0.39 | 0.50 |

Tuning consistently improved R² across all three targets over the untuned baseline.

## Key Insight: Why Conductivity Is Harder to Predict

Conductivity is predicted noticeably worse than UTS or elongation, even after tuning. This is consistent with metallurgical theory: electrical conductivity in aluminium is governed primarily by **alloy composition and impurity/solute content** (solid solution scattering of electrons), not by thermomechanical process parameters like casting temperature or cooling rate. UTS and elongation, by contrast, are strongly influenced by grain structure and work hardening, which *are* shaped by rolling speed and cooling rate. The model's weaker conductivity performance is therefore an expected outcome of the feature set, not a modeling failure — composition-related features (e.g., alloy grade, trace element content) would likely be needed to predict conductivity well.

## Repository Structure

```
├── Aluminium_Properties.ipynb   # Full analysis: EDA, modeling, tuning, SHAP
├── requirements.txt             # Dependencies
└── README.md
```

## Tech Stack

Python, pandas, scikit-learn, XGBoost, SHAP, matplotlib, seaborn

## Future Improvements

- Replace synthetic data with real plant/lab measurements
- Add composition-related features (alloy grade, impurity content) to improve conductivity prediction
- Experiment with feature engineering (e.g., interaction terms between cooling rate and rolling speed)
