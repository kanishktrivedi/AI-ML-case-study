# AlertRank: Machine-Learning-Based Prioritization of Security Alerts for SOC Triage
Case Study No. 85 – Machine Learning Fundamentals, B.Tech CSE 2024-28, Sem V

## 1. Problem Definition
A SOC receives thousands of alerts daily; most are false or benign. Analyst time is the bottleneck.
**Formulation:** supervised, multi-class classification of each alert into {TruePositive, BenignPositive, FalsePositive}
(IncidentGrade assigned by analysts). The class probabilities are converted into a **priority score**
(P(TP) + 0.3·P(BP)) so the queue can be *ranked*. Success = finding most real threats while reviewing a small fraction of alerts.
**Why ML:** static severity rules ignore context (entity type, time of day, detector reliability, geography).

## 2. Dataset & Documentation
- **Primary source (use this for your final submission):** Microsoft GUIDE – "Microsoft Security Incident Prediction", Kaggle
  (≈9.5M alert rows, labelled TP/BP/FP, MITRE techniques, entity, geo and OS fields). Place the file at `data/alerts.csv`
  (sample ≤200k rows; `train.py` already does this) and keep the same column names.
- **Fallback used in this prototype:** `generate_data.py` creates a synthetic dataset with the same schema, with injected
  duplicates and missing values. *Results on synthetic data must not be reported as real-world findings.*
- Variables: Timestamp, DetectorId, AlertTitle, Category, MitreTechniques, EntityType, EvidenceRole, OSFamily, CountryCode, IncidentGrade (target).
- Quality issues found (see `outputs/quality.json`, `outputs/missing.png`): duplicates, ~30% missing MITRE, ~8% missing OS,
  ~5% missing country, missing labels, class imbalance, high-cardinality IDs.

## 3. EDA (outputs/eda.png)
Record observations: class balance; grade mix per Category; TP rate by hour (off-hours effect); TP rate by MITRE count; missingness patterns.
Replace with the real findings after running on GUIDE.

## 4. Preprocessing
Drop duplicates and unlabeled rows → parse timestamp (Hour, DayOfWeek, OffHours) → MITRE count + missing flag →
missing categoricals = "Unknown" → one-hot (rare levels grouped, min_frequency=20) + scaling for linear model; ordinal encoding for trees →
stratified 80/20 split → class weighting for imbalance.

## 5. Model Development
Baseline (majority), Logistic Regression, Random Forest, HistGradientBoosting (`outputs/model_comparison.csv`).
Justification: LR = interpretable baseline; RF/HGB capture non-linear feature interactions on tabular data.
Suggested extra experiments: hyper-parameter grid, SMOTE vs class weights, with/without DetectorId, time-based split.

## 6. Evaluation
Accuracy is misleading under imbalance, so use **macro-F1, TP recall, PR-AUC for TP, Recall@top-20%** and the gain curve.
Add: confusion matrix, error analysis (`missed_true_positives.csv`), permutation importance.
**Limitations:** noisy analyst labels; random split overstates performance (concept drift → use time split); FP is costlier-missed than reviewed,
so thresholds should be tuned to SOC capacity; fairness over geography/country features; adversarial adaptation.

## 7. Streamlit App
`streamlit run app.py` – single-alert scoring, CSV queue ranking with download, model/EDA dashboard.

## 8. Conclusion & Future Work
Summarise gains from the ranking vs random order; future: sequence/incident-level modelling, SHAP, online learning.
