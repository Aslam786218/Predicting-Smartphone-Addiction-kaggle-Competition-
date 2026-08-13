# Predicting Smartphone Addiction — Kaggle Playground Series S6E8

Solution for the [Kaggle Playground Series - Season 6, Episode 8](https://www.kaggle.com/competitions/playground-series-s6e8) competition, hosted by Google LLC and Kaggle.

## Problem Statement

Predict the probability that an individual is addicted to their smartphone (`addicted_label`) based on behavioral and demographic features, using a synthetically generated tabular dataset.

**Evaluation metric:** ROC-AUC (Area Under the Receiver Operating Characteristic Curve)

## Dataset

| File | Rows | Description |
|---|---|---|
| `train.csv` | 691,369 | Training data with target label |
| `test.csv` | 296,302 | Test data for prediction |
| `sample_submission.csv` | 296,302 | Submission format reference |

### Features

| Column | Type | Description |
|---|---|---|
| `age` | Numeric | Age of respondent |
| `daily_screen_time_hours` | Numeric | Average daily screen time |
| `social_media_hours` | Numeric | Daily social media usage |
| `gaming_hours` | Numeric | Daily gaming usage |
| `work_study_hours` | Numeric | Daily work/study hours |
| `sleep_hours` | Numeric | Average daily sleep |
| `notifications_per_day` | Numeric | Notification count per day |
| `app_opens_per_day` | Numeric | App open count per day |
| `weekend_screen_time` | Numeric | Weekend screen time |
| `gender` | Categorical | Male / Female / Other |
| `stress_level` | Categorical | Low / Medium / High |
| `academic_work_impact` | Categorical | Yes / No |
| `addicted_label` | Target | 0 = Not addicted, 1 = Addicted |

All feature columns contain missing values (roughly 15–20% per column). The target is imbalanced (~71% positive, ~29% negative).

## Approach

1. **Preprocessing**
   - `stress_level` ordinal-encoded (Low=0, Medium=1, High=2) to preserve natural order
   - `academic_work_impact` binary-encoded (No=0, Yes=1)
   - `gender` handled as a native categorical feature
   - No manual imputation — missing values are passed directly to LightGBM, which handles them internally via optimal split learning

2. **Model**
   - [LightGBM](https://lightgbm.readthedocs.io/) gradient boosting classifier, binary objective
   - Key hyperparameters: `learning_rate=0.05`, `num_leaves=63`, `subsample=0.8`, `colsample_bytree=0.8`, L1/L2 regularization

3. **Validation**
   - 5-fold Stratified K-Fold cross-validation to preserve class balance across folds
   - Early stopping on validation AUC per fold
   - Out-of-fold (OOF) predictions used to compute an unbiased overall CV score

4. **Inference**
   - Test predictions averaged across all 5 fold models

## Results

| Submission | Model | Features | Approach/Solution | CV AUC | Leaderboard | Status |
|---|---|---|---|---|---|---|
| 1 | LightGBM | 13 original | 5-fold CV, native NaN handling, early stopping | 0.96358 | 0.9636 | Completed |
| 2 | CatBoost | 13 original | Native categorical support, 5-fold CV, early stopping | 0.96387 | 0.96387 | Completed |
| 3 | Ensemble (LGB + CB) | 13 original | 50/50 blend of LightGBM + CatBoost predictions | 0.96361 + 0.96387 | **0.96524** | Completed |
| 4 | CatBoost + Features | 13 + 9 engineered | Added ratio features (social_media_ratio, gaming_ratio, app_efficiency, sleep_deficit, etc.) | 0.96401 | 0.96401 | Completed |
| 5 | Triple Blend | 13 + 9 engineered | 33/33/33 blend of LightGBM + CatBoost + CatBoost+Features | 	0.96372 | 0.96485 | Completed |
| 6 | Weighted Ensemble (LGB + CB) | 13 original |  60/40 weighted blend of LightGBM + CatBoost predictions |  0.96378 | 0.96507 | Completed |
| 7 | Triple Blend (LGB + CB + XGB) |   13 original  |     50/20/30 weighted blend of LightGBM + CatBoost + XGBoost predictions |0.96418|0.96553 ✅ Best|Completed|
| 8 | Stacking (LGB + CB Meta-Learner)  | 13 original | Logistic Regression meta-learner trained on LGB + CB OOF predictions to learn optimal weights |0.967171||completed|
| 9 |  |     |      ||||

## Repository Structure
```
.
├── CSV
├── LICENSE
├── Notebook
├── README.md
└── requirements.txt
```

## Requirements
1.   pandas>=1.3.0
2.   numpy>=1.20.0
3.   scikit-learn>=1.0.0
4.   lightgbm>=3.0.0
5.   catboost>=1.0.0


## Usage

**Option 1**
1. Download the competition data from Kaggle and place `train.csv`, `test.csv` in the working directory (or run directly as a Kaggle Notebook, where data auto-mounts).
2. Run `notebook.ipynb` top to bottom.
3. `submission.csv` is generated in the specified output directory, ready to submit to the competition.

**Option 2**
1.   Go to the competition page → Code → New Notebook.
2.   Kaggle automatically mounts data at /kaggle/input/competitions/playground-series-s6e8/.
3.   Pick a notebook from the notebooks/ folder and copy the code.
4.   Paste into a cell and Run All.
5.   'submission.csv' is generated in the notebook's Output folder.

## Submitting to Kaggle

There are two ways to submit predictions for this competition:

**Option 1 — CSV Upload**
1. Go to the competition page → **Submit Predictions**
2. Upload the generated `submission.csv` directly
3. Add an optional description and confirm

**Option 2 — Notebook Submission (recommended)**
1. Create/open the notebook on Kaggle with the competition data auto-attached
2. Run `notebook.ipynb` top to bottom — this generates `submission.csv` in the notebook's output
3. From the notebook's **Output** tab, click **Submit to Competition** — no manual download/upload required

Notebook submission is preferred since Kaggle links the exact code that produced each submission, which is useful for tracking experiments and satisfying reproducibility requirements if a submission places.

## Possible Next Steps

- Feature engineering (ratio features, e.g. `social_media_hours / daily_screen_time_hours`)
- Hyperparameter tuning (grid/Bayesian search)
- Ensembling with CatBoost / XGBoost
- Comparing native NaN handling against explicit imputation strategies

## Citation

Yao Yan, Walter Reade, Elizabeth Park. *Predicting Smartphone Addiction.* https://kaggle.com/competitions/playground-series-s6e8, 2026. Kaggle.

## Author

Aslam Hasan Sayyad
