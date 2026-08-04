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

| Fold | AUC |
|---|---|
| 1 | 0.96278 |
| 2 | 0.96353 |
| 3 | 0.96397 |
| 4 | 0.96439 |
| 5 | 0.96322 |

**Mean fold AUC:** 0.96358 (± 0.00056)
**Overall OOF AUC:** 0.96357

## Repository Structure

```
.
├── README.md
├── notebook.ipynb          # Full training + inference pipeline
├── submission.csv          # Final submitted predictions
└── requirements.txt        # Python dependencies
```

## Requirements

```
pandas
numpy
scikit-learn
lightgbm
```

## Usage

1. Download the competition data from Kaggle and place `train.csv`, `test.csv` in the working directory (or run directly as a Kaggle Notebook, where data auto-mounts).
2. Run `notebook.ipynb` top to bottom.
3. `submission.csv` is generated in the specified output directory, ready to submit to the competition.

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
