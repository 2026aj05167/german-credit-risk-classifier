# Credit Risk Classification: German Credit Data

ML Assignment 2, Machine Learning (S1-26_AIMLZG565), M.Tech AIML, BITS Pilani WILP

- **Live Streamlit app:** https://2026aj05167-german-credit-risk-classifier.streamlit.app/
- **GitHub repository:** https://github.com/2026aj05167/german-credit-risk-classifier

---

## a. Problem statement

A bank has to decide whether a loan applicant is a **good** credit risk (will repay) or a **bad** one (likely to default).
Giving a loan to a bad applicant costs the bank much more than turning away a good one, so the main goal is to find the
bad applicants while not rejecting too many good ones.

This is a **binary classification** problem. I trained five classifiers on the same data, compared them on six
metrics and deployed them in a Streamlit app. In the app you can upload test data, pick a model and see its
metrics, confusion matrix and classification report.

**Bad** is used as the positive class (label 1), so Precision, Recall and F1 in this report are all for the bad class.

## b. Dataset description

| Item | Details |
|------------|------------------------------------|
| Dataset | Statlog (German Credit Data), Prof. Hans Hofmann, University of Hamburg |
| Source | UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data (DOI 10.24432/C5NC77, CC BY 4.0) |
| Type of problem | Binary classification (good / bad credit risk) |
| Instances | 1,000 |
| Features | 20 input features: 7 numeric, 13 categorical |
| Numeric features | duration_months, credit_amount, installment_rate, residence_since, age, existing_credits, num_dependents |
| Categorical features | checking_account, credit_history, purpose, savings_account, employment_since, personal_status_sex, other_debtors, property, other_installment_plans, housing, job, telephone, foreign_worker |
| Target | `credit_risk`: good (700) / bad (300) |
| Missing values / duplicates | None / none, so no rows were removed |
| Preprocessing | The raw file stores categories as codes (A11, A34, ...). I decoded them into readable labels using `german.doc`. Inside each model pipeline, numeric columns go through StandardScaler and categorical columns through OneHotEncoder (`handle_unknown="ignore"`), which gives 61 columns after encoding. |
| Train / test split | 80 / 20 (800 / 200 rows), stratified on the target, `random_state=5167` |
| Class distribution | 30% bad in the full data, train set and test set (thanks to stratification) |

Quick findings from EDA:

- The data is imbalanced (70/30). Predicting "good" for everyone already gives 70% accuracy with MCC = 0, so accuracy alone is not a fair way to compare the models.
- Bad loans run longer (median 24 vs 18 months) and are slightly larger (median 2,575 vs 2,244 DM).
- Checking account status is a strong signal: about 49% of applicants with a negative balance are bad, against about 12% of those with no checking account.

## c. GitHub repository link

<GITHUB_REPO_LINK>

```
german-credit-risk-classifier/
├── app.py                  # Streamlit app (no retraining)
├── model_training.ipynb    # prep, EDA, training, evaluation
├── requirements.txt
├── README.md
├── test_data.csv           # 200-row hold-out test set
├── data/
│   ├── german.data             # original UCI file
│   ├── german.doc              # UCI docs (code meanings)
│   └── german_credit_clean.csv # decoded, readable version
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    └── model_comparison.csv
```

Each `.joblib` file is a full sklearn `Pipeline` (preprocessing + classifier), so the app can take the raw CSV columns as they are.

## d. Models used

All five models were trained on the same 800 training rows. Hyper-parameters were chosen with `GridSearchCV`
(5-fold stratified CV on the training set only, scoring = F1 of the bad class). The table shows the results on the
200-row test set.

| Model | Chosen hyper-parameters |
|-------------|---------------------------|
| Logistic Regression | C = 10, class_weight = balanced |
| Decision Tree | gini, max_depth = 4, min_samples_leaf = 20, class_weight = balanced |
| KNN | k = 3, uniform weights |
| Naive Bayes (Gaussian) | var_smoothing = 1e-3 |
| SVM | RBF kernel, C = 1, gamma = 0.01, class_weight = balanced, probability = True |

### Comparison table (test set, 200 rows)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|------------------|----------|-------|-----------|--------|-------|-------|
| Logistic Regression | 0.6900 | 0.7725 | 0.4884 | 0.7000 | 0.5753 | 0.3570 |
| Decision Tree | 0.6150 | 0.6783 | 0.4261 | 0.8167 | 0.5600 | 0.3200 |
| KNN | 0.7250 | 0.7125 | 0.5714 | 0.3333 | 0.4211 | 0.2728 |
| Naive Bayes | 0.6750 | 0.7364 | 0.4737 | 0.7500 | 0.5806 | 0.3605 |
| Support Vector Machine | 0.6850 | **0.7890** | 0.4839 | 0.7500 | **0.5882** | **0.3741** |

Reference: always predicting "good" gives accuracy 0.700 and MCC 0.000.

### Observations

| ML Model Name | Observation about model performance |
|-----------|--------------------------------------------|
| Logistic Regression | AUC 0.77, MCC 0.36. Grid search picked balanced class weights, so it catches 70% of the bad applicants (recall 0.70) at precision 0.49. A simple linear model, and still close to the best, so after one-hot encoding most of the signal is roughly linear. Its coefficients are also easy to explain. |
| Decision Tree | Highest recall (0.82), but the lowest accuracy (0.615) and AUC (0.68). The tree is shallow (depth 4) with balanced weights, so it labels many good applicants as bad (precision 0.43). Its probability scores are coarse because there are only a few leaves, which keeps AUC low. |
| KNN | Highest accuracy (0.725), but this number is misleading: recall is only 0.33, so it misses two out of three bad applicants. KNN has no class-weight option, so with 70% "good" neighbours it mostly follows the majority class. It has the lowest F1 (0.42) and MCC (0.27). |
| Naive Bayes | Better than I expected: recall 0.75, F1 0.58, MCC 0.36, about the same as Logistic Regression. The independence assumption does not really hold (duration and credit amount have a correlation of about 0.62), but it still ranks applicants reasonably well (AUC 0.74). Fastest to train. |
| Support Vector Machine | Best AUC (0.789), F1 (0.588) and MCC (0.374), with recall 0.75. The RBF kernel with balanced class weights separates the classes a little better than the linear model. Slowest to train and the hardest to interpret. |
| **Overall winner for this dataset?** | **SVM (RBF kernel).** It is the best on AUC, F1 and MCC, which are the metrics that matter when the bad class is the minority. Naive Bayes and Logistic Regression are close behind. With only 200 test rows the gap between the top three is small, and if the bank had to explain each decision I would pick Logistic Regression. |

## Streamlit app

| Required feature | How it is done in `app.py` |
|------------|------------------------------------|
| Dataset upload (CSV) | Sidebar file uploader for the test CSV. The app checks that all 20 feature columns and `credit_risk` are present and shows an error listing any missing columns. A download button for `test_data.csv` and a "use sample data" checkbox are also there. |
| Model selection dropdown | Sidebar select box with the five models, with a one-line note of the tuned settings of the selected model. |
| Evaluation metrics | Six metric cards: Accuracy, AUC, Precision, Recall, F1, MCC (hover over "?" for what each one means). |
| Confusion matrix / classification report | Confusion matrix heatmap plus the full classification report. A second tab compares all 5 models on the uploaded file, and a third tab shows row-level predictions that can be downloaded. |

The models are loaded with `joblib.load` from `model/`. The app only calls `predict` / `predict_proba` and never retrains.

## How to run it locally

```bash
pip install -r requirements.txt
jupyter notebook model_training.ipynb   # optional: re-runs training and re-creates model/ and test_data.csv
streamlit run app.py
```

The saved models were created with scikit-learn 1.9.1, so use the versions pinned in `requirements.txt`
(Python 3.12 was used for deployment on Streamlit Community Cloud).

## Limitations

- The dataset is small (1,000 rows), so a 200-row test set gives fairly noisy metrics. A different split could change the order of the top three models.
- `personal_status_sex` and `foreign_worker` are sensitive attributes. I kept them because they are part of the original dataset, but a real credit model would need a fairness review before using them.
