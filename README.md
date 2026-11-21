# Lead Scoring Conversion Prediction (Binary Classification)

An end-to-end machine learning classification workflow built for **Module 3 of the DataTalks.Club Machine Learning Zoomcamp (2026 Cohort)**. The project predicts whether incoming marketing leads will convert (`converted = 1`) or not (`converted = 0`) using demographic, behavioral, and engagement features.

---

## Project Overview & Workflow

The pipeline follows a structured 12-step machine learning lifecycle:
1. **Data Ingestion & Cleaning:** Standardized column names and string values (`pandas 3.0+` compatible), imputing missing categorical values with `'NA'` and missing numerical values with `0.0`.
2. **Validation Framework:** Split the dataset into **60% train / 20% validation / 20% test** (`random_state=42`) to prevent data leakage.
3. **Exploratory Data Analysis (EDA):** Analyzed class balance, category frequencies, group conversion rates, and risk ratios.
4. **Feature Importance:** Evaluated categorical predictive signal using **Mutual Information** and numerical relationships via **Pearson Correlation**.
5. **Feature Engineering:** One-hot encoded categorical variables using Scikit-Learn's `DictVectorizer`.
6. **Modeling & Interpretation:** Trained a baseline `LogisticRegression` classifier (`liblinear` solver), inspected learned feature weights, and ran leave-one-out **feature elimination**.
7. **Hyperparameter Tuning & Inference:** Tuned the inverse regularization strength (`C`) and evaluated the final model on the held-out test set.

## Key Analytical Findings

### 1. Categorical Feature Insights (Conversion Rates & Mutual Information)
* **`lead_source` (Strongest Categorical Signal — MI: `0.03`):** Conversion rates vary by over **35 percentage points**, led by `referral` (**76.4%**, risk ratio `1.34`) and `organic_search` (**67.8%**, risk ratio `1.19`), compared to `events` (**41.1%**) and `paid_ads` (**42.8%**).
* **`employment_status` (MI: `0.02`):** `employed` leads convert at **65.6%** (`risk_ratio = 1.15`), significantly higher than `student` leads (**42.0%**, `risk_ratio = 0.74`).
* **`industry` & `location` (MI: `0.00`):** Show minimal standalone predictive spread across categories.

### 2. Numerical Feature Correlations
* **`lead_score` (`r = 0.48`)**, **`interaction_count` (`r = 0.45`)**, and **`number_of_courses_viewed` (`r = 0.39`)** show strong positive correlations with conversion.
* Among input features, **`interaction_count` and `lead_score`** exhibit the strongest multicollinearity (**`r = 0.9166`**).

### 3. Model Performance & Feature Elimination

| Experiment / Configuration | Validation Accuracy | Test Accuracy | Key Takeaway |
| :--- | :---: | :---: | :--- |
| Baseline Logistic Regression (`C=1.0`, all features) | `0.6450` | — | Unscaled `annual_income` dominates L2 regularization penalty |
| Feature Elimination: Without `industry` / `location` / `employment_status` | `0.6450` | — | `0.000` change in validation accuracy |
| Feature Elimination: Without `annual_income` | `0.7240` | — | `+0.079` accuracy boost when unscaled income is removed |
| **Tuned Model (`C=0.01`, full feature set)** | — | **`0.6700`** | Stronger regularization mitigates unscaled feature dominance |

---

## Tech Stack
* **Language:** Python 3.13
* **Data Manipulation:** Pandas 3.0+, NumPy
* **Machine Learning:** Scikit-Learn (`train_test_split`, `DictVectorizer`, `LogisticRegression`, `mutual_info_score`, `accuracy_score`)
* **Environment:** Jupyter Notebook

## How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com<your-username>/ml-zoomcamp-lead-scoring-classification.git
   cd ml-zoomcamp-lead-scoring-classification
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter and run the notebook:**
   ```bash
   jupyter notebook notebooks/03_lead_scoring_classification.ipynb
   ```

---

## Future Improvements
* Apply `StandardScaler` to numerical features (`annual_income`, `interaction_count`, `lead_score`, `number_of_courses_viewed`) before fitting `LogisticRegression` so L2 regularization penalizes all features evenly.
* Benchmark tree-based classifiers (Random Forest, XGBoost, LightGBM) and evaluate ROC-AUC alongside accuracy.
