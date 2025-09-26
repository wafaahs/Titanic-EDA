# Titanic Survival Prediction — End-to-End ML Tutorial

## Project Overview
This project explores the **Titanic: Machine Learning from Disaster** dataset on Kaggle. It is designed as a **tutorial and portfolio project** that demonstrates a complete machine learning workflow on tabular data:

- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Building with Pipelines
- Cross-Validation and Model Comparison
- Evaluation (ROC, PR, Calibration, Confusion Matrix)
- Interpretability (Permutation Importance)
- Saving/Loading Models for Reuse

Dataset: https://www.kaggle.com/c/titanic

---

## Repository Structure
```
titanic-ml-tutorial/
├─ notebooks/
│   └─ titanic_full_pipeline.ipynb        # main end-to-end notebook
├─ reports/
│   ├─ figures/                           # saved plots (EDA + evaluation)
│   │   ├─ 01_missingness.png
│   │   ├─ 02_survival_by_sex_pclass.png
│   │   ├─ 03_age_fare.png
│   │   ├─ 04_survival_by_title.png
│   │   ├─ 07_permutation_importance.png
│   │   ├─ 15_eval_curves.png
│   │   └─ 16_calibration.png
│   └─ leaderboard.csv                    # model comparison results
├─ models/
│   └─ best_pipeline.joblib               # serialized model (optional)
├─ requirements.txt                       # dependencies
└─ README.md                              # this file
```

---

## Getting Started

### 1) Environment
Install dependencies locally:
```bash
pip install -r requirements.txt
```

Minimal requirements (pin as needed):
```
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
```

### 2) Data
Do **not** commit Kaggle CSVs. Use the competition page to access the data. In Kaggle Notebooks, attach the "Titanic - Machine Learning from Disaster" dataset and load:
```python
train = pd.read_csv("/kaggle/input/titanic/train.csv")
test  = pd.read_csv("/kaggle/input/titanic/test.csv")
```

### 3) Run the Notebook
Open and run:
```
notebooks/titanic_full_pipeline.ipynb
```
This will generate figures in `reports/figures/`, a `reports/leaderboard.csv`, and optionally a serialized pipeline in `models/`.

---

## EDA Highlights
Key descriptive findings typically observed on this dataset:
- Women have substantially higher survival rates than men.
- First-class passengers have higher survival rates than third-class.
- Younger passengers (children) tend to survive more often than adults.
- Titles extracted from names (Mr, Mrs, Miss, Master, Rare) are predictive.

Example visualization placeholder:
```
reports/figures/02_survival_by_sex_pclass.png
```

---

## Feature Engineering
Features created beyond the raw dataset:
- `Title` (from names) → {Mr, Mrs, Miss, Master, Rare}
- `FamilySize` = `SibSp` + `Parch` + 1
- `IsAlone` = 1 if `FamilySize == 1` else 0
- `CabinDeck` = first letter of `Cabin` and `HasCabin` indicator
- `GroupSize` from ticket counts
- `FarePerPerson` = `Fare / GroupSize`

These features are integrated via an `sklearn` `ColumnTransformer` within a `Pipeline` to avoid leakage and keep preprocessing tied to the model.

---

## Model Comparison (Cross-Validation Leaderboard)
All models are evaluated with **Stratified 5-Fold Cross-Validation**. Primary metric: **ROC-AUC** (mean ± std). Accuracy is also reported.

| Model               | ROC-AUC | Accuracy | Notes                  |
|---------------------|---------|----------|------------------------|
| Gradient Boosting   | 0.84    | 0.82     | Often strongest        |
| Random Forest       | 0.83    | 0.81     | Strong baseline        |
| Logistic Regression | 0.77    | 0.79     | Interpretable baseline |
| Decision Tree       | 0.70    | 0.75     | Overfits easily        |
| KNN                 | 0.72    | 0.73     | Sensitive to scaling   |

See: `reports/leaderboard.csv` for your exact run.

---

## Evaluation of Selected Model
For the chosen model (based on CV leaderboard), the notebook generates:
- ROC curve
- Precision–Recall curve
- Confusion matrix
- Calibration curve and Brier score

Example figure placeholders:
```
reports/figures/15_eval_curves.png
reports/figures/16_calibration.png
```

---

## Interpretability
Permutation importance (model-agnostic) is produced for the validation split to identify drivers of predictions.

Expected top features:
- Sex
- Pclass
- Title
- FamilySize / IsAlone
- FarePerPerson

See: `reports/figures/07_permutation_importance.png`.

---

## Reproducibility Notes
- Random seeds are set for `numpy` and scikit-learn where applicable.
- Preprocessing is encapsulated in an `sklearn.Pipeline` for portability.
- The trained pipeline can be saved and reloaded via `joblib`:
```python
import joblib
pipe = joblib.load("models/best_pipeline.joblib")
# X_new should have the same columns as training (numeric + categorical)
y_hat = pipe.predict(X_new)
p_hat = pipe.predict_proba(X_new)[:, 1]
```

---

## Links
- Kaggle competition: https://www.kaggle.com/c/titanic
- (Add Kaggle Notebook URL once published)
- (Add GitHub repository URL once published)

---

## Next Steps
- Hyperparameter tuning (RandomizedSearch or Optuna)
- Try boosted tree libraries (XGBoost / LightGBM) if available
- Fairness slice analysis (metrics by Sex, Pclass, Embarked)
- Optional: small Streamlit app to demo predictions

---

## License
Specify a license if you plan to share the code (e.g., MIT).
