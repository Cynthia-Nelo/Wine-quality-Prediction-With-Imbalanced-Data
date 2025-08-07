# Wine Quality Prediction (with Imbalanced Data)

This project presents a robust machine learning pipeline for predicting wine quality using physicochemical characteristics. The key challenge addressed is the **class imbalance** inherent in wine quality labels, where lower- and higher-quality wines are underrepresented but crucial for decision-making in quality control.

---

## Project Goals

- Build an **interpretable and accurate classification model** for wine quality.
- Handle **class imbalance** effectively to avoid biased predictions.
- Derive **insightful features** that impact wine quality.
- Provide **business-relevant recommendations** based on model behavior and threshold tuning.

---

## Problem Overview

Wine quality scores (0–10) are heavily skewed, with most wines falling into the middle range. Poor and excellent wines are rare, which leads to class imbalance. This project simplifies the prediction into **binary (good/bad)** and **multiclass** classification tasks and applies **advanced techniques** to ensure minority classes are properly detected.

---

##  Tools and Libraries

- **Python 3**
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn`
- `xgboost`
- `imblearn` (for ADASYN)
- `joblib`

---

## Data and Features

- Dataset: Red wine physicochemical properties
- Features include: `alcohol`, `volatile acidity`, `sulphates`, `citric acid`, `residual sugar`, etc.
- Target: `quality` score (integer scale)

Key Feature Insights:
- **Alcohol** has the strongest positive correlation with quality.
- **Volatile acidity** negatively impacts quality.
- **Sulphates** moderately boost perceived quality.

---

## Modeling Approach

### Preprocessing
- Standardization of numerical features
- Dropping low-importance/noisy features
- Handling multicollinearity via correlation filtering

### Class Imbalance Handling
- **ADASYN** for synthetic oversampling of minority classes
- Class merging for rare labels
- Custom threshold tuning to balance precision vs recall

### Models Trained
- **Random Forest**
- **XGBoost**
- **Logistic Regression**
- **Voting Classifier (ensemble)**
- **Hybrid binary + multiclass** model pipeline

### Model Tuning
- **GridSearchCV** for hyperparameter optimization
- Evaluation using:
  - Accuracy
  - F1-score (macro & class-wise)
  - Confusion matrix
  - ROC AUC (micro & macro)

---

## Results

| Metric | Binary Model | Multiclass Model |
|--------|--------------|------------------|
| Accuracy | ~92% | ~86% |
| ROC AUC | 0.90 (binary) | 0.72–0.73 |
| Recall (Minority) | ~50% | Improved via ADASYN |
| Precision | Tuned via thresholding | Balanced via ensemble |

Threshold tuning enabled custom trade-offs:
- Lower threshold: prioritize **recall** (catch all good/bad wines)
- Higher threshold: prioritize **precision** (avoid false positives)

---

## Business Interpretation

- Wines predicted as low quality can be flagged for **manual review**—minimizing the risk of poor products reaching customers.
- Precision vs. recall trade-off allows the model to adapt to **specific quality control goals** (e.g., brand protection or yield maximization).
- Alcohol and acidity levels are actionable levers for **production teams** to fine-tune quality.

---

## Recommendations

- **Use current threshold (0.582)** if balance between recall and precision is desired.
- Adjust threshold based on business needs:
  - ↑ Threshold: fewer false positives, better brand protection
  - ↓ Threshold: higher recall, less risk of missing good wines
- Explore custom loss functions or probability calibration for future improvements.

---

## How to Run

```bash
# Clone repository
git clone <your-repo-url>
cd wine-quality-prediction

# Create virtual environment and install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook wine_Quality_prediction.ipynb
```

---

## Folder Structure

```
wine-quality-prediction-model/
│
├── data/                   # Raw dataset (CSV)
├── models/                 # Saved model binaries (pickle)
├── wine_Quality_prediction.ipynb
└── README.md
```

---

## Author

**Cynthia Chinelomma Amakeze**    
Machine Learning Enthusiast | Data-Driven Decision Maker

---

## Contact

For questions or collaborations, feel free to reach out:  
Email [nwankwocynthia26@gmail.com]  
Linkedin [[www.linkedin.com/in/cynthia-nwankwo-83278687]]
