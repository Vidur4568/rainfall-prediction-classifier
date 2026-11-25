# Rainfall Prediction Classifier: Melbourne, Australia

Predicting whether it will rain **today** in the Melbourne area using the previous day's weather, built with scikit-learn. Compares a tuned Random Forest against Logistic Regression and looks at why accuracy alone is misleading on imbalanced data.


## Results

| Model | Accuracy | Recall (rain) | Precision (rain) |
|---|---|---|---|
| Baseline: always predict "No rain" | 0.76 | 0.00 | n/a |
| Random Forest (grid search) | 0.84 | 0.51 | 0.75 |
| Logistic Regression | 0.83 | 0.51 | 0.69 |
| Logistic Regression, `class_weight='balanced'` | 0.77 | 0.74 | 0.52 |

Evaluated on a stratified 20% hold-out set (1,512 days, 358 of them rainy).

## Key takeaways

- **Accuracy flatters the model.** About 76% of days are dry, so 0.84 accuracy sounds strong, but both default models catch only about half of the rainy days (181 of 358 for the Random Forest).
- **Class balancing is a trade-off, not a fix.** It raised recall for rainy days from 0.51 to 0.74, but false alarms rose from 83 to 249 and accuracy dropped to 0.77. The better choice depends on whether a missed rain day or a false alarm costs more.
- **Humidity and pressure drive the predictions.** `Humidity3pm` was the most important feature, followed by `Pressure3pm`, `Pressure9am`, and `Sunshine`.

## Approach

1. **Data:** Daily observations from Australia's Bureau of Meteorology (2008 to 2017), via the Kaggle [Rain in Australia](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package) dataset. Dropping rows with missing values left 56,420 of 145,460 rows, and narrowing to Melbourne, Melbourne Airport, and Watsonia left 7,557 days.
2. **Leakage:** Features like `Rainfall` and `Sunshine` describe a whole day, so the problem was framed as predicting today's rain from yesterday's full record.
3. **Feature engineering:** A `Season` feature (southern hemisphere) replaced the raw date.
4. **Pipeline:** Numeric features scaled and categoricals one-hot encoded, all inside a scikit-learn `Pipeline` so cross-validation never sees held-out data.
5. **Tuning:** `GridSearchCV` with stratified 5-fold cross-validation.
6. **Evaluation:** Classification report and confusion matrix, with a focus on recall for the rain class.

## Limitations and next steps

- The train/test split is random, but weather is time-ordered. A time-based split would give a more realistic estimate.
- Dropping rows with missing values discards a lot of data. Imputation could help.
- The best Random Forest parameters sat at the edge of the search grid, so a wider search may help.
- Try threshold tuning, lag features (the last few days of pressure and humidity), and gradient boosting.

## Run it yourself

```bash
git clone https://github.com/Vidur4568/rainfall-prediction-classifier.git
cd rainfall-prediction-classifier
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/rainfall_prediction.ipynb
```

The notebook loads its data from a URL, so no local files are needed.

## Repository structure

```
notebooks/rainfall_prediction.ipynb   # full analysis
requirements.txt                      # Python dependencies
LICENSE
```

## Tech

Python, pandas, scikit-learn, matplotlib, seaborn
