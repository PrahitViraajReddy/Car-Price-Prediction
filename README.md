# Car Price Prediction

A supervised machine-learning project that estimates used-car resale prices from vehicle attributes. The Streamlit interface consumes a saved scikit-learn pipeline trained by `car.py`.

## Run locally

```bash
python -m pip install -r requirements.txt
python car.py
streamlit run app.py
```

Place `car data.csv` in the repository root. Training stops with a clear error if the dataset is missing or required columns are absent.

To use a specific manufacturing-year reference when creating `Car_Age`, set `CAR_REFERENCE_YEAR` before training. The default is the current calendar year at training time. The saved artifact records the chosen reference year; the app takes car age as a direct input and does not silently recompute it.

Windows PowerShell example:

```powershell
$env:CAR_REFERENCE_YEAR = "2026"
python car.py
```

## Workflow and evaluation

The script removes duplicate records, validates required columns and numeric fields, then splits the data before fitting the preprocessing pipeline. Numerical scaling and one-hot encoding are fitted only on training folds. The estimator is a Decision Tree Regressor with a fixed random seed and maximum depth of five.

The script prints training R², held-out test R², test MAE and RMSE (price units are lakhs), and five-fold cross-validation R². Run the script on the committed dataset to obtain the current metrics; scores are intentionally not hard-coded here because they depend on the exact data and training configuration.

Feature importance is labelled as Decision Tree feature importance, not a linear-regression coefficient. The actual-versus-predicted chart sorts test rows by actual price and applies the same ordering to predictions.

## Inputs

- Present showroom price (lakhs)
- Kilometres driven
- Number of previous owners
- Car age in years
- Fuel type
- Seller type
- Transmission

Predictions are estimates based on the training data, not guaranteed market valuations.

## Tests

```bash
python -m pytest -q
```
