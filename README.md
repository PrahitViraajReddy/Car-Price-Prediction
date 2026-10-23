# 🚗 Car Price Prediction

**A supervised machine-learning application that estimates used-car resale prices from vehicle attributes.**

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white">
  <img alt="Machine Learning" src="https://img.shields.io/badge/ML-Decision%20Tree-orange?style=flat-square">
  <img alt="scikit-learn" src="https://img.shields.io/badge/Library-scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white">
  <img alt="Streamlit" src="https://img.shields.io/badge/App-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white">
</p>

**[View the source code](https://github.com/PrahitViraajReddy/Car-Price-Prediction)**

---

## What is Car Price Prediction?

Car Price Prediction is a machine-learning project that estimates a used car's resale price from its vehicle details. It combines data validation, preprocessing, model training and evaluation, and a Streamlit interface for entering vehicle attributes and viewing a prediction.

The output is an estimate learned from the training data, not a guaranteed market valuation or a substitute for a professional vehicle inspection.

## Features

- **Vehicle price estimation** — predict an estimated resale price from selected vehicle attributes.
- **Data validation** — check required columns and numeric values before training.
- **Preprocessing pipeline** — scale numerical inputs and one-hot encode categorical features.
- **Decision Tree regression** — train a bounded-depth Decision Tree Regressor.
- **Model evaluation** — report R², MAE, RMSE, and cross-validation R².
- **Feature importance** — inspect Decision Tree feature importance.
- **Prediction visualization** — compare actual and predicted prices on held-out test data.
- **Interactive interface** — use the Streamlit application to submit vehicle details.

## How it works

```text
Used-car dataset
       │
       ▼
Data validation and duplicate handling
       │
       ▼
Train/test split
       │
       ▼
Preprocessing pipeline
(numerical scaling + one-hot encoding)
       │
       ▼
Decision Tree Regressor
       │
       ├──────────────► Evaluation metrics and plots
       │
       └──────────────► Saved model pipeline
                              │
                              ▼
                       Streamlit application
                              │
                              ▼
                       Estimated car price
```

The preprocessing pipeline is fitted using training data to reduce data leakage. The application uses the saved scikit-learn pipeline produced by the training script.

## Prediction inputs

The model uses these vehicle attributes:

| Input | Description |
|---|---|
| Present showroom price | Current showroom price, in lakhs |
| Kilometres driven | Distance driven by the car |
| Previous owners | Number of previous owners |
| Car age | Age of the car, in years |
| Fuel type | Fuel category |
| Seller type | Individual or dealer category represented by the dataset |
| Transmission | Transmission category |

Predictions depend on the training data and the values provided. Unusual or out-of-range inputs may produce less reliable estimates.

## Machine learning and evaluation

**Model:** Decision Tree Regressor with a maximum depth of five and a fixed random seed.

The training workflow removes duplicate records, validates required columns and numeric fields, splits the data, and fits numerical scaling and categorical encoding within the training pipeline.

Evaluation includes:

| Metric | What it measures |
|---|---|
| Training R² | Fit on the training data |
| Test R² | Explained variance on held-out data |
| MAE | Mean absolute prediction error |
| RMSE | Root mean squared prediction error |
| Cross-validation R² | Performance across five validation folds |

Price errors are expressed in the dataset's price units (lakhs). Metrics are not hard-coded here; run the training script to calculate them for the current dataset and configuration.

The project also generates feature-importance and actual-versus-predicted visualizations. Feature importance is specific to the fitted Decision Tree and should not be interpreted as a causal effect.

## Technology stack

| Layer | Technologies |
|---|---|
| Language | Python |
| Data processing | Pandas, NumPy |
| Machine learning | scikit-learn |
| Regression model | Decision Tree Regressor |
| Preprocessing | Numerical scaling, one-hot encoding |
| Evaluation | R², MAE, RMSE, cross-validation |
| Web interface | Streamlit |
| Visualization | Matplotlib |

## Run locally

### 1. Clone the repository

```bash
git clone https://github.com/PrahitViraajReddy/Car-Price-Prediction.git
cd Car-Price-Prediction
```

### 2. Prepare the dataset

Place `car data.csv` in the repository root. Training requires the dataset and the expected columns; it stops with a clear error if the data is missing or the required columns are absent.

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Train the model

```bash
python car.py
```

To set the manufacturing-year reference used to create `Car_Age`, configure `CAR_REFERENCE_YEAR` before training.

**Windows PowerShell**
```powershell
$env:CAR_REFERENCE_YEAR = "2026"
python car.py
```

The default reference year is the current calendar year at training time. The saved artifact records the chosen reference year; the app accepts car age directly and does not silently recalculate it.

### 5. Launch the application

```bash
streamlit run app.py
```

### 6. Run tests

```bash
python -m pytest -q
```

## Repository structure

```text
Car-Price-Prediction/
├── app.py
├── car.py
├── requirements.txt
├── car data.csv
└── README.md
```

This is a high-level overview; generated model artifacts, tests, and additional files may also be present in the repository.

## Engineering highlights

- Data validation and duplicate handling
- Train/test separation before fitting preprocessing
- Reusable scikit-learn preprocessing and prediction pipeline
- Regression evaluation with multiple metrics
- Feature-importance and actual-versus-predicted plots
- Explicit manufacturing-year reference for car-age construction
- Interactive model inference through Streamlit

## Links

- **Live app:** https://car-price-prediction-jxco5kdfo4uqgcxyeu7ack.streamlit.app/
- **GitHub repository:** https://github.com/PrahitViraajReddy/Car-Price-Prediction
- **Author:** [Madupu Prahit Viraaj Reddy](https://github.com/PrahitViraajReddy)
- **LinkedIn:** [Prahit Viraaj Reddy](https://www.linkedin.com/in/prahit-viraaj-reddy-madupu-5169332ba/)

---

<div align="center">

Built with Python, pandas, and scikit-learn.

</div>
