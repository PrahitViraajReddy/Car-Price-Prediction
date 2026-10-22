"""Train and evaluate the used-car resale price model.

Set CAR_REFERENCE_YEAR to the manufacturing-year reference used for training.
The Streamlit app accepts car age directly, so it does not recompute this value.
"""
from datetime import date
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

DATA_PATH = Path("car data.csv")
MODEL_PATH = Path("car_price_model.pkl")
RANDOM_STATE = 42
REFERENCE_YEAR = int(os.environ.get("CAR_REFERENCE_YEAR", date.today().year))
NUMERIC_FEATURES = ["Present_Price", "Driven_kms", "Owner", "Car_Age"]
CATEGORICAL_FEATURES = ["Fuel_Type", "Selling_type", "Transmission"]
TARGET = "Selling_Price"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and validate the training dataset; fail loudly when unavailable."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path.resolve()}. "
            "Place 'car data.csv' in the project root before training."
        )
    df = pd.read_csv(path)
    required = {
        "Year", "Present_Price", "Driven_kms", "Owner", "Fuel_Type",
        "Selling_type", "Transmission", TARGET,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if df.empty:
        raise ValueError("The car dataset contains no rows.")
    df = df.drop_duplicates().copy()
    for col in ["Year", "Present_Price", "Driven_kms", "Owner", TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=list(required - {"Fuel_Type", "Selling_type", "Transmission"}))
    if df.empty:
        raise ValueError("No valid rows remain after validating numeric fields.")
    if (df[TARGET] < 0).any() or (df["Present_Price"] < 0).any() or (df["Driven_kms"] < 0).any():
        raise ValueError("Prices and driven kilometres must not be negative.")
    df["Car_Age"] = (REFERENCE_YEAR - df["Year"]).clip(lower=0)
    return df.drop(columns=["Car_Name", "Year"], errors="ignore")


def build_model() -> Pipeline:
    """Build preprocessing and estimator together to prevent train/test leakage."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE)),
    ])


def main() -> None:
    df = load_data()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    model = build_model()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"Reference year: {REFERENCE_YEAR} (set CAR_REFERENCE_YEAR to override)")
    print(f"Rows after cleaning: {len(df)}")
    print(f"Train R²: {r2_score(y_train, model.predict(X_train)):.3f}")
    print(f"Test R²: {r2_score(y_test, predictions):.3f}")
    print(f"Test MAE: {mean_absolute_error(y_test, predictions):.3f} lakhs")
    print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, predictions)):.3f} lakhs")

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
    print(f"5-fold CV R²: {cv_scores.round(3)}")
    print(f"Mean CV R²: {cv_scores.mean():.3f}")

    joblib.dump(
        {"model": model, "reference_year": REFERENCE_YEAR, "feature_columns": list(X.columns)},
        MODEL_PATH,
    )
    print(f"Saved model artifact to {MODEL_PATH.resolve()}")

    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    importances = model.named_steps["regressor"].feature_importances_
    importance = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    importance = importance.sort_values("Importance", ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance, x="Importance", y="Feature")
    plt.xlabel("Decision Tree feature importance")
    plt.title("Feature Importance — Decision Tree Regressor")
    plt.tight_layout()
    plt.show()

    residuals = y_test.to_numpy() - predictions
    plt.figure(figsize=(8, 5))
    plt.scatter(predictions, residuals)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted selling price (lakhs)")
    plt.ylabel("Residual (actual − predicted, lakhs)")
    plt.title("Residual Plot")
    plt.tight_layout()
    plt.show()

    actual = y_test.to_numpy()
    order = np.argsort(actual)
    plt.figure(figsize=(10, 6))
    plt.plot(actual[order], label="Actual price", marker="o")
    plt.plot(predictions[order], label="Predicted price", marker="x")
    plt.xlabel("Test samples sorted by actual selling price")
    plt.ylabel("Selling price (lakhs)")
    plt.title("Actual vs Predicted Car Prices")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
