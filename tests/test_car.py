"""Regression tests for car-price data validation and preprocessing."""
import pandas as pd
import pytest

import car


def test_missing_dataset_fails_with_nonzero_exception(tmp_path):
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        car.load_data(tmp_path / "missing.csv")


def test_load_data_drops_duplicates_and_builds_car_age(tmp_path, monkeypatch):
    source = tmp_path / "cars.csv"
    pd.DataFrame({
        "Car_Name": ["A", "A"],
        "Year": [2020, 2020],
        "Present_Price": [5.0, 5.0],
        "Driven_kms": [10000, 10000],
        "Owner": [0, 0],
        "Fuel_Type": ["Petrol", "Petrol"],
        "Selling_type": ["Dealer", "Dealer"],
        "Transmission": ["Manual", "Manual"],
        "Selling_Price": [4.0, 4.0],
    }).to_csv(source, index=False)
    monkeypatch.setattr(car, "REFERENCE_YEAR", 2026)
    result = car.load_data(source)
    assert len(result) == 1
    assert result.iloc[0]["Car_Age"] == 6
    assert "Year" not in result.columns
    assert "Car_Name" not in result.columns


def test_missing_required_column_is_reported(tmp_path):
    source = tmp_path / "cars.csv"
    pd.DataFrame({"Year": [2020], "Selling_Price": [4]}).to_csv(source, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        car.load_data(source)


def test_pipeline_ignores_unseen_categories():
    model = car.build_model()
    X = pd.DataFrame({
        "Present_Price": [5.0, 8.0, 12.0, 4.0],
        "Driven_kms": [1000, 20000, 40000, 8000],
        "Owner": [0, 1, 0, 2],
        "Car_Age": [2, 5, 7, 3],
        "Fuel_Type": ["Petrol", "Diesel", "Petrol", "Diesel"],
        "Selling_type": ["Dealer", "Individual", "Dealer", "Individual"],
        "Transmission": ["Manual", "Automatic", "Manual", "Automatic"],
    })
    y = pd.Series([4.0, 6.0, 8.0, 3.0])
    model.fit(X, y)
    prediction = model.predict(pd.DataFrame({
        "Present_Price": [6.0], "Driven_kms": [12000], "Owner": [0],
        "Car_Age": [4], "Fuel_Type": ["Electric"],
        "Selling_type": ["Dealer"], "Transmission": ["Automatic"],
    }))
    assert len(prediction) == 1
    assert pd.notna(prediction[0])
