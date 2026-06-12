"""
Load trained model and make a sample prediction.
"""
import pandas as pd
import joblib

model = joblib.load("outputs/random_forest_model.pkl")
feature_cols = joblib.load("outputs/feature_cols.pkl")

df = pd.read_csv("data/sales_data_processed.csv", parse_dates=["date"])

# Take last 5 rows as a sample for prediction
sample = df[feature_cols].tail(5)
preds = model.predict(sample)

for i, p in enumerate(preds):
    print(f"Sample {i+1}: Predicted units sold = {p:.1f} | Actual = {df['units_sold'].iloc[-(5-i)]}")
