"""
Data Cleaning & Feature Engineering for Sales Forecasting
"""
import pandas as pd
import numpy as np

def load_and_clean(path="data/sales_data.csv"):
    df = pd.read_csv(path, parse_dates=["date"])

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"Removed {before - len(df)} duplicate rows")

    # Handle missing values - fill temperature with category/store-wise mean
    df["temperature"] = df["temperature"].fillna(df["temperature"].mean())

    # Remove any negative or invalid units_sold
    df = df[df["units_sold"] >= 0]

    return df


def engineer_features(df):
    df = df.copy()

    # Date-based features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["quarter"] = df["date"].dt.quarter
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    # Cyclical encoding for month and day_of_week
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Lag features (per store + category)
    df = df.sort_values(["store", "category", "date"])
    df["units_sold_lag_1"] = df.groupby(["store", "category"])["units_sold"].shift(1)
    df["units_sold_lag_7"] = df.groupby(["store", "category"])["units_sold"].shift(7)

    # Rolling average (7-day)
    df["units_sold_roll_mean_7"] = (
        df.groupby(["store", "category"])["units_sold"]
        .transform(lambda x: x.shift(1).rolling(7).mean())
    )

    # Drop rows with NaNs created by lag/rolling features
    df = df.dropna().reset_index(drop=True)

    # Encode categoricals
    df = pd.get_dummies(df, columns=["store", "category"], drop_first=True)

    return df


if __name__ == "__main__":
    df = load_and_clean()
    df = engineer_features(df)
    df.to_csv("data/sales_data_processed.csv", index=False)
    print(f"Processed dataset saved: {df.shape}")
    print(df.columns.tolist())
