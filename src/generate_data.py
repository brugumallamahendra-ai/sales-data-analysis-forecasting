"""
Generate a synthetic multi-source sales dataset for the Sales Forecasting project.
"""
import numpy as np
import pandas as pd

np.random.seed(42)

# Date range: 3 years of daily data
dates = pd.date_range(start="2022-01-01", end="2024-12-31", freq="D")

stores = ["Store_A", "Store_B", "Store_C"]
categories = ["Electronics", "Clothing", "Groceries", "Furniture", "Toys"]

rows = []
for date in dates:
    for store in stores:
        for category in categories:
            # Base demand
            base = {
                "Electronics": 120, "Clothing": 90, "Groceries": 200,
                "Furniture": 40, "Toys": 60
            }[category]

            # Seasonality - higher sales in Nov-Dec, lower in Feb
            month = date.month
            seasonal = 1.0
            if month in [11, 12]:
                seasonal = 1.6
            elif month == 2:
                seasonal = 0.8
            elif month in [6, 7]:
                seasonal = 1.2

            # Weekly pattern - weekends higher
            weekday = date.weekday()
            weekly = 1.3 if weekday >= 5 else 1.0

            # Store factor
            store_factor = {"Store_A": 1.2, "Store_B": 1.0, "Store_C": 0.8}[store]

            # Trend - slight growth over time
            days_since_start = (date - dates[0]).days
            trend = 1 + (days_since_start / len(dates)) * 0.3

            # Promotion flag (random ~10% of days)
            promotion = np.random.choice([0, 1], p=[0.9, 0.1])
            promo_boost = 1.4 if promotion else 1.0

            # Price (varies slightly by category, with small noise)
            base_price = {
                "Electronics": 250, "Clothing": 45, "Groceries": 12,
                "Furniture": 300, "Toys": 25
            }[category]
            price = round(base_price * np.random.uniform(0.9, 1.1), 2)

            # Final units sold with noise
            mean_units = base * seasonal * weekly * store_factor * trend * promo_boost
            units_sold = max(0, int(np.random.normal(mean_units, mean_units * 0.1)))

            revenue = round(units_sold * price, 2)

            rows.append({
                "date": date,
                "store": store,
                "category": category,
                "units_sold": units_sold,
                "price": price,
                "revenue": revenue,
                "promotion": promotion,
                "temperature": round(np.random.normal(25, 8), 1),  # external factor
                "holiday": 1 if (month == 12 and date.day in [24, 25, 31]) or
                                 (month == 1 and date.day == 1) else 0
            })

df = pd.DataFrame(rows)

# Introduce a few missing values and duplicates to simulate real-world messy data
mask = np.random.choice(df.index, size=200, replace=False)
df.loc[mask, "temperature"] = np.nan

dup_rows = df.sample(50, random_state=1)
df = pd.concat([df, dup_rows], ignore_index=True)

df.to_csv("data/sales_data.csv", index=False)
print(f"Generated dataset with {len(df)} rows -> data/sales_data.csv")
print(df.head())
