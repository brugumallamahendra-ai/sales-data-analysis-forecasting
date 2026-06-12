"""
Exploratory Data Analysis & Business Dashboards
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("outputs", exist_ok=True)
sns.set_style("whitegrid")

df = pd.read_csv("data/sales_data.csv", parse_dates=["date"])
df = df.drop_duplicates()
df["temperature"] = df["temperature"].fillna(df["temperature"].mean())

# 1. Revenue trend over time
monthly = df.groupby(pd.Grouper(key="date", freq="ME"))["revenue"].sum().reset_index()
plt.figure(figsize=(10, 5))
plt.plot(monthly["date"], monthly["revenue"], marker="o")
plt.title("Monthly Revenue Trend")
plt.xlabel("Date")
plt.ylabel("Revenue")
plt.tight_layout()
plt.savefig("outputs/monthly_revenue_trend.png", dpi=120)
plt.close()

# 2. Sales by category
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="category", y="units_sold", estimator=sum, errorbar=None)
plt.title("Total Units Sold by Category")
plt.tight_layout()
plt.savefig("outputs/sales_by_category.png", dpi=120)
plt.close()

# 3. Sales by store
plt.figure(figsize=(7, 5))
sns.barplot(data=df, x="store", y="revenue", estimator=sum, errorbar=None)
plt.title("Total Revenue by Store")
plt.tight_layout()
plt.savefig("outputs/revenue_by_store.png", dpi=120)
plt.close()

# 4. Weekday vs weekend sales
df["day_type"] = df["date"].dt.dayofweek.apply(lambda x: "Weekend" if x >= 5 else "Weekday")
plt.figure(figsize=(6, 5))
sns.boxplot(data=df, x="day_type", y="units_sold")
plt.title("Units Sold: Weekday vs Weekend")
plt.tight_layout()
plt.savefig("outputs/weekday_vs_weekend.png", dpi=120)
plt.close()

# 5. Correlation heatmap
numeric_cols = ["units_sold", "price", "revenue", "promotion", "temperature", "holiday"]
plt.figure(figsize=(7, 6))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png", dpi=120)
plt.close()

# 6. Effect of promotions
plt.figure(figsize=(6, 5))
sns.boxplot(data=df, x="promotion", y="units_sold")
plt.title("Units Sold: Promotion vs No Promotion")
plt.xlabel("Promotion (0 = No, 1 = Yes)")
plt.tight_layout()
plt.savefig("outputs/promotion_effect.png", dpi=120)
plt.close()

print("EDA charts saved to outputs/")
