"""
Train and evaluate regression models for sales forecasting.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

os.makedirs("outputs", exist_ok=True)

# ---------------------------------------------------------
# Load processed data
# ---------------------------------------------------------
df = pd.read_csv("data/sales_data_processed.csv", parse_dates=["date"])

target = "units_sold"
drop_cols = ["date", target, "revenue"]  # revenue leaks target info
feature_cols = [c for c in df.columns if c not in drop_cols]

X = df[feature_cols]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------
# Scale features (for Linear Regression)
# ---------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# Train models
# ---------------------------------------------------------
results = {}

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

rf = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)


def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    results[name] = {"RMSE": rmse, "MAE": mae, "R2": r2}
    print(f"{name:20s} RMSE={rmse:8.3f}  MAE={mae:8.3f}  R2={r2:.4f}")


print("Model Evaluation:")
evaluate("Linear Regression", y_test, y_pred_lr)
evaluate("Random Forest", y_test, y_pred_rf)

results_df = pd.DataFrame(results).T
results_df.to_csv("outputs/model_results.csv")

# ---------------------------------------------------------
# Save models
# ---------------------------------------------------------
joblib.dump(lr, "outputs/linear_regression_model.pkl")
joblib.dump(rf, "outputs/random_forest_model.pkl")
joblib.dump(scaler, "outputs/scaler.pkl")
joblib.dump(feature_cols, "outputs/feature_cols.pkl")

# ---------------------------------------------------------
# Visualizations
# ---------------------------------------------------------
sns.set_style("whitegrid")

# 1. Actual vs Predicted (Random Forest)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.3, s=10)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual Units Sold")
plt.ylabel("Predicted Units Sold")
plt.title("Random Forest: Actual vs Predicted")
plt.tight_layout()
plt.savefig("outputs/actual_vs_predicted_rf.png", dpi=120)
plt.close()

# 2. Feature importance (Random Forest)
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False).head(15)
plt.figure(figsize=(8, 6))
sns.barplot(x=importances.values, y=importances.index)
plt.title("Top 15 Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png", dpi=120)
plt.close()

# 3. Model comparison bar chart
plt.figure(figsize=(7, 5))
results_df[["RMSE", "MAE"]].plot(kind="bar", figsize=(7, 5))
plt.title("Model Comparison: RMSE & MAE")
plt.ylabel("Error")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("outputs/model_comparison.png", dpi=120)
plt.close()

print("\nSaved models and plots to outputs/")
