# 📊 Sales Data Analysis & Forecasting System

## 🎯 Problem Statement
Businesses need accurate sales forecasts to plan inventory, budget, and strategy.

## 🛠️ Tech Stack
- Python, Pandas, NumPy
- Scikit-learn, Matplotlib, Seaborn
- Jupyter Notebook

## 📁 Project Structure
```
sales-forecast/
├── data/
│   ├── sales_data.csv              # Raw synthetic sales dataset
│   └── sales_data_processed.csv    # Cleaned + feature-engineered dataset
├── notebooks/
│   └── sales_forecasting.ipynb     # End-to-end analysis notebook
├── src/
│   ├── generate_data.py            # Generates the synthetic dataset
│   ├── data_processing.py          # Cleaning & feature engineering
│   ├── eda.py                      # EDA dashboards
│   ├── train_models.py             # Model training & evaluation
│   └── predict.py                  # Sample predictions with saved model
├── outputs/                         # Generated charts, models, results
├── requirements.txt
└── README.md
```

## 🔍 What This Project Does
- Generates a realistic multi-store, multi-category sales dataset (3 years, daily)
- Cleans data: handles duplicates and missing values
- Performs EDA: revenue trends, category/store comparisons, promotion effects, correlations
- Feature engineering: date features, cyclical encoding, lag features, rolling averages
- Builds regression models: **Linear Regression** and **Random Forest**
- Evaluates with **RMSE, MAE, R² Score**
- Generates visual dashboards for business insights

## 📈 Results
| Model             | RMSE  | MAE   | R²    |
|-------------------|-------|-------|-------|
| Linear Regression | ~28.0 | ~19.1 | ~0.93 |
| Random Forest     | ~23.9 | ~15.0 | ~0.95 |

Random Forest outperforms Linear Regression, capturing non-linear seasonal and promotional effects.

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate the dataset
```bash
python src/generate_data.py
```

### 3. Process data (cleaning + feature engineering)
```bash
python src/data_processing.py
```

### 4. Run EDA dashboards
```bash
python src/eda.py
```

### 5. Train and evaluate models
```bash
python src/train_models.py
```

### 6. Run sample predictions
```bash
python src/predict.py
```

### Or explore everything interactively:
```bash
jupyter notebook notebooks/sales_forecasting.ipynb
```

## 📊 Sample Outputs
All charts and model files are saved in `outputs/`:
- `monthly_revenue_trend.png`
- `sales_by_category.png`
- `revenue_by_store.png`
- `correlation_heatmap.png`
- `promotion_effect.png`
- `actual_vs_predicted_rf.png`
- `feature_importance.png`
- `model_comparison.png`
- `random_forest_model.pkl`, `linear_regression_model.pkl`

## 📝 Notes
The dataset is synthetically generated to simulate real-world sales patterns (seasonality, weekday/weekend effects, promotions, holidays, multiple stores and product categories) so the project can be run end-to-end without external data dependencies.
