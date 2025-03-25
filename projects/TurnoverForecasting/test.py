import numpy as np
import pandas as pd
import itertools
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
import gradio as gr
from huggingface_hub import HfApi
from skops import hub_utils
# from skops.card import CardData, Card
from collections import OrderedDict
from tempfile import mkdtemp
from pathlib import Path
import pickle
import shutil
import warnings
warnings.filterwarnings("ignore")

# Set random seed
np.random.seed(42)

# Load dataset
df = pd.read_csv("data/Top_12_German_Companies_Financial_Data.csv")
company = "SAP SE"
print(f"Company: {company}")
df = df[df["Company"] == company].copy()
df["Period"] = pd.to_datetime(df["Period"], format="%m/%d/%Y")
df.sort_values(by="Period", inplace=True)
df.set_index("Period", inplace=True)
df["Revenue"] = pd.to_numeric(df["Revenue"], errors="coerce")
series = df["Revenue"]

# Train-validation-test split
train_idx = int(len(series) * 0.8)
val_idx = int(len(series) * 0.9)
train, val, test = series[:train_idx], series[train_idx:val_idx], series[val_idx:]

# SARIMA tuning
p_values, d_values, q_values = range(0, 6), range(0, 3), range(0, 6)
P_values, D_values, Q_values = range(0, 3), range(0, 2), range(0, 3)
S = 12
best_score, best_cfg = float("inf"), None

for p, d, q, P, D, Q in itertools.product(p_values, d_values, q_values, P_values, D_values, Q_values):
    try:
        model = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, S),
                        enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit(disp=False)
        pred = model_fit.forecast(steps=len(val))
        error = mean_absolute_error(val, pred)
        if error < best_score:
            best_score, best_cfg = error, (p, d, q, P, D, Q)
    except:
        continue

# Train on full data
best_p, best_d, best_q, best_P, best_D, best_Q = best_cfg
full_model = SARIMAX(series, order=(best_p, best_d, best_q),
                     seasonal_order=(best_P, best_D, best_Q, S),
                     enforce_stationarity=False, enforce_invertibility=False,
                     initialization="approximate_diffuse")
full_model_fit = full_model.fit(disp=False)

# Save model to a temporary path
model_path = "sarima_sap_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(full_model_fit, f)

# Create base temp folder
base_temp_dir = Path(mkdtemp(prefix="sarima-sap-hf-"))

# Define a subfolder where `init()` will build the repo
hf_repo_path = base_temp_dir / "hf_repo"

data = df.reset_index()
data["Period"] = data["Period"].astype(str)  # Convert datetime to str

hub_utils.init(
    model=Path(model_path),
    requirements=["pandas", "statsmodels", "scikit-learn"],
    dst=hf_repo_path,
    task="tabular-regression",
    data=data
)
readme_path = hf_repo_path / "README.md"
readme_content = f"""---
title: TurnoverForecasting
emoji: 📊
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 5.22.0
app_file: app.py
pinned: false
license: mit
short_description: Forecasting SAP SE Revenue with AI
---

# 📊 AI-Powered Turnover Forecasting for SAP SE

## 🚀 Project Overview

This project delivers **AI-driven revenue forecasting** for **SAP SE** using a **univariate SARIMA model**.
It shows how accurate forecasts can be built from limited data (just historical turnover).

---

## 🏢 Why SAP SE?

- SAP SE is a **global leader in enterprise software**
- Revenue forecasts support **strategic planning & growth**
- Perfect case for **AI-powered financial forecasting**

---

## 🧠 Model Details

- **Model type**: SARIMA (Seasonal ARIMA)
- **Trained on**: SAP SE revenue from Top 12 German Companies Dataset (Kaggle)
- **SARIMA Order**: ({best_p}, {best_d}, {best_q})
- **Seasonal Order**: ({best_P}, {best_D}, {best_Q}, {S})
- **Evaluation Metric**: MAE (Mean Absolute Error)
- **Validation**: Walk-forward validation with test set (last 10%)

---

## ⚙️ How to Use

```python
import pickle

with open("sarima_sap_model.pkl", "rb") as f:
    model = pickle.load(f)

forecast = model.forecast(steps=4)
print(forecast)
```

## 📌 Intended Use & Limitations
✅ Forecast SAP SE revenue for next 1–6 quarters  
📈 Great for univariate, seasonal time series  
🚫 Not suitable for multivariate or non-seasonal data  
⚠️ Requires careful preprocessing (e.g., stationarity)

👨‍💻 Author: Pranav Sharma
"""
# Save the card
with open(readme_path, "w") as f:
    f.write(readme_content)

# Now push to HF Hub
hub_utils.push(
    repo_id="PranavSharma/turnover-forecasting-model",
    source=hf_repo_path,
    commit_message="📈 Pushed SARIMA model and card for SAP SE",
    create_remote=True,
)

print("✅ Model pushed successfully to Hugging Face Hub!")
