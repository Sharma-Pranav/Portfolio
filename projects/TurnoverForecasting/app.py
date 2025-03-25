import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
import gradio as gr
from skops import hub_utils
from tempfile import mkdtemp
from pathlib import Path
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

# Set random seed
np.random.seed(42)

# Get token from environment variable
token = os.getenv("HF_TOKEN")

# Load the dataset
df = pd.read_csv("data/Top_12_German_Companies_Financial_Data.csv")
companies = np.unique(df.Company)
company = companies[9]
print(f"Company: {company}")

# Filter for the selected company
df = df[df["Company"] == company].copy()
df["Period"] = pd.to_datetime(df["Period"], format="%m/%d/%Y")
df = df.sort_values(by="Period")
df.set_index("Period", inplace=True)
df["Revenue"] = pd.to_numeric(df["Revenue"], errors="coerce")
series = df['Revenue']

# Train-validation-test split
train_idx = int(len(series) * 0.8)
val_idx = int(len(series) * 0.9)
train, val, test = series[:train_idx], series[train_idx:val_idx], series[val_idx:]

# Define parameter ranges for SARIMA tuning
p_values, d_values, q_values = range(0, 6), range(0, 3), range(0, 6)
P_values, D_values, Q_values = range(0, 3), range(0, 2), range(0, 3)
S = 12  # Quarterly seasonality

best_score, best_cfg = float("inf"), None

# Grid search over SARIMA parameter combinations
for p, d, q, P, D, Q in  tqdm(itertools.product(p_values, d_values, q_values, P_values, D_values, Q_values)):
    try:
        model = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, S), enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit(disp=False)
        predictions = model_fit.forecast(steps=len(val))
        error = mean_absolute_error(val, predictions)
        if error < best_score:
            best_score, best_cfg = error, (p, d, q, P, D, Q)
    except:
        continue

# Train best SARIMA model
best_p, best_d, best_q, best_P, best_D, best_Q = best_cfg
final_model = SARIMAX(pd.concat([train, val]), order=(best_p, best_d, best_q), seasonal_order=(best_P, best_D, best_Q, S), enforce_stationarity=False, enforce_invertibility=False, initialization="approximate_diffuse")
final_model_fit = final_model.fit(disp=False)


# Train on full dataset for next year prediction
full_model = SARIMAX(series, order=(best_p, best_d, best_q), seasonal_order=(best_P, best_D, best_Q, S), enforce_stationarity=False, enforce_invertibility=False, initialization="approximate_diffuse")
full_model_fit = full_model.fit(disp=False)



def forecast_turnover(horizon, confidence_level):
    try:
        horizon = int(horizon)
        alpha_value = 1 - (confidence_level / 100)  # Convert % to alpha
        predictions_result = final_model_fit.get_forecast(steps=horizon)
        final_predictions = predictions_result.predicted_mean
        conf_int = predictions_result.conf_int(alpha=alpha_value)

        last_date = test.index.min()
        future_dates = pd.date_range(start=last_date, periods=horizon, freq="Q")

        # Create interactive Plotly plot
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=train.index, y=train, mode='lines', name='Training Data', line=dict(color='blue')))
        fig1.add_trace(go.Scatter(x=val.index, y=val, mode='lines', name='Validation Data', line=dict(color='orange')))
        fig1.add_trace(go.Scatter(x=test.index, y=test, mode='lines+markers', name='Test Data', line=dict(color='green')))
        fig1.add_trace(go.Scatter(x=future_dates, y=final_predictions, mode='lines+markers', name=f'Forecast ({confidence_level}%)', line=dict(color='red', dash='dash')))

        # Confidence interval fill
        fig1.add_trace(go.Scatter(
            x=future_dates.tolist() + future_dates[::-1].tolist(),
            y=conf_int.iloc[:, 0].tolist() + conf_int.iloc[:, 1].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=True,
            name=f'Confidence Interval ({confidence_level}%)'
        ))

        fig1.update_layout(title=f"SARIMA Forecast for {company} Revenue", xaxis_title="Year", yaxis_title="Revenue", hovermode='x')

        # Predict next year using full model
        next_year_result = full_model_fit.get_forecast(steps=4)
        next_year_predictions = next_year_result.predicted_mean
        next_year_conf_int = next_year_result.conf_int(alpha=alpha_value)
        next_year_dates = pd.date_range(start=series.index.max(), periods=4, freq="Q")

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=series.index, y=series, mode='lines', name='Full Data', line=dict(color='blue')))
        fig2.add_trace(go.Scatter(x=next_year_dates, y=next_year_predictions, mode='lines+markers', name='Next Year Forecast', line=dict(color='red', dash='dash')))
        fig2.add_trace(go.Scatter(
            x=next_year_dates.tolist() + next_year_dates[::-1].tolist(),
            y=next_year_conf_int.iloc[:, 0].tolist() + next_year_conf_int.iloc[:, 1].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=True,
            name=f'Confidence Interval ({confidence_level}%)'
        ))

        fig2.update_layout(title=f"SARIMA Forecast for {company} Revenue for 2025", xaxis_title="Year", yaxis_title="Revenue", hovermode='x')

        return fig1, fig2
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


# Save model to a temporary path
model_path = "sarima_sap_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(full_model_fit, f)

# Hugging Face push (moved up to run before Gradio launch)
base_temp_dir = Path(mkdtemp(prefix="sarima-sap-hf-"))
hf_repo_path = base_temp_dir / "hf_repo"
hf_repo_path.mkdir(parents=True, exist_ok=True)

data = df.reset_index()
data["Period"] = data["Period"].astype(str)

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
👍 Forecast SAP SE revenue for next 1–6 quarters  
📈 Great for univariate, seasonal time series  
🚫 Not suitable for multivariate or non-seasonal data  
⚠️ Requires careful preprocessing (e.g., stationarity)

👨‍💻 Author: Pranav Sharma
"""

with open(readme_path, "w") as f:
    f.write(readme_content)

hub_utils.push(
    repo_id="PranavSharma/turnover-forecasting-model",
    source=hf_repo_path,
    commit_message="📈 Pushed SARIMA model and card for SAP SE",
    create_remote=True,
    token=token  # Pass the token for authentication
)

print("pushed to HF Hub")

with gr.Blocks() as demo:
    gr.Markdown(f"# {company} Revenue Forecast")
    gr.Markdown("📈 Select the forecast horizon (in quarters) and confidence level for revenue predictions.")

    with gr.Column():
        horizon = gr.Slider(minimum=1, maximum=6, step=1, label="Forecast Horizon (Quarters)")
        confidence = gr.Slider(minimum=50, maximum=99, step=1, label="Confidence Level (%)")
        submit = gr.Button("📊 Forecast")

        plot1 = gr.Plot(label="Validation & Forecast")
        plot2 = gr.Plot(label="Full Data & 2025 Forecast")

        def wrapped_forecast(h, c):
            return forecast_turnover(h, c)

        submit.click(fn=wrapped_forecast, inputs=[horizon, confidence], outputs=[plot1, plot2])

demo.launch(debug=True)