import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM
from neuralforecast.losses.pytorch import MAE

# 📌 Load Data
df = pd.read_csv("data/Top_12_German_Companies_Financial_Data.csv")  # Organized in `data/` folder
df = df[df["Company"] == "Merck KGaA"].copy()
# added test comment
print("Data Loaded Successfully!")
df["Period"] = pd.to_datetime(df["Period"], format="%m/%d/%Y")

df = df.sort_values(by="Period")

df["Revenue"] = pd.to_numeric(df["Revenue"], errors="coerce")
df = df.rename(columns={"Period": "ds", "Revenue": "y"})
df["unique_id"] = "all"
df = df[["unique_id", "ds", "y"]]

# Train-Test Split
val_size = int(len(df) * 0.2)
train, val = df[:-val_size], df[-val_size:]

# Train LSTM Model
hidden_neurons = 12
input_size = 8
best_model = LSTM(h=hidden_neurons, input_size=input_size, loss=MAE(), alias="LSTM_12")

nf_best = NeuralForecast(models=[best_model], freq="Q")
nf_best.fit(df=train)

# Forecast Function
def forecast_turnover(horizon):
    horizon = int(horizon)
    forecast_df = nf_best.predict().reset_index()

    if forecast_df.empty:
        return None, "🚨 No forecast data available!"

    pred_col = "LSTM_12"
    predictions = forecast_df[pred_col][:horizon].values

    last_date = train["ds"].max()
    future_dates = pd.date_range(start=last_date, periods=horizon, freq="QE-DEC")

    # 📊 Create Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train["ds"], y=train["y"], mode="lines", name="Training Data"))
    fig.add_trace(go.Scatter(x=val["ds"], y=val["y"], mode="lines", name="Actual Revenue"))
    fig.add_trace(go.Scatter(x=future_dates, y=predictions, mode="lines+markers", name="Forecast"))

    fig.update_layout(title="Turnover Forecast (Merck KGaA)", xaxis_title="Date", yaxis_title="Revenue (€)", template="plotly_white")

    return fig, f"✅ Forecast for {horizon} quarters."

# Gradio Interface
iface = gr.Interface(
    fn=forecast_turnover,
    inputs=gr.Slider(minimum=1, maximum=6, step=1, label="Forecast Horizon (Quarters)"),
    outputs=[gr.Plot(), gr.Textbox()],
    title="Merck KGaA Turnover Forecast",
    description="Select the forecast horizon (in quarters) to generate turnover predictions.",
)

iface.launch()
