import numpy as np
import pandas as pd
import itertools
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
import gradio as gr
import warnings
warnings.filterwarnings("ignore")
# Set random seed for reproducibility
np.random.seed(42)

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
for p, d, q, P, D, Q in itertools.product(p_values, d_values, q_values, P_values, D_values, Q_values):
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


# # Launch Gradio Interface
# iface = gr.Interface(
#     fn=forecast_turnover,
#     inputs=[
#         gr.Slider(minimum=1, maximum=6, step=1, label="Forecast Horizon (Quarters)"),
#         gr.Slider(minimum=50, maximum=99, step=1, label="Confidence Level (%)")
#     ],
#     outputs=[gr.Plot(), gr.Plot()],
#     title=f"{company} Revenue Forecast",
#     description="Select the forecast horizon (in quarters) and confidence level for revenue predictions.",
# )

# iface.launch(debug=True)


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