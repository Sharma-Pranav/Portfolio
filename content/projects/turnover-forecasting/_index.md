---
title: "Turnover Forecasting"
date: 2025-02-28
type: "projects"
layout: "single"
summary: "AI-powered quarterly revenue forecasting for SAP SE using SARIMA"
menu:
  main:
    parent: "projects"
    weight: 1
_build:
  list: always
  render: always
---

## Overview

This project delivers **AI-driven revenue forecasting** for **SAP SE** using a **univariate SARIMA model**. It demonstrates how accurate forecasts can be built from limited data (just historical turnover), providing actionable insights for strategic planning and growth.

## Dataset

- **Source**: Top 12 German Companies Financial Data (Kaggle)
- **Description**: Historical revenue data for SAP SE.

## Features

- Time series forecasting with the **SARIMA** model.
- Forecasts revenue for the next **1-6 quarters**.
- Interactive **Gradio** interface for model interaction.

## Tools and Libraries

- **Python**
- **Pandas, NumPy**
- **Statsmodels** (SARIMA)
- **Gradio** (for the UI)
- **Plotly** (for interactive visualizations)

## How to Run

1. Clone the repository:  
   `git clone https://github.com/Sharma-Pranav/Portfolio.git`
2. Navigate to the project directory:  
   `cd projects/TurnoverForecasting`
3. Install dependencies:  
   `pip install -r requirements.txt`
4. Run the application:  
   `python app.py`

## Results

- Accurate **quarterly revenue forecasting** for SAP SE.
- Insights on **future revenue trends** and confidence intervals.

---

📌 **Intended Use & Limitations**
- ✅ Forecast SAP SE revenue for next 1–6 quarters
- 📈 Great for univariate, seasonal time series
- 🚫 Not suitable for multivariate or non-seasonal data
- ⚠️ Requires careful preprocessing (e.g., stationarity)

👨‍💻 **Author**: Pranav Sharma
