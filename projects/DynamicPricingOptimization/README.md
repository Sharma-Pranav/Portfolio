---
title: RidePricingInsightEngine
emoji: 🚗
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.22.0"
app_file: app.py
pinned: false
license: mit
short_description: ML-powered ride fare prediction using regression
---

# Dynamic Ride Pricing Optimization

## 🚀 Overview

This project presents a machine learning-based solution for **ride fare prediction and optimization**, using a rich synthetic dataset of historical ride data. By analyzing real-time supply-demand conditions and customer attributes, the system aims to help ride-sharing platforms implement **data-driven dynamic pricing strategies**.

## 📊 Dataset

- **Source**: [Dynamic Pricing Dataset (Kaggle)](https://www.kaggle.com/datasets/arashnic/dynamic-pricing-dataset)
- **Description**: A **synthetic dataset** designed for ride-sharing fare prediction. Features include:
  - Number of Riders / Drivers
  - Location Category
  - Loyalty Status
  - Number of Past Rides
  - Ratings
  - Booking Time
  - Vehicle Type
  - Expected Ride Duration
  - Historical Cost of Ride

## 🔍 Problem Statement

The objective is to identify which features most influence ride fares and approximate pricing behavior through various **regression models**. This includes:
- Finding key predictors for optimal fare setting.
- Estimating price sensitivity across rider and supply characteristics.
- Offering recommendations for **dynamic fare adjustments** in real time.

## 🧠 Model & Features

- **Regression-based modeling** (e.g., Linear, Ridge, XGBoost)
- **Final model**: Huber Regressor with non-zero intercept — robust to outliers, encouraging sparse feature usage
- **Feature selection pipeline** to reduce unnecessary variables
- **Price approximation engine** that generalizes across different booking conditions
- **Gradio-powered UI** for hands-on exploration

## 🛠️ Tools and Libraries

- **Python**
- **Scikit-learn**, **XGBoost**
- **Pandas**, **NumPy**
- **Gradio** for app deployment
- **Plotly** for interactive visualization

## 🧪 How to Run Locally

```bash
git clone https://github.com/Sharma-Pranav/Portfolio.git
cd Portfolio/projects/DynamicPricingOptimization
pip install -r requirements.txt
python app.py
```

## 📈 Results

- Clear insights into **fare-driving features** like rider demand, loyalty, and time of day.
- **Huber Regressor** outperformed others due to robustness and minimal feature reliance.
- **Optimal pricing zones** visualized for strategic recommendations.
- Portable deployment for teams to simulate and iterate pricing strategies.

---

> ✨ Developed by **Pranav Sharma** | 🚀 Hugging Face Space: [`RidePricingInsightEngine`](https://huggingface.co/spaces/PranavSharma/RidePricingInsightEngine)
