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

# 📊 TurnoverForecasting – AI Revenue Forecasting for SAP SE

## 🔄 Overview
This project builds an **AI-powered turnover forecasting system** for **SAP SE**, using a **univariate SARIMA model**. It demonstrates how reliable forecasts can be generated from minimal data — only historical revenue — making it ideal for early-stage AI adoption, SMEs, and strategic financial planning.

---

## 📂 Dataset
**🔗 Source:** [Top 12 German Companies Financial Data (Kaggle)](https://www.kaggle.com/datasets)  
**📅 Description:** A financial dataset focused on German enterprises, with historical turnover values for SAP SE used to train and validate the forecasting model.

---

## 🗃️ Repository & Deployment
**🔗 GitHub Repository:** [View on GitHub](https://github.com/Sharma-Pranav/Portfolio/tree/main/projects/TurnoverForecasting)  
**🚀 Live Demo:** [Try on Hugging Face](https://huggingface.co/spaces/PranavSharma/TurnoverForecasting)

---

## ✨ Features
- 🗕️ Accurate revenue forecasts up to **6 quarters ahead**  
- 🎯 Dynamic controls for **forecast horizon** and **confidence intervals**  
- 🧠 Clean, **Gradio-based interactive dashboard**  
- 📱 Mobile-friendly single-column layout  
- 📈 Insightful visuals: Training, Validation, Test & Forecasts  
- 🧩 Ideal for strategic planning, budgeting, and executive reporting

---

## 🛠️ Tools and Libraries
- **Language:** Python  
- **Libraries:** `pandas`, `numpy`, `statsmodels`, `plotly`  
- **Deployment:** `gradio`, hosted on **Hugging Face Spaces**

---

## 🔧 How to Run Locally

```bash
# Clone the repository
git clone https://github.com/Sharma-Pranav/Portfolio.git

# Navigate to the project directory
cd projects/TurnoverForecasting

# Install dependencies
pip install -r requirements.txt

# Run the Gradio app
python app.py
```

---

## 📊 Results
- 📊 Reliable quarterly forecasts of SAP SE revenue  
- ✅ Model validated using **walk-forward validation**  
- 📉 Clear visualization of historical vs. forecasted revenue  
- 💼 Actionable insights for **financial strategy and planning**
````

