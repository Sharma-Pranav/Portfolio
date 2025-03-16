# NBER-CES Manufacturing Industry Database Overview

## 1. Dataset Overview
- **Source:** NBER-CES Manufacturing Industry Database  
- **Time Span:** 1958 to 2018 (Annual Data)  
- **Region:** U.S.-based manufacturing industries (sectoral level)  
- **Industries Covered:** Over 450 manufacturing industries  
- **Key Features:** Revenue, employment, capital investment, R&D spending, productivity, and energy usage  

## 2. Data Coverage
The NBER-CES database is a comprehensive resource for analyzing various aspects of the U.S. manufacturing sector. It includes:

- **Industry Classification:**
  - **1987 SIC (Standard Industrial Classification):** 459 four-digit industries.
  - **1997 NAICS (North American Industry Classification System):** 473 six-digit industries.
  - **2012 NAICS:** 364 six-digit industries.
- **Key Variables:**
  - **Output Measures:** Value of Shipments, Value Added
  - **Input Measures:** Employment, Payroll, Cost of Materials, Energy Consumption
  - **Investment and Capital:** Capital Expenditures, Capital Stocks
  - **Productivity Metrics:** Total Factor Productivity (TFP), Labor Productivity
  - **Price Indexes:** Industry-specific price deflators

## 3. Applications
This dataset can be utilized for:
- **Economic Research:** Analyzing trends in manufacturing output, productivity, and employment.
- **Policy Analysis:** Assessing the impact of policy changes on different manufacturing industries.
- **Business Strategy:** Supporting investment, production, and resource allocation decisions.

## 4. Data Format
The dataset is available in multiple formats for ease of analysis:
- **Stata**
- **SAS**
- **Excel**
- **CSV**

## 5. Documentation
Comprehensive documentation is provided, including:
- **Variable Descriptions & Summary Statistics**: Explains each variable and its statistical properties.
- **Technical Notes**: Details methodology used in data collection and processing.
- **Industry Concordances**: Helps navigate industry classification changes over time.

## 6. Citation
If using this database, please cite:
> Becker, Randy A., Wayne B. Gray, and Jordan Marvakov. (2021). “NBER-CES Manufacturing Industry Database (1958-2018, version 2021a).” National Bureau of Economic Research.

---

## 7. Key Features for Turnover Forecasting
The dataset includes several features that are critical for turnover forecasting:

| Feature  | Description  | Importance  |
|----------|-------------|-------------|
| **VSHIP (Value of Shipments)** | Total revenue from shipments of goods | Primary revenue indicator for turnover forecasting |
| **EMP (Employment)** | Number of employees in the sector | Correlates labor with revenue growth |
| **CAPEX (Capital Expenditure)** | Investment in machinery and assets | Higher CAPEX often leads to future revenue growth |
| **ENERGY (Energy Usage)** | Power consumption in industry | Signals productivity and operational efficiency |
| **MATCOST (Materials Cost)** | Cost of raw materials used | Higher costs may impact profit margins and revenue |
| **RD (R&D Expenditure)** | Investment in innovation and new tech | High R&D leads to long-term revenue growth |
| **WAGE (Wages)** | Total wages paid in the industry | Useful for modeling cost-revenue relationships |
| **PROD (Productivity)** | Output per worker or per machine | Efficiency metric to predict revenue shifts |

---

## 8. Additional Variables for Turnover Forecasting

| Column | Description | Importance |
|--------|-------------|------------|
| **NAICS** | NAICS Industry Code | Unique identifier for each industry classification |
| **Year** | Year of observation | Used for time-series analysis and forecasting trends |
| **PRODE** | Productivity (Output per employee) | Measures efficiency; affects turnover growth |
| **PRODH** | Productivity (Output per hour worked) | Higher values indicate better labor efficiency |
| **PRODW** | Productivity (Output per wage dollar) | Helps in measuring cost efficiency |
| **VADD** | Value Added (Revenue - Input Costs) | Represents economic contribution of the industry |
| **INVEST** | Investment (Capital Expenditure - CAPEX) | High investments often lead to future revenue growth |
| **INVENT** | Inventory Levels | Impacts supply chain and demand forecasting |
| **ENERGY** | Energy Costs | Higher energy costs reduce profit margins |
| **CAP** | Capital Stock | Total capital assets; influences production capacity |
| **EQUIP** | Equipment Stock | Investment in machinery; affects manufacturing output |
| **PLANT** | Plant Stock | Investment in physical infrastructure |
| **PISHIP** | Price Index for Shipments | Adjusts revenue for inflation effects |
| **PIMAT** | Price Index for Materials | Adjusts material costs for inflation |
| **PIINV** | Price Index for Inventory | Adjusts inventory value for inflation |
| **PIEN** | Price Index for Energy | Adjusts energy costs for inflation |
| **DTFP5** | Δ Total Factor Productivity (5-factor model) | Measures efficiency improvements over time |
| **TFP5** | Total Factor Productivity (5-factor model) | Higher values indicate better overall efficiency |
| **DTFP4** | Δ Total Factor Productivity (4-factor model) | Alternative measure of productivity growth |
| **TFP4** | Total Factor Productivity (4-factor model) | Measures multi-factor efficiency |

---
