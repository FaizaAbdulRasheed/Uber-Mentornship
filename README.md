# 🚖 Urban Traffic Congestion Forecasting

### 📌 Project Overview
An end-to-end data science project optimizing ride-sharing logistics. By integrating **Traffic Sensors**, **Weather APIs**, and **Event Data**, this system forecasts hourly traffic volume to aid in Dynamic Pricing and Driver Allocation.

### 🛠️ Tech Stack
* **Python:** Data Integration & Feature Engineering
* **Machine Learning:** Random Forest Regressor (Scikit-Learn)
* **Dashboard:** Streamlit (Real-time inference UI)
* **Data:** Standardized Multi-variate Time Series

### 📊 Key Results
* Identified **Peak Congestion Windows** (8 AM & 6 PM).
* Quantified the impact of **Rain (>5mm)** on traffic flow reduction.
* Model Accuracy: **R² Score of ~0.75**.

### 💻 How to Run
1.  Install requirements: `pip install -r requirements.txt`
2.  Generate Models: `python generate_analysis.py`
3.  Launch Dashboard: `streamlit run app.py`