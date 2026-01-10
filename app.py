import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Page Config
st.set_page_config(page_title="Urban Traffic Engine", page_icon="🚦", layout="wide")

# Custom CSS for Cards
st.markdown("""
    <style>
    .main {background-color: #f0f2f6;}
    h1 {color: #2c3e50;}
    
    /* CSS to turn metrics into dark cards with white text */
    [data-testid="stMetric"] {
        background-color: #2c3e50; /* Dark Blue Background */
        color: #ffffff; /* White Text */
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        border: 1px solid #34495e;
    }
    
    /* Force labels to be white */
    [data-testid="stMetricLabel"] {
        color: #ecf0f1 !important;
        font-weight: 600;
        font-size: 1rem;
    }
    
    /* Force values to be white and large */
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.8rem;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("🚦 Urban Traffic Congestion Forecasting")
st.markdown("A Machine Learning system integrating **Traffic Sensors**, **Weather Data**, and **Event Calendars** to predict city congestion.")

# Load Data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('traffic_data.csv')
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df['Hour'] = df['DateTime'].dt.hour
        df['DayOfWeek'] = df['DateTime'].dt.day_name()
        return df
    except FileNotFoundError:
        return None

df = load_data()
if df is None:
    st.error("Error: 'traffic_data.csv' not found. Please place the file in the directory.")
    st.stop()

# Sidebar
st.sidebar.title("🔧 Controls")
junction_opt = st.sidebar.selectbox("Select Junction", df['Junction'].unique())

# Filter Data
filtered = df[df['Junction'] == junction_opt]

# --- Top KPIs (Key Performance Indicators) ---
st.markdown("### 📈 Key Metrics Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Traffic Records", f"{len(filtered):,}")
col2.metric("Avg Traffic (Std)", f"{filtered['Vehicles'].mean():.2f}")
col3.metric("Busiest Hour (Hist)", f"{filtered.groupby('Hour')['Vehicles'].mean().idxmax()}:00")
col4.metric("Max Temp Recorded", f"{filtered['Temperature_C'].max()} °C")

# --- Explanation Section ---
with st.expander("ℹ️ What do these values represent?"):
    st.markdown("""
    * **Total Traffic Records:** The total number of hourly data points collected for the selected junction.
    * **Avg Traffic (Std):** The average traffic volume. This is a **standardized score** (Z-score).
        * `0.0` = Exactly average traffic.
        * `> 0.0` = Higher than average (Congestion building).
        * `< 0.0` = Lower than average (Free flow).
    * **Busiest Hour (Hist):** The hour of the day that has historically had the highest average traffic volume.
    * **Max Temp Recorded:** The highest temperature observed in the dataset for this location.
    """)

# --- Visualizations ---
st.markdown("---")
st.markdown("### 📊 Live Traffic Analysis")
c1, c2 = st.columns(2)

with c1:
    st.caption("Hourly Congestion Patterns")
    fig, ax = plt.subplots(figsize=(10, 5))
    hourly = filtered.groupby('Hour')['Vehicles'].mean()
    sns.lineplot(x=hourly.index, y=hourly.values, marker='o', color='#E74C3C', ax=ax)
    ax.set_ylabel("Standardized Volume")
    ax.set_xlabel("Hour of Day (0-23)")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

with c2:
    st.caption("Traffic Density by Day of Week")
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x='DayOfWeek', y='Vehicles', data=filtered, order=days, palette="Blues_d", ax=ax, errorbar=None)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# --- Model Prediction Section ---
st.markdown("---")
st.markdown("### 🤖 Real-Time Prediction Simulator")
st.info("Adjust the sliders below to simulate conditions (e.g., A rainy Friday evening).")

c_pred1, c_pred2 = st.columns(2)
with c_pred1:
    input_hour = st.slider("Time of Day (Hour)", 0, 23, 17)
    input_temp = st.slider("Temperature (°C)", -10, 40, 25)
with c_pred2:
    input_rain = st.slider("Precipitation (mm)", 0.0, 20.0, 0.0)
    # Using average lag values for simulation simplicity
    avg_lag = filtered['Vehicles'].mean()

if st.button("Predict Congestion Level"):
    # Load model
    try:
        model = joblib.load('traffic_model.pkl')
        # Features: ['Hour', 'Temperature_C', 'Precipitation_mm', 'Lag_1h', 'Lag_24h']
        prediction = model.predict([[input_hour, input_temp, input_rain, avg_lag, avg_lag]])[0]
        
        # --- NEW: Interpretation Logic ---
        st.write("---")
        c_res1, c_res2 = st.columns([1, 2])
        
        with c_res1:
            # This metric will also take on the card style automatically
            st.metric("Predicted Score", f"{prediction:.4f}")
        
        with c_res2:
            if prediction > 1.5:
                st.error("🔴 **Status: HEAVY CONGESTION (Peak Traffic)**")
                st.markdown("Traffic is significantly higher than average. Expect delays.")
            elif prediction > 0.5:
                st.warning("🟠 **Status: MODERATE TRAFFIC**")
                st.markdown("Traffic is building up. Standard commuting times.")
            elif prediction > -0.5:
                st.success("🟢 **Status: NORMAL FLOW**")
                st.markdown("Traffic is moving smoothly. No significant delays.")
            else:
                st.info("🔵 **Status: LOW TRAFFIC (Free Flow)**")
                st.markdown("Roads are clear. Optimal time for travel.")
                
    except FileNotFoundError:
        st.warning("Model file not found. Please run 'generate_analysis.py' first to train the model.")