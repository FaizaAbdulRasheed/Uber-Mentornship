"""
🚦 Traffic Volume Forecasting & Analytics
Streamlit dashboard — production-ready, crash-resistant.
"""

import sys
import os

# ── path fix so imports work whether run from root or app/ ──────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from src.pipeline.orchestrate import run_pipeline
from src.features.engineer import FEATURE_COLS

# ── page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Traffic Forecast",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Syne', sans-serif;
        background: #0d0e14;
        color: #e8e9f0;
    }
    [data-testid="stSidebar"] {
        background: #111219 !important;
        border-right: 1px solid #1f2130;
    }
    [data-testid="stSidebar"] * { color: #b8bdd8 !important; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label { color: #7b82a8 !important; font-size: 0.78rem !important; letter-spacing: .08em; text-transform: uppercase; }

    h1 { font-family: 'Syne', sans-serif; font-weight: 800; font-size: 2rem; color: #ffffff; letter-spacing: -.02em; }
    h2, h3 { font-family: 'Syne', sans-serif; font-weight: 600; color: #c8cbdf; }

    /* metric cards */
    [data-testid="metric-container"] {
        background: #14161f;
        border: 1px solid #1e2235;
        border-radius: 12px;
        padding: 1rem 1.2rem;
    }
    [data-testid="metric-container"] label { color: #5c6285 !important; font-size: 0.72rem; text-transform: uppercase; letter-spacing: .1em; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #f0f2ff !important; font-family: 'Space Mono', monospace; font-size: 1.6rem; }
    [data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: 0.8rem; }

    /* tabs */
    [data-baseweb="tab-list"] { border-bottom: 1px solid #1f2130; gap: 0.25rem; }
    [data-baseweb="tab"] { background: transparent; border-radius: 8px 8px 0 0; color: #5c6285; font-size: 0.85rem; font-weight: 600; }
    [aria-selected="true"][data-baseweb="tab"] { background: #1a1d2b; color: #7c8dff; border-bottom: 2px solid #7c8dff; }

    /* divider */
    hr { border-color: #1f2130; }

    /* buttons */
    .stButton > button {
        background: #7c8dff;
        color: #fff;
        border: none;
        border-radius: 8px;
        font-family: 'Space Mono', monospace;
        font-size: 0.82rem;
        padding: 0.55rem 1.2rem;
        transition: background .2s;
    }
    .stButton > button:hover { background: #5c6eff; }

    /* selectbox, slider */
    [data-testid="stSelectbox"] > div > div { background: #14161f; border-color: #1e2235; border-radius: 8px; color: #e8e9f0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── plotly theme ─────────────────────────────────────────────────────────────
CHART_BG  = "#0d0e14"
PAPER_BG  = "#0d0e14"
GRID_COL  = "#1a1d2b"
ACCENT    = "#7c8dff"
ACCENT2   = "#ff7c8d"
ACCENT3   = "#7cffca"


def _base_layout(**kw):
    return dict(
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=CHART_BG,
        font=dict(family="Syne, sans-serif", color="#b8bdd8", size=12),
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(gridcolor=GRID_COL, linecolor=GRID_COL, zerolinecolor=GRID_COL),
        yaxis=dict(gridcolor=GRID_COL, linecolor=GRID_COL, zerolinecolor=GRID_COL),
        **kw,
    )


# ── data loading (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training model on synthetic traffic data…")
def load_data(model_type: str = "gbr"):
    return run_pipeline(model_type=model_type)


# ── sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚦 Controls")
    st.markdown("---")

    model_choice = st.selectbox(
        "Model",
        ["Gradient Boosting (GBR)", "Random Forest (RF)"],
        index=0,
    )
    model_key = "gbr" if "Gradient" in model_choice else "rf"

    st.markdown("---")
    st.markdown("#### Filter")
    selected_junction = st.selectbox("Junction", [1, 2, 3, 4], index=0)

    st.markdown("---")
    st.markdown("#### Predict")
    pred_hour    = st.slider("Hour of day", 0, 23, 8)
    pred_dow     = st.slider("Day of week (0=Mon)", 0, 6, 0)
    pred_month   = st.slider("Month", 1, 12, 6)
    pred_temp    = st.slider("Temperature (°C)", -5, 45, 24)
    pred_humid   = st.slider("Humidity (%)", 10, 100, 65)
    pred_wind    = st.slider("Wind (km/h)", 0, 80, 10)
    pred_precip  = st.slider("Precipitation (mm)", 0, 50, 0)
    pred_event   = st.checkbox("Event day?", value=False)

    retrain = st.button("🔄 Retrain model")

# ── load data ────────────────────────────────────────────────────────────────
if retrain:
    st.cache_resource.clear()

state = load_data(model_key)
df_raw = state["df_raw"]
df_feat = state["df_feat"]
metrics = state["metrics"]
y_test  = state["y_test"]
y_pred  = state["y_pred"]
fi      = state["feature_importance"]
pipe    = state["pipe"]

# ── header ───────────────────────────────────────────────────────────────────
st.markdown("# 🚦 Traffic Volume Forecasting")
st.markdown(
    "<p style='color:#5c6285;font-size:0.9rem;margin-top:-0.5rem;'>"
    "End-to-end ML forecasting · 4 junctions · 365 days · 18 features"
    "</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ── KPI row ──────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("MAE",          f"{metrics['mae']:.2f}",  help="Mean Absolute Error on test set")
k2.metric("RMSE",         f"{metrics['rmse']:.2f}", help="Root Mean Squared Error")
k3.metric("R²",           f"{metrics['r2']:.3f}",  help="Coefficient of determination")
k4.metric("Baseline MAE", f"{metrics['baseline_mae']:.2f}", help="Naïve lag-1 baseline")
k5.metric("Improvement",  f"{metrics['improvement']}%",
          delta=f"vs baseline", help="MAE improvement over naïve baseline")

st.markdown("---")

# ── tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Traffic Trends",
    "🔍 Actual vs Predicted",
    "🏙️ Junction Analysis",
    "🌡️ Weather Impact",
    "🤖 Feature Importance",
])

# ── TAB 1: Traffic trends ─────────────────────────────────────────────────────
with tab1:
    df_j = df_raw[df_raw["junction"] == selected_junction].copy()

    # Daily average
    daily = df_j.groupby("date")["vehicles"].mean().reset_index()
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=daily["date"], y=daily["vehicles"],
        fill="tozeroy", fillcolor="rgba(124,141,255,0.12)",
        line=dict(color=ACCENT, width=1.5),
        name="Daily avg traffic",
    ))
    fig1.update_layout(
        **_base_layout(title=f"Daily Average Traffic — Junction {selected_junction}"),
        height=300,
        showlegend=False,
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Hourly heatmap
    st.markdown("#### 🗓️ Hourly Traffic Heatmap (Day of Week × Hour)")
    pivot = df_j.pivot_table(values="vehicles", index="day_of_week", columns="hour", aggfunc="mean")
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig2 = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"{h:02d}:00" for h in pivot.columns],
        y=[day_labels[d] for d in pivot.index],
        colorscale=[[0, "#0d0e14"], [0.5, "#3d4aaa"], [1, "#7c8dff"]],
        showscale=True,
        colorbar=dict(tickfont=dict(color="#b8bdd8")),
    ))
    fig2.update_layout(**_base_layout(), height=260)
    st.plotly_chart(fig2, use_container_width=True)

    # Monthly trend
    df_raw["month_label"] = df_raw["datetime"].dt.strftime("%b")
    monthly = (
        df_raw[df_raw["junction"] == selected_junction]
        .groupby(df_raw["datetime"].dt.month)["vehicles"]
        .mean()
        .reset_index()
    )
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly["month_name"] = monthly["datetime"].apply(lambda m: month_names[m-1])

    fig3 = go.Figure(go.Bar(
        x=monthly["month_name"], y=monthly["vehicles"],
        marker_color=ACCENT, marker_line_width=0,
        text=monthly["vehicles"].round(1), textposition="outside",
        textfont=dict(color="#b8bdd8", size=10),
    ))
    fig3.update_layout(**_base_layout(title="Monthly Average Traffic"), height=280)
    st.plotly_chart(fig3, use_container_width=True)

# ── TAB 2: Actual vs Predicted ───────────────────────────────────────────────
with tab2:
    n_plot = min(500, len(y_test))
    x_ax   = list(range(n_plot))

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=x_ax, y=y_test[:n_plot],
        line=dict(color=ACCENT3, width=1.2),
        name="Actual",
    ))
    fig4.add_trace(go.Scatter(
        x=x_ax, y=y_pred[:n_plot],
        line=dict(color=ACCENT2, width=1.2, dash="dot"),
        name="Predicted",
    ))
    fig4.update_layout(
        **_base_layout(title=f"Actual vs Predicted — first {n_plot} test samples"),
        height=360,
        legend=dict(bgcolor="#14161f", bordercolor="#1e2235"),
    )
    st.plotly_chart(fig4, use_container_width=True)

    # Residuals
    residuals = y_test[:n_plot] - y_pred[:n_plot]
    fig5 = go.Figure(go.Scatter(
        x=x_ax, y=residuals,
        mode="markers",
        marker=dict(color=ACCENT, size=3, opacity=0.5),
        name="Residual",
    ))
    fig5.add_hline(y=0, line_dash="dash", line_color=ACCENT2, line_width=1)
    fig5.update_layout(**_base_layout(title="Residuals (Actual − Predicted)"), height=260)
    st.plotly_chart(fig5, use_container_width=True)

    # Error distribution
    fig6 = go.Figure(go.Histogram(
        x=residuals, nbinsx=50,
        marker_color=ACCENT, marker_line_width=0, opacity=0.85,
    ))
    fig6.update_layout(**_base_layout(title="Residual Distribution"), height=260)
    st.plotly_chart(fig6, use_container_width=True)

# ── TAB 3: Junction analysis ─────────────────────────────────────────────────
with tab3:
    junction_stats = df_raw.groupby("junction")["vehicles"].agg(
        Mean="mean", Median="median", Max="max", Std="std"
    ).reset_index().round(2)
    junction_stats.columns = ["Junction", "Mean", "Median", "Peak", "Std Dev"]
    st.dataframe(
    junction_stats,
    use_container_width=True,
    hide_index=True,
)

    # Bar comparison
    fig7 = go.Figure()
    for col, color in zip(["Mean", "Peak"], [ACCENT, ACCENT2]):
        fig7.add_trace(go.Bar(
            x=[f"J{j}" for j in junction_stats["Junction"]],
            y=junction_stats[col],
            name=col, marker_color=color,
        ))
    fig7.update_layout(
        **_base_layout(title="Junction Comparison — Mean vs Peak"),
        barmode="group", height=320,
        legend=dict(bgcolor="#14161f", bordercolor="#1e2235"),
    )
    st.plotly_chart(fig7, use_container_width=True)

    # Hourly profile per junction
    hourly_all = df_raw.groupby(["junction", "hour"])["vehicles"].mean().reset_index()
    fig8 = go.Figure()
    colors = [ACCENT, ACCENT2, ACCENT3, "#ffca7c"]
    for i, jn in enumerate([1, 2, 3, 4]):
        sub = hourly_all[hourly_all["junction"] == jn]
        fig8.add_trace(go.Scatter(
            x=sub["hour"], y=sub["vehicles"],
            name=f"Junction {jn}",
            line=dict(color=colors[i], width=2),
        ))
    fig8.update_layout(
        **_base_layout(title="Hourly Traffic Profile by Junction"),
        height=320,
        legend=dict(bgcolor="#14161f", bordercolor="#1e2235"),
    )
    st.plotly_chart(fig8, use_container_width=True)

# ── TAB 4: Weather impact ─────────────────────────────────────────────────────
with tab4:
    df_w = df_raw[df_raw["junction"] == selected_junction].copy()

    c1, c2 = st.columns(2)
    with c1:
        fig9 = px.scatter(
            df_w.sample(min(2000, len(df_w)), random_state=1),
            x="temperature", y="vehicles",
            color="is_weekend",
            color_discrete_map={0: ACCENT, 1: ACCENT2},
            labels={"temperature": "Temp (°C)", "vehicles": "Vehicles", "is_weekend": "Weekend"},
            title="Temperature vs Traffic",
        )
        fig9.update_layout(**_base_layout(), height=300)
        fig9.update_traces(marker=dict(size=4, opacity=0.5))
        st.plotly_chart(fig9, use_container_width=True)

    with c2:
        fig10 = px.scatter(
            df_w.sample(min(2000, len(df_w)), random_state=2),
            x="precipitation", y="vehicles",
            color="is_event",
            color_discrete_map={0: ACCENT3, 1: ACCENT2},
            labels={"precipitation": "Precip (mm)", "vehicles": "Vehicles", "is_event": "Event"},
            title="Precipitation vs Traffic",
        )
        fig10.update_layout(**_base_layout(), height=300)
        fig10.update_traces(marker=dict(size=4, opacity=0.5))
        st.plotly_chart(fig10, use_container_width=True)

    # Weekend vs weekday
    we_comp = (
        df_w.groupby("is_weekend")["vehicles"]
        .mean()
        .reset_index()
        .replace({"is_weekend": {0: "Weekday", 1: "Weekend"}})
    )
    fig11 = go.Figure(go.Bar(
        x=we_comp["is_weekend"], y=we_comp["vehicles"],
        marker_color=[ACCENT, ACCENT2], marker_line_width=0,
        text=we_comp["vehicles"].round(1), textposition="outside",
        textfont=dict(color="#b8bdd8"),
    ))
    fig11.update_layout(**_base_layout(title="Weekday vs Weekend Average Traffic"), height=260)
    st.plotly_chart(fig11, use_container_width=True)

# ── TAB 5: Feature importance ────────────────────────────────────────────────
with tab5:
    if fi:
        fi_df = pd.DataFrame(
            {"Feature": list(fi.keys()), "Importance": list(fi.values())}
        ).sort_values("Importance", ascending=True)

        fig12 = go.Figure(go.Bar(
            x=fi_df["Importance"], y=fi_df["Feature"],
            orientation="h",
            marker=dict(
                color=fi_df["Importance"],
                colorscale=[[0, "#1a1d2b"], [1, ACCENT]],
                showscale=False,
                line_width=0,
            ),
            text=fi_df["Importance"].round(3), textposition="outside",
            textfont=dict(color="#b8bdd8", size=10),
        ))
        fig12.update_layout(
    **_base_layout(title="Feature Importance"),
    height=520,
)
        st.plotly_chart(fig12, use_container_width=True)
    else:
        st.info("Feature importances not available for this model type.")

# ── live prediction panel ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🤖 Live Prediction")

# Build a feature vector matching FEATURE_COLS
# FEATURE_COLS = [
#   "hour","day_of_week","month","is_weekend","is_event",
#   "temperature","humidity","wind","precipitation",
#   "lag_1","lag_24","lag_168",
#   "rolling_mean_3","rolling_mean_24","rolling_std_3","rolling_std_24",
#   "hour_sin","hour_cos",
# ]

is_weekend_val = int(pred_dow >= 5)

# Use junction median lags as proxies when user doesn't supply history
df_j_feat = df_feat[df_feat["junction"] == selected_junction]
lag_median     = df_j_feat["lag_1"].median()
roll3_median   = df_j_feat["rolling_mean_3"].median()
roll24_median  = df_j_feat["rolling_mean_24"].median()
std3_median    = df_j_feat["rolling_std_3"].median()
std24_median   = df_j_feat["rolling_std_24"].median()

import math
hour_sin_val = math.sin(2 * math.pi * pred_hour / 24)
hour_cos_val = math.cos(2 * math.pi * pred_hour / 24)

feat_vec = pd.DataFrame([{
    "hour":             pred_hour,
    "day_of_week":      pred_dow,
    "month":            pred_month,
    "is_weekend":       is_weekend_val,
    "is_event":         int(pred_event),
    "temperature":      pred_temp,
    "humidity":         pred_humid,
    "wind":             pred_wind,
    "precipitation":    pred_precip,
    "lag_1":            lag_median,
    "lag_24":           lag_median,
    "lag_168":          lag_median,
    "rolling_mean_3":   roll3_median,
    "rolling_mean_24":  roll24_median,
    "rolling_std_3":    std3_median,
    "rolling_std_24":   std24_median,
    "hour_sin":         hour_sin_val,
    "hour_cos":         hour_cos_val,
}])[FEATURE_COLS]

try:
    prediction = float(np.clip(pipe.predict(feat_vec)[0], 0, None))
except Exception as e:
    prediction = None
    st.error(f"Prediction error: {e}")

if prediction is not None:
    pc1, pc2, pc3 = st.columns([1, 1, 2])
    pc1.metric("Predicted Vehicles", f"{prediction:.1f}")
    level = "🔴 Heavy" if prediction > 20 else ("🟡 Moderate" if prediction > 10 else "🟢 Light")
    pc2.metric("Congestion Level", level)
    pc3.markdown(
        f"<p style='color:#5c6285;font-size:0.85rem;padding-top:0.4rem;'>"
        f"Junction <b style='color:#e8e9f0'>{selected_junction}</b> · "
        f"Hour <b style='color:#e8e9f0'>{pred_hour:02d}:00</b> · "
        f"{'Weekend' if is_weekend_val else 'Weekday'} · "
        f"{'📅 Event day' if pred_event else 'Normal day'}"
        f"</p>",
        unsafe_allow_html=True,
    )

# ── footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#2a2d3e;font-size:0.75rem;'>"
    "Traffic Forecasting · Built with Streamlit · Gradient Boosting · Plotly"
    "</p>",
    unsafe_allow_html=True,
)
