import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# =========================
# CONFIG
# =========================
HIST_PATH = "data/output/engineered_data.csv"
FCST_PATH = "data/output/forecast_result.csv"


# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_all_data():
    hist = pd.read_csv(HIST_PATH)
    hist["date"] = pd.to_datetime(hist["date"])

    fcst = pd.read_csv(FCST_PATH)
    fcst["date"] = pd.to_datetime(fcst["date"])

    return hist, fcst


# =========================
# APP
# =========================
def main():
    st.title("Financial Services Forecast Dashboard")

    hist_df, fcst_df = load_all_data()

    # =========================
    # FILTERS
    # =========================
    st.sidebar.header("Filter")

    branches = sorted(hist_df["branch"].unique())
    products = sorted(hist_df["product"].unique())

    selected_branch = st.sidebar.selectbox("Select Branch", ["All"] + branches)
    selected_product = st.sidebar.selectbox("Select Product", ["All"] + products)

    date_range = st.sidebar.date_input(
        "Select Date Range",
        [hist_df["date"].min(), fcst_df["date"].max()]
    )

    # =========================
    # APPLY FILTER
    # =========================
    filtered_hist = hist_df.copy()
    filtered_fcst = fcst_df.copy()

    if selected_branch != "All":
        filtered_hist = filtered_hist[filtered_hist["branch"] == selected_branch]
        filtered_fcst = filtered_fcst[filtered_fcst["branch"] == selected_branch]

    if selected_product != "All":
        filtered_hist = filtered_hist[filtered_hist["product"] == selected_product]
        filtered_fcst = filtered_fcst[filtered_fcst["product"] == selected_product]

    if len(date_range) == 2:
        start_date, end_date = date_range

        filtered_hist = filtered_hist[
            (filtered_hist["date"] >= pd.to_datetime(start_date)) &
            (filtered_hist["date"] <= pd.to_datetime(end_date))
        ]

        filtered_fcst = filtered_fcst[
            (filtered_fcst["date"] >= pd.to_datetime(start_date)) &
            (filtered_fcst["date"] <= pd.to_datetime(end_date))
        ]

    # =========================
    # TABLE VIEW
    # =========================
    st.subheader("Filtered Data (Actual)")
    st.dataframe(filtered_hist.head(50))

    st.subheader("Filtered Data (Forecast)")
    st.dataframe(filtered_fcst.head(50))

    # =========================
    # PREPARE DATA FOR PLOT
    # =========================
    hist_plot = (
        filtered_hist
        .groupby("date")["revenue"]
        .sum()
        .reset_index()
        .sort_values("date")
    )

    fcst_plot = (
        filtered_fcst
        .groupby("date")["forecast"]
        .sum()
        .reset_index()
        .sort_values("date")
    )

    # =========================
    # PLOT
    # =========================
    st.subheader("Actual vs Forecast")

    fig = go.Figure()

    # ACTUAL
    fig.add_trace(go.Scatter(
        x=hist_plot["date"],
        y=hist_plot["revenue"],
        mode="lines+markers",
        name="Actual",
        line=dict(dash="solid")
    ))

    # FORECAST
    fig.add_trace(go.Scatter(
        x=fcst_plot["date"],
        y=fcst_plot["forecast"],
        mode="lines+markers",
        name="Forecast",
        line=dict(dash="dash")
    ))

    # =========================
    # LAYOUT
    # =========================
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Revenue",
        hovermode="x unified",
        yaxis=dict(range=[0, None])  # 🔥 force min = 0
    )

    st.plotly_chart(fig, use_container_width=True)


# =========================
# RUN
# =========================
if __name__ == "__main__":
    main()