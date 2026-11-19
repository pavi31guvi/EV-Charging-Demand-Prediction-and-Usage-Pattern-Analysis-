"""
EV Charging Demand Dashboard (Streamlit)

Run:      streamlit run app.py
Install:  pip install streamlit pandas plotly scikit-learn lightgbm

Expected files next to this app.py:
    data/ev_charging_featured.csv
    models/lightgbm_regression.pkl
    models/lightgbm_classification.pkl
    graph PNG files saved by the notebooks (see GRAPH_FOLDERS below)
"""

import os
import pickle
from datetime import datetime, time
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="EV Charging Demand", page_icon="⚡", layout="wide")

# ---------------------------------------------------------------
# 1. Paths and constants
# ---------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "ev_charging_featured.csv")
REG_MODEL_PATH = os.path.join(BASE_DIR, "models", "lightgbm_regression.pkl")
CLF_MODEL_PATH = os.path.join(BASE_DIR, "models", "lightgbm_classification.pkl")

# Folders searched for the PNG graphs saved by the notebooks
GRAPH_FOLDERS = [
    os.path.join(BASE_DIR, "models"),
    os.path.join(BASE_DIR, "models", "images"),
    os.path.join(BASE_DIR, "images"),
]

MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Same input columns the saved models expect (order does not matter to the pipelines)
MODEL_FEATURES = [
    "duration", "start_day", "start_hour", "start_minute", "week_of_year",
    "user_previous_sessions", "charger_previous_sessions", "location_previous_sessions",
    "chargercompany", "location", "chargertype", "day_type", "time_of_day",
    "month_name", "day_name",
]

COMPANY_LABELS = {0: "Company 0", 1: "Company 1"}
TYPE_LABELS = {0: "Type 0", 1: "Type 1"}

# Scores saved from the notebooks (full dataset, last 20% = test period)
REGRESSION_RESULTS = pd.DataFrame(
    [
        ["Linear Regression", 7.1171, 112.6245, 10.6125, 0.5097, 0.4158],
        ["Decision Tree", 4.9781, 89.3586, 9.4530, 0.8411, 0.5365],
        ["Random Forest", 4.5852, 59.8993, 7.7395, 0.8954, 0.6893],
        ["Gradient Boosting", 4.2880, 56.5744, 7.5216, 0.8384, 0.7065],
        ["XGBoost", 4.2383, 55.9723, 7.4815, 0.8335, 0.7097],
        ["LightGBM", 4.1510, 54.5723, 7.3873, 0.8537, 0.7169],
    ],
    columns=["Model", "MAE", "MSE", "RMSE", "Train R²", "Test R²"],
)

CLASSIFICATION_RESULTS = pd.DataFrame(
    [
        ["Logistic Regression", 0.8407, 0.8276, 0.8265, 0.8200, 0.8232, 0.8881],
        ["Decision Tree", 0.9196, 0.8612, 0.8379, 0.8885, 0.8624, 0.8958],
        ["Random Forest", 0.9079, 0.8750, 0.8286, 0.9389, 0.8803, 0.9339],
        ["XGBoost", 0.9109, 0.8860, 0.8547, 0.9244, 0.8881, 0.9505],
        ["LightGBM", 0.9116, 0.8880, 0.8561, 0.9269, 0.8901, 0.9496],
    ],
    columns=["Model", "Train Accuracy", "Test Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
)

FORECAST_RESULTS = pd.DataFrame(
    [
        ["7-Day Moving Average (walk-forward)", 495.04, 686.51],
        ["ARIMA(1,1,1)", 568.27, 788.70],
        ["Naive Baseline", 629.47, 859.25],
        ["Exp. Smoothing (trend)", 784.00, 1028.75],
        ["Exp. Smoothing (trend + weekly)", 790.79, 1028.69],
    ],
    columns=["Model", "MAE", "RMSE"],
)


# ---------------------------------------------------------------
# 2. Helpers
# ---------------------------------------------------------------
def show_chart(fig):
    """Draw a Plotly chart across the full width of the tab."""
    st.plotly_chart(fig, use_container_width=True)


# Display width in pixels for each saved graph. Change a number to resize that graph.
# Use "stretch" instead of a number to fill the whole column.
GRAPH_WIDTHS = {
    "eda_demand_distribution.png": 1200,
    "eda_sessions_by_hour.png": 900,
    "eda_duration_vs_demand.png": 900,
    "eda_demand_by_charger_type.png": 700,
    "eda_location_hour_heatmap.png": 1200,
    "residual_lightgbm.png":1300,
    "lightgbm_regression_feature_importance.png": 800,
    "classification_feature_importance.png":1000,
    "confusion_matrix_lightgbm.png":600,
    "time_series_daily_demand.png":950,
    "monthly_demand.png":1000,
    "time_of_day_7day.png":1000,
    "moving_average_7day_forecast.png":1000
}
DEFAULT_GRAPH_WIDTH = 700


def show_graph(filename, caption="", width=None):
    """Show a saved PNG. Width comes from the table above unless you pass one in."""
    width = width or GRAPH_WIDTHS.get(filename, DEFAULT_GRAPH_WIDTH)
    for folder in GRAPH_FOLDERS:
        path = os.path.join(folder, filename)
        if os.path.exists(path):
            st.image(path, caption=caption, width=width)
            return
    st.info(f"Graph not found: {filename}")


def show_graph_grid(items, columns=2):
    """items is a list of (filename, caption). Shows them side by side."""
    cols = st.columns(columns)
    for i, (filename, caption) in enumerate(items):
        with cols[i % columns]:
            show_graph(filename, caption)


def time_of_day_label(hour):
    """Same rule the EDA notebook used to create time_of_day."""
    if hour < 6:
        return "Night"
    if hour < 12:
        return "Morning"
    if hour < 17:
        return "Afternoon"
    if hour < 21:
        return "Evening"
    return "Night"


# ---------------------------------------------------------------
# 3. Loading data and models
# ---------------------------------------------------------------
@st.cache_data(show_spinner="Loading charging data...")
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["start_full"] = pd.to_datetime(df["start_full"], errors="coerce")
    df = df.dropna(subset=["start_full"]).reset_index(drop=True)

    # Same rule as the Classification notebook: High Demand = demand >= median
    median_demand = float(df["demand"].median())
    df["target"] = (df["demand"] >= median_demand).astype(int)
    return df, median_demand


@st.cache_resource(show_spinner="Loading saved models...")
def load_models():
    with open(REG_MODEL_PATH, "rb") as f:
        reg_model = pickle.load(f)
    with open(CLF_MODEL_PATH, "rb") as f:
        clf_model = pickle.load(f)
    return reg_model, clf_model

# ---------------------------------------------------------------
# EDA graphs (saved by EDA.ipynb)
# ---------------------------------------------------------------
def tab_eda_graphs():
    st.subheader("Exploratory data analysis (EDA)")

    st.markdown("**1. Distribution of demand and duration**")
    show_graph("eda_demand_distribution.png",
               "Demand and duration are right-skewed: most sessions are short or moderate, "
               "with a few very long or high-demand sessions.")

    st.markdown("**2. Charging sessions by start hour**")
    show_graph("eda_sessions_by_hour.png",
               "Most charging starts between 17:00 and 19:00. Very early morning hours have the fewest sessions.")

    st.markdown("**3. Duration vs energy demand**")
    show_graph("eda_duration_vs_demand.png",
               "Longer sessions generally have higher demand, but the spread is wide, "
               "so duration alone does not explain demand.")

    st.markdown("**4. Demand by charger type**")
    show_graph("eda_demand_by_charger_type.png",
               "Charger Type 1 sessions usually have higher demand and are shorter than Type 0 sessions.")

    st.markdown("**5. Median demand by location and start hour**")
    show_graph("eda_location_hour_heatmap.png",
               "Darker cells mean higher typical demand. Locations with very few sessions, "
               "such as bus garage and accommodation, give less reliable results.")
# ---------------------------------------------------------------
# 4. Tab 1: Usage patterns (metrics only)
# ---------------------------------------------------------------
def tab_usage_patterns(df):
    st.subheader("Overall EV charging usage")
    st.caption("Calculated from all charging sessions in the dataset.")

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Sessions", f"{len(df):,}")
    k2.metric("Total demand", f"{df['demand'].sum():,.0f}")
    k3.metric("Mean demand", f"{df['demand'].mean():.2f}")
    k4.metric("Median demand", f"{df['demand'].median():.2f}")
    k5.metric("Mean duration (min)", f"{df['duration'].mean():.0f}")
    k6.metric("High-demand share", f"{df['target'].mean() * 100:.1f}%")

    st.divider()
    tab_eda_graphs()

# ---------------------------------------------------------------
# 5. Tab 2: Model comparison (regression and classification)
# ---------------------------------------------------------------
REGRESSION_EXPLANATIONS = {
    "LightGBM": "It builds many small trees one after another, each fixing the previous trees' mistakes, "
                "and gave the smallest errors and the highest Test R².",
    "XGBoost": "It is another boosting method that builds trees step by step, and it performs almost as well as LightGBM.",
    "Gradient Boosting": "It builds trees one after another to fix earlier errors, but it is slightly less accurate than LightGBM.",
    "Random Forest": "It averages many independent trees, which is stable, but it overfits more than the boosting models.",
    "Decision Tree": "It splits sessions into groups using simple rules, but it memorizes the training data too closely.",
    "Linear Regression": "It fits a straight-line relationship, which is too simple for this charging data.",
}

CLASSIFICATION_EXPLANATIONS = {
    "LightGBM": "It is a boosted tree model with the best balance of correct predictions and catching High Demand sessions (highest F1).",
    "XGBoost": "It is a boosted tree model with the highest ROC-AUC, so it separates High and Normal demand very well.",
    "Random Forest": "It catches the most High Demand sessions (highest recall), but it raises more false alarms.",
    "Decision Tree": "It is a single set of rules that works reasonably well but overfits more than the boosting models.",
    "Logistic Regression": "It is a simple straight-line baseline and the weakest of the models compared.",
}

def slug(name):
    """'Random Forest' -> 'random_forest' (the same naming the notebooks used for saved graphs)."""
    return name.lower().replace(" ", "_")

def highlight_best(table, best_name):
    """Show the table with the best model's row in bold."""
    styled = table.style.format("{:.8f}")
    return styled.apply(
        lambda row: ["font-weight: bold" if row.name == best_name else "" for _ in row],
        axis=1,
    )


def correlation_with_demand(df):
    """Same idea as the Regression notebook: correlation of every numeric column with demand."""
    numeric = df.drop(columns=["target"]).select_dtypes(include="number")  # target is derived from demand
    corr = numeric.corr()["demand"].sort_values(ascending=False)
    return corr.rename("Correlation with demand").to_frame()


def regression_section(df):
    st.subheader("Regression: predicting charging demand")

    table = REGRESSION_RESULTS.set_index("Model")
    best = table["Test R²"].idxmax()  # best = highest Test R²

    st.table(highlight_best(table, best))
    st.success(f"**Best regression model: {best}.** {REGRESSION_EXPLANATIONS[best]}")

    st.markdown("**1. Correlation with demand**")
    st.dataframe(correlation_with_demand(df).round(4))
    st.caption("Correlation runs from -1 to +1. Values near +1 mean demand rises with that feature, "
               "values near 0 mean little linear relationship, and values near -1 mean demand falls.")

    st.markdown("**2. Residual analysis (LightGBM)**")
    show_graph(f"residual_{slug(best)}.png",
               f"{best}: residual distribution (left) and residuals vs predicted demand (right)")
    st.caption("A residual is actual demand minus predicted demand. A random cloud around the red zero line "
               "is good. Patterns like funnels or bands mean the errors are not random.")

    st.markdown("**3. Feature importance (LightGBM)**")
    show_graph(f"{slug(best)}_regression_feature_importance.png",
               f"{best}: which inputs the model used most when splitting its trees")


def classification_section():
    st.subheader("Classification: predicting High Demand sessions")

    table = CLASSIFICATION_RESULTS.set_index("Model")
    best = table["F1"].idxmax()  # best = highest F1 (balance of precision and recall)

    st.table(highlight_best(table, best))
    st.success(f"**Best classification model: {best}.** {CLASSIFICATION_EXPLANATIONS[best]}")

    st.markdown("**1. Confusion matrix (LightGBM))**")
    show_graph(f"confusion_matrix_{slug(best)}.png",
               f"{best}: rows are actual classes, columns are predicted classes. Diagonal cells are correct.")

    st.markdown("**2. ROC curve**")
    show_graph("classification_roc_curve.png",
               "ROC curve for all models: the closer a curve is to the top-left corner, the better")

    st.markdown("**3. Feature importance (Light BGM)**")
    show_graph("classification_feature_importance.png",
               f"{best}: which inputs the model used most when splitting its trees")

def tab_model_comparison(df):
    regression_section(df)
    st.divider()
    classification_section()

# ---------------------------------------------------------------
# 6. Tab 3: Time Forecast
# ---------------------------------------------------------------
def tab_time_forecast():
    st.subheader("**1. Daily demand patterns**")
    show_graph("time_series_daily_demand.png", "Daily total demand over the year")

    st.subheader("**2.Monthly demand patterns**")
    show_graph("monthly_demand.png", "Monthly total demand (first and last months are partial)")

    st.subheader("**3. Daily demand patterns**")
    show_graph("time_of_day_7day.png", "7-day average demand by time of day")

    st.subheader("**4. Forecast comparison**")
    st.caption("Lower MAE and RMSE are better. The 7-day moving average uses each real test day as it "
               "arrives, so it is effectively a one-day-ahead forecast. The other models forecast the "
               "whole test period at once, so this comparison is not like-for-like.")
    st.dataframe(FORECAST_RESULTS, hide_index=True)
    show_chart(px.bar(FORECAST_RESULTS, x="Model", y="MAE", title="MAE (lower is better)"))

    st.subheader("**5. 7-day Moving average forecast**")
    show_graph("moving_average_7day_forecast.png", "7-day moving average forecast")

# ---------------------------------------------------------------
# 7. Tab 4: Predict session
# ---------------------------------------------------------------
def tab_predict_session(df, reg_model, clf_model, median_demand):
    st.subheader("Predict a charging session")

    col1, col2, col3 = st.columns(3)
    with col1:
        session_date = st.date_input("Start date", value=df["start_full"].max().date())
        start_time = st.time_input("Start time", value=time(18, 0))
        duration = st.number_input("Expected duration (minutes)", min_value=1.0, max_value=5000.0,
                                   value=float(df["duration"].median()), step=5.0)
    with col2:
        location = st.selectbox("Location", sorted(df["location"].unique()))
        charger_type = st.selectbox("Charger type", [0, 1], format_func=lambda v: TYPE_LABELS[v])
        charger_company = st.selectbox("Charger company", [0, 1], format_func=lambda v: COMPANY_LABELS[v])
    with col3:
        user_prev = st.number_input("User previous sessions", min_value=0,
                                    value=int(df["user_previous_sessions"].median()))
        charger_prev = st.number_input("Charger previous sessions", min_value=0,
                                       value=int(df["charger_previous_sessions"].median()))
        location_prev = st.number_input("Location previous sessions", min_value=0,
                                        value=int(df["location_previous_sessions"].median()))

    if not st.button("Predict", type="primary"):
        return

    # Rebuild the calendar features the same way the EDA notebook did
    ts = pd.Timestamp(datetime.combine(session_date, start_time))
    row = pd.DataFrame([{
        "duration": float(duration),
        "start_day": ts.day,
        "start_hour": ts.hour,
        "start_minute": ts.minute,
        "week_of_year": int(ts.isocalendar()[1]),
        "user_previous_sessions": int(user_prev),
        "charger_previous_sessions": int(charger_prev),
        "location_previous_sessions": int(location_prev),
        "chargercompany": int(charger_company),
        "location": location,
        "chargertype": int(charger_type),
        "day_type": "Weekend" if ts.dayofweek >= 5 else "Weekday",
        "time_of_day": time_of_day_label(ts.hour),
        "month_name": MONTH_ORDER[ts.month - 1],
        "day_name": DAY_ORDER[ts.dayofweek],
    }])[MODEL_FEATURES]

    predicted_demand = float(reg_model.predict(row)[0])
    high_prob = float(clf_model.predict_proba(row)[0, 1])

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted demand", f"{predicted_demand:.2f}")
    c2.metric("High-demand probability", f"{high_prob * 100:.1f}%")
    c3.metric("Predicted class", "High Demand" if high_prob >= 0.5 else "Normal Demand")
    st.caption(f"High Demand means demand at or above the dataset median of {median_demand:.2f}.")

# ---------------------------------------------------------------
# 8. App entry point
# ---------------------------------------------------------------
def main():
    st.title("⚡ EV Charging Demand Dashboard")

    missing = [p for p in (DATA_PATH, REG_MODEL_PATH, CLF_MODEL_PATH) if not os.path.exists(p)]
    if missing:
        st.error("These files were not found:\n\n" + "\n".join(missing))
        st.stop()

    df, median_demand = load_data()
    reg_model, clf_model = load_models()

    tab_usage, tab_compare, tab_forecast, tab_predict = st.tabs(
        ["**Usage patterns**", "**Model comparison**", "**Time Forecast**", "**Predict session**"]
    )
    with tab_usage:
        tab_usage_patterns(df)
    with tab_compare:
        tab_model_comparison(df)
    with tab_forecast:
        tab_time_forecast()
    with tab_predict:
        tab_predict_session(df, reg_model, clf_model, median_demand)


if __name__ == "__main__":
    main()