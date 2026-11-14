"""
EV Charging Demand Dashboard (beginner version)

Put this file in the "EV Charging demand session" folder, next to the
`data` folder and the saved .pkl models, then run:

    streamlit run app.py

Pages are chosen from the sidebar (no horizontal tabs):
    Overview         -> EDA.ipynb
    Regression       -> Regression.ipynb
    Classification   -> Classification.ipynb
    Time Series      -> Time_Series.ipynb
    Predict Session  -> uses the regression and classification models
"""

import pickle
import warnings
from datetime import datetime, time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

st.set_page_config(page_title="EV Charging Demand", page_icon="⚡", layout="wide")

# ---------------------------------------------------------------------------
# Paths and constants (same values the notebooks use)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

DATA_FILES = [
    BASE_DIR / "data" / "ev_charging_featured_model.csv",
    BASE_DIR / "ev_charging_featured_model.csv",
]
REGRESSION_MODEL_FILES = [BASE_DIR / "lightgbm_regression.pkl"]
CLASSIFICATION_MODEL_FILES = [BASE_DIR / "lightgbm_classification.pkl"]

NUMERIC_FEATURES = [
    "duration",
    "start_month",
    "start_day",
    "start_hour",
    "start_minute",
    "day_of_week_number",
    "week_of_year",
    "is_weekend",
    "user_previous_sessions",
    "charger_previous_sessions",
    "location_previous_sessions",
]
CATEGORICAL_FEATURES = ["location", "time_of_day", "chargertype", "chargercompany"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TRAIN_SHARE = 0.80  # first 80% of rows (by time) for training, last 20% for testing

# ---------------------------------------------------------------------------
# Results saved in the notebooks (shown as reference tables)
# ---------------------------------------------------------------------------
REG_NOTEBOOK = pd.DataFrame(
    {
        "Model": [
            "Linear Regression",
            "Decision Tree",
            "LightGBM",
            "Gradient Boosting",
            "XGBoost",
            "Random Forest",
        ],
        "MAE": [7.0809, 4.8980, 4.1483, 4.3462, 4.2197, 4.5933],
        "RMSE": [10.5988, 9.1923, 7.3597, 7.5698, 7.4496, 7.7321],
        "Train R²": [0.5089, 0.8410, 0.8535, 0.8382, 0.8361, 0.8957],
        "Test R²": [0.4173, 0.5617, 0.7190, 0.7028, 0.7121, 0.6899],
    }
)

CLS_NOTEBOOK = pd.DataFrame(
    {
        "Model": ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost", "LightGBM"],
        "Accuracy": [0.8429, 0.8604, 0.8776, 0.8873, 0.8872],
        "Precision": [0.8283, 0.8346, 0.8339, 0.8544, 0.8519],
        "Recall": [0.8567, 0.8916, 0.9364, 0.9278, 0.9316],
        "F1": [0.8423, 0.8621, 0.8822, 0.8896, 0.8899],
        "ROC-AUC": [0.8920, 0.8958, 0.9406, 0.9498, 0.9501],
    }
)

TS_NOTEBOOK = pd.DataFrame(
    {
        "Model": [
            "Exponential smoothing (trend only)",
            "Exponential smoothing (trend + weekly)",
            "7-day moving average (walk-forward)",
            "Naive baseline",
            "ARIMA(1,1,1)",
        ],
        "MAE": [784.00, 790.79, 495.04, 629.47, 568.27],
        "RMSE": [1028.75, 1028.69, 686.51, 859.25, 788.70],
    }
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def time_of_day_label(hour: int) -> str:
    """Same rule as get_time_of_day() in EDA.ipynb."""
    if hour < 6:
        return "Night"
    if hour < 12:
        return "Morning"
    if hour < 17:
        return "Afternoon"
    if hour < 21:
        return "Evening"
    return "Night"


def regression_scores(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mse),
        "R²": r2_score(y_true, y_pred),
    }


def classification_scores(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_proba),
    }


def build_pipeline(estimator):
    """Same preprocessing as the notebooks: scale numbers, one-hot encode categories."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])


def preprocessor_of(pipe):
    return pipe.steps[0][1]


def estimator_of(pipe):
    return pipe.steps[-1][1]


def model_feature_names(pipe):
    names = preprocessor_of(pipe).get_feature_names_out()
    return [name.split("__", 1)[-1] for name in names]


def importance_table(pipe):
    table = pd.DataFrame(
        {
            "Feature": model_feature_names(pipe),
            "Importance": estimator_of(pipe).feature_importances_,
        }
    )
    return table.sort_values("Importance", ascending=False).reset_index(drop=True)


def load_saved_model(paths):
    for path in paths:
        if path.exists():
            with open(path, "rb") as file:
                return pickle.load(file), path.name
    return None, None


# ---------------------------------------------------------------------------
# Data loading and splitting
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading charging records...")
def load_data():
    path = next((p for p in DATA_FILES if p.exists()), None)
    if path is None:
        return None
    df = pd.read_csv(path)
    df["start_full"] = pd.to_datetime(df["start_full"], errors="coerce")
    df = df.dropna(subset=["start_full"])
    # Sort by time, oldest first. This is what makes the 80/20 split chronological.
    return df.sort_values("start_full", kind="stable").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def split_data():
    """Chronological 80/20 split, with no shuffling, as in the notebooks."""
    df = load_data()
    threshold = float(df["demand"].median())  # 14.25 in the notebooks
    split = int(len(df) * TRAIN_SHARE)
    x = df[FEATURES]
    high_demand = (df["demand"] >= threshold).astype(int)
    return {
        "threshold": threshold,
        "x_train": x.iloc[:split],
        "x_test": x.iloc[split:],
        "y_reg_train": df["demand"].iloc[:split],
        "y_reg_test": df["demand"].iloc[split:],
        "y_cls_train": high_demand.iloc[:split],
        "y_cls_test": high_demand.iloc[split:],
    }


# ---------------------------------------------------------------------------
# Models (uses the saved .pkl files when they work, otherwise trains them)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Preparing the regression model...")
def get_regression_model():
    data = split_data()
    model, source = load_saved_model(REGRESSION_MODEL_FILES)
    if model is not None:
        try:
            model.predict(data["x_test"].head(3))
            return model, f"the saved file {source}"
        except Exception:
            pass  # saved file does not match this data, so retrain below
    model = build_pipeline(
        lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbose=-1)
    )
    model.fit(data["x_train"], data["y_reg_train"])
    return model, "a fresh training run (no usable saved file found)"


@st.cache_resource(show_spinner="Preparing the classification model...")
def get_classification_model():
    data = split_data()
    model, source = load_saved_model(CLASSIFICATION_MODEL_FILES)
    if model is not None:
        try:
            model.predict(data["x_test"].head(3))
            return model, f"the saved file {source}"
        except Exception:
            pass
    model = build_pipeline(
        lgb.LGBMClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, verbose=-1
        )
    )
    model.fit(data["x_train"], data["y_cls_train"])
    return model, "a fresh training run (no usable saved file found)"


# ---------------------------------------------------------------------------
# Time-series helpers (Time_Series.ipynb)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def daily_demand_table():
    """One row per day: the sum of demand for all sessions that day."""
    df = load_data()
    daily = df.groupby(df["start_full"].dt.date)["demand"].sum().reset_index()
    daily.columns = ["date", "total_demand"]
    daily["date"] = pd.to_datetime(daily["date"])
    return daily.sort_values("date").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def daily_by_time_of_day():
    df = load_data()
    t = df.groupby([df["start_full"].dt.date, "time_of_day"])["demand"].sum().reset_index()
    t.columns = ["date", "time_of_day", "total_demand"]
    t["date"] = pd.to_datetime(t["date"])
    return t.pivot(index="date", columns="time_of_day", values="total_demand").fillna(0)


@st.cache_data(show_spinner=False)
def decomposition_parts():
    daily = daily_demand_table()
    series = daily.set_index("date")["total_demand"].asfreq("D")
    result = seasonal_decompose(series, model="additive", period=7)
    return pd.DataFrame(
        {
            "Observed": result.observed,
            "Trend": result.trend,
            "Seasonal": result.seasonal,
            "Residual": result.resid,
        }
    )


@st.cache_data(show_spinner=False)
def adf_results():
    daily = daily_demand_table()
    raw = adfuller(daily["total_demand"])
    diff = adfuller(daily["total_demand"].diff().dropna())
    return raw[0], raw[1], diff[0], diff[1]


@st.cache_data(show_spinner="Fitting forecasting models...")
def run_forecasts():
    """Same 80/20 split and models as Time_Series.ipynb."""
    daily = daily_demand_table()
    split = int(len(daily) * TRAIN_SHARE)
    y_train = daily["total_demand"].iloc[:split].reset_index(drop=True)
    y_test = daily["total_demand"].iloc[split:].to_numpy()
    test_dates = pd.DatetimeIndex(daily["date"].iloc[split:])
    n = len(y_test)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        es_trend = ExponentialSmoothing(y_train, trend="add").fit().forecast(n)
        es_weekly = (
            ExponentialSmoothing(y_train, trend="add", seasonal="add", seasonal_periods=7)
            .fit()
            .forecast(n)
        )
        arima = ARIMA(y_train, order=(1, 1, 1)).fit().forecast(steps=n)

    # 7-day moving average, updated with each real test day as it arrives
    history = list(y_train)
    ma7 = []
    for actual in y_test:
        ma7.append(float(np.mean(history[-7:])))
        history.append(actual)

    forecasts = {
        "Exponential smoothing (trend only)": np.asarray(es_trend),
        "Exponential smoothing (trend + weekly)": np.asarray(es_weekly),
        "7-day moving average (walk-forward)": np.array(ma7),
        "Naive baseline": np.repeat(float(y_train.iloc[-1]), n),
        "ARIMA(1,1,1)": np.asarray(arima),
    }
    return test_dates, y_test, forecasts


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_overview(df):
    st.title("EV Charging Demand: Overview")
    st.caption("Based on EDA.ipynb")

    first, last = df["start_full"].min(), df["start_full"].max()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Charging sessions", f"{len(df):,}")
    c2.metric("Total demand", f"{df['demand'].sum():,.0f}")
    c3.metric("Median duration", f"{df['duration'].median():,.0f} min")
    c4.metric("Median demand", f"{df['demand'].median():.2f}")
    st.caption(f"Data covers {first:%d %b %Y} to {last:%d %b %Y}.")

    st.subheader("When do sessions start?")
    by_hour = df.groupby("start_hour").size().rename("Sessions")
    st.bar_chart(by_hour)
    st.caption("Starts peak in the late afternoon and evening, and are rare in the early morning.")

    st.subheader("Which locations are most common?")
    st.bar_chart(df["location"].value_counts().rename("Sessions"))

    st.subheader("Total demand by time of day")
    tod = df.groupby("time_of_day")["demand"].sum().sort_values(ascending=False)
    st.bar_chart(tod.rename("Total demand"))

    st.subheader("How skewed are the numbers?")
    dur, dem = df["duration"], df["demand"]
    skew_table = pd.DataFrame(
        {
            "Measure": ["Mean", "Median", "Skewness", "Skewness after log transform"],
            "Duration (min)": [dur.mean(), dur.median(), dur.skew(), np.log1p(dur).skew()],
            "Demand": [dem.mean(), dem.median(), dem.skew(), np.log1p(dem).skew()],
        }
    )
    st.dataframe(skew_table.round(2), hide_index=True)
    st.caption(
        "Both columns have a long right tail. A few very long sessions pull the mean above the "
        "median, so the log transform was used to make the spread easier to model."
    )


def page_regression(df):
    st.title("Regression: how much demand will a session have?")
    st.write(
        "Based on **Regression.ipynb**. The model reads the details of a charging session and "
        "estimates its numeric `demand`. It learned from the older 80% of sessions and was tested "
        "on the newer 20%, so these scores show how it handles sessions it has not seen."
    )

    data = split_data()
    model, source = get_regression_model()
    st.caption(f"Model: LightGBM regressor, loaded from {source}.")

    y_pred_test = model.predict(data["x_test"])
    scores = regression_scores(data["y_reg_test"], y_pred_test)
    train_r2 = r2_score(data["y_reg_train"], model.predict(data["x_train"]))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", f"{scores['MAE']:.2f}", help="Average size of the error, in demand units.")
    c2.metric("RMSE", f"{scores['RMSE']:.2f}", help="Like MAE, but large misses count more.")
    c3.metric("Test R²", f"{scores['R²']:.3f}", help="Share of the test variation the model explains.")
    c4.metric(
        "Train R²",
        f"{train_r2:.3f}",
        help="The same score on training data. A large gap from test R² suggests overfitting.",
    )

    with st.expander("Compare with all six models from the notebook"):
        st.dataframe(REG_NOTEBOOK, hide_index=True)
        st.caption(
            "LightGBM has the lowest error and the highest test R², so it was chosen as the final model."
        )

    st.subheader("Which features matter most?")
    importance = importance_table(model)
    st.bar_chart(importance.head(15).set_index("Feature")["Importance"].rename("Importance"))
    st.caption(
        "Importance counts how often the model split on a feature. It shows what the model relies on, "
        "not whether a feature raises demand."
    )

    st.subheader("Where does the model go wrong?")
    residuals = data["y_reg_test"].to_numpy() - y_pred_test
    st.write(
        "**Residual** = actual demand − predicted demand. Positive means the model under-predicted; "
        "negative means it over-predicted."
    )
    counts, edges = np.histogram(residuals, bins=30)
    centers = np.round((edges[:-1] + edges[1:]) / 2, 1)
    st.bar_chart(pd.Series(counts, index=centers, name="Sessions"))

    st.write("**Residual vs predicted demand.** A random cloud around zero is the ideal pattern.")
    sample = pd.DataFrame({"Predicted demand": y_pred_test, "Residual": residuals})
    sample = sample.sample(min(3000, len(sample)), random_state=42)
    st.scatter_chart(sample, x="Predicted demand", y="Residual")
    st.caption(
        "The spread widens as predicted demand rises, so errors are larger for high-demand sessions."
    )


def page_classification(df):
    st.title("Classification: will a session be High Demand?")
    st.write(
        "Based on **Classification.ipynb**. Each session is labelled High Demand (1) or Normal Demand (0). "
        "The label comes from the median demand, so the two classes are almost equal in size."
    )

    data = split_data()
    model, source = get_classification_model()
    st.info(
        f"Target: demand ≥ {data['threshold']:.2f} is **High Demand**. Anything below is **Normal Demand**. "
        "The demand column is used to create the label, but it is not an input to the model."
    )
    st.caption(f"Model: LightGBM classifier, loaded from {source}.")

    x_train, x_test = data["x_train"], data["x_test"]
    y_train, y_test = data["y_cls_train"], data["y_cls_test"]
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]
    scores = classification_scores(y_test, y_pred, y_proba)
    train_acc = accuracy_score(y_train, model.predict(x_train))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{scores['Accuracy']:.3f}")
    c2.metric("Precision", f"{scores['Precision']:.3f}", help="Of sessions predicted High, how many were High.")
    c3.metric("Recall", f"{scores['Recall']:.3f}", help="Of real High sessions, how many the model caught.")
    c4.metric("F1", f"{scores['F1']:.3f}", help="Balance between precision and recall.")
    c5.metric("ROC-AUC", f"{scores['ROC-AUC']:.3f}", help="How well the model separates the two classes.")
    st.caption(
        f"Train accuracy {train_acc:.3f} vs test accuracy {scores['Accuracy']:.3f}. "
        "A small gap means the model generalises well."
    )

    st.subheader("Confusion matrix")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Normal", "Actual High"],
        columns=["Predicted Normal", "Predicted High"],
    )
    st.dataframe(cm_df)
    st.caption("Diagonal cells are correct predictions. Off-diagonal cells are mistakes.")

    st.subheader("Which features matter most?")
    importance = importance_table(model)
    st.bar_chart(importance.head(20).set_index("Feature")["Importance"].rename("Importance"))

    with st.expander("Compare with all five models from the notebook"):
        st.dataframe(CLS_NOTEBOOK, hide_index=True)
        st.caption(
            "XGBoost and LightGBM are almost tied. LightGBM has the best F1 and ROC-AUC, "
            "so it was chosen. XGBoost has the highest accuracy and precision."
        )


def page_time_series(df):
    st.title("Time series: daily demand over time")
    st.write(
        "Based on **Time_Series.ipynb**. The sessions are added up into one total per day, and then "
        "the series is studied and forecast."
    )

    daily = daily_demand_table()
    c1, c2, c3 = st.columns(3)
    c1.metric("Days", f"{len(daily)}")
    c2.metric("First day", f"{daily['date'].min():%d %b %Y}")
    c3.metric("Last day", f"{daily['date'].max():%d %b %Y}")

    # 1. Daily, weekly and monthly views
    st.subheader("1. Demand over time")
    series = daily.set_index("date")["total_demand"]
    view = st.selectbox("Time scale", ["Daily", "Weekly", "Monthly"])
    if view == "Daily":
        st.line_chart(pd.DataFrame({"Actual": series, "7-day average": series.rolling(7).mean()}))
        st.caption("The 7-day average smooths out day-to-day spikes and shows the underlying direction.")
    elif view == "Weekly":
        st.bar_chart(series.resample("W-SUN").sum().rename("Total demand"))
        st.caption("The first week is incomplete because the data starts on Thursday 30 Sep 2021.")
    else:
        monthly = series.resample("ME").sum()
        monthly.index = monthly.index.strftime("%b %Y")
        st.bar_chart(monthly.rename("Total demand"))
        st.caption("Sep 2021 is partial because the data starts on 30 Sep. Treat it as incomplete, not low.")

    st.write("**Biggest one-day changes**")
    pct = series.pct_change() * 100
    top_up, top_down = pct.idxmax(), pct.idxmin()
    c1, c2 = st.columns(2)
    c1.metric(
        "Largest rise",
        f"{pct.max():+.1f}%",
        help=f"On {top_up:%d %b %Y}, demand was {series.loc[top_up]:,.0f}.",
    )
    c2.metric(
        "Largest drop",
        f"{pct.min():+.1f}%",
        help=f"On {top_down:%d %b %Y}, demand was {series.loc[top_down]:,.0f}.",
    )
    st.caption(
        "Percentage changes look extreme when the previous day was unusually small, so read them "
        "alongside the actual totals."
    )

    # 2. Time of day
    st.subheader("2. Demand by time of day")
    totals = df.groupby("time_of_day")["demand"].sum().sort_values(ascending=False)
    st.bar_chart(totals.rename("Total demand"))
    st.write("**Daily demand by time of day (7-day average)**")
    st.line_chart(daily_by_time_of_day().rolling(7).mean())
    st.caption("Evening is consistently the highest period.")

    # 3. Decomposition
    st.subheader("3. Trend, weekly pattern and noise")
    st.write(
        "Decomposition splits the daily series into **Observed ≈ Trend + Seasonal + Residual**. "
        "The seasonal part repeats every 7 days."
    )
    parts = decomposition_parts()
    left, right = st.columns(2)
    with left:
        st.write("**Observed**")
        st.line_chart(parts["Observed"])
        st.write("**Seasonal (weekly)**")
        st.line_chart(parts["Seasonal"])
    with right:
        st.write("**Trend**")
        st.line_chart(parts["Trend"])
        st.write("**Residual (noise)**")
        st.line_chart(parts["Residual"])

    # 4. Stationarity
    st.subheader("4. Is the series stationary?")
    st.write(
        "A stationary series keeps a stable level and spread over time. The ADF test checks this. "
        "A p-value of 0.05 or less means the series is stationary."
    )
    raw_stat, raw_p, diff_stat, diff_p = adf_results()
    adf_table = pd.DataFrame(
        {
            "Series": ["Original daily demand", "First difference (day-to-day change)"],
            "ADF statistic": [raw_stat, diff_stat],
            "p-value": [raw_p, diff_p],
            "Stationary?": ["Yes" if raw_p <= 0.05 else "No", "Yes" if diff_p <= 0.05 else "No"],
        }
    )
    st.dataframe(adf_table.round(4), hide_index=True)
    st.caption("The original series is not stationary, but the first difference is. That is why d = 1 for ARIMA.")

    # 5. Forecasts
    st.subheader("5. Forecast comparison")
    test_dates, y_test, forecasts = run_forecasts()
    split_day = len(daily) - len(y_test)
    st.caption(
        f"Train: {daily['date'].iloc[0]:%d %b %Y} to {daily['date'].iloc[split_day - 1]:%d %b %Y}. "
        f"Test: {test_dates[0]:%d %b %Y} to {test_dates[-1]:%d %b %Y} ({len(y_test)} days)."
    )

    rows = []
    for name, pred in forecasts.items():
        rows.append(
            {
                "Model": name,
                "MAE": mean_absolute_error(y_test, pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            }
        )
    score_table = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    st.dataframe(score_table.round(2), hide_index=True)
    st.info(
        "The 7-day moving average is updated with each real day as it arrives, so it always "
        "predicts one day ahead. ARIMA and exponential smoothing forecast all test days at once. "
        "Its low error is therefore not a like-for-like win for real forecasting."
    )

    choice = st.selectbox("Compare a model with the actual demand", list(forecasts.keys()))
    chart = pd.DataFrame(
        {"Actual": y_test, choice: forecasts[choice]},
        index=pd.DatetimeIndex(test_dates),
    )
    st.line_chart(chart)

    with st.expander("Compare with the notebook results"):
        st.dataframe(TS_NOTEBOOK.round(2), hide_index=True)


def page_predict(df):
    st.title("Predict a charging session")
    st.write(
        "Enter the details of a session. The regression model estimates its demand, and the "
        "classification model says whether it is likely to be High Demand."
    )

    data = split_data()
    reg_model, _ = get_regression_model()
    cls_model, _ = get_classification_model()

    locations = sorted(df["location"].unique().tolist())
    default_location = locations.index("apartment") if "apartment" in locations else 0
    default_date = df["start_full"].max().date()

    with st.form("predict_form"):
        left, right = st.columns(2)
        with left:
            session_date = st.date_input("Session date", value=default_date)
            session_time = st.time_input("Start time", value=time(18, 0))
            duration = st.number_input("Duration (minutes)", min_value=1, max_value=5000, value=120, step=5)
            location = st.selectbox("Location", locations, index=default_location)
        with right:
            charger_type = st.selectbox("Charger type", [0, 1], format_func=lambda v: f"Type {v}")
            charger_company = st.selectbox("Charger company", [0, 1], format_func=lambda v: f"Company {v}")
            user_prev = st.number_input("Previous sessions by this user", min_value=0, value=0, step=1)
            charger_prev = st.number_input("Previous sessions on this charger", min_value=0, value=0, step=1)
            location_prev = st.number_input("Previous sessions at this location", min_value=0, value=0, step=1)
        submitted = st.form_submit_button("Predict", type="primary")

    if not submitted:
        st.info("Fill in the details above and press **Predict**.")
        return

    start = datetime.combine(session_date, session_time)
    row = build_feature_row(
        start, duration, location, charger_type, charger_company,
        user_prev, charger_prev, location_prev,
    )

    # Demand is never negative, so the regression output is floored at zero.
    predicted_demand = max(0.0, float(reg_model.predict(row)[0]))
    high_probability = float(cls_model.predict_proba(row)[0, 1])
    label = "High Demand" if cls_model.predict(row)[0] == 1 else "Normal Demand"

    st.subheader("Result")
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted demand", f"{predicted_demand:.2f}")
    c2.metric("Predicted class", label)
    c3.metric("Chance of High Demand", f"{high_probability:.0%}")
    st.caption(
        f"High Demand means demand ≥ {data['threshold']:.2f}, the median in the training data. "
        "These are estimates from past sessions, so treat them as a guide."
    )

    with st.expander("See the features the model received"):
        st.dataframe(row.T.rename(columns={0: "Value"}))


def build_feature_row(start, duration, location, charger_type, charger_company,
                      user_prev, charger_prev, location_prev):
    """Builds one row with the same columns and rules as the notebooks."""
    weekday = start.weekday()  # Monday = 0, as in the notebooks
    hour = start.hour
    row = pd.DataFrame(
        [
            {
                "duration": float(duration),
                "start_month": start.month,
                "start_day": start.day,
                "start_hour": hour,
                "start_minute": start.minute,
                "day_of_week_number": weekday,
                "week_of_year": start.isocalendar()[1],
                "is_weekend": int(weekday >= 5),
                "user_previous_sessions": int(user_prev),
                "charger_previous_sessions": int(charger_prev),
                "location_previous_sessions": int(location_prev),
                "location": location,
                "time_of_day": time_of_day_label(hour),
                "chargertype": int(charger_type),
                "chargercompany": int(charger_company),
            }
        ]
    )
    return row[FEATURES]


PAGES = {
    "Overview": page_overview,
    "Regression": page_regression,
    "Classification": page_classification,
    "Time Series": page_time_series,
    "Predict Session": page_predict,
}


def main():
    df = load_data()
    if df is None:
        st.error(
            "Could not find `ev_charging_featured_model.csv`. Put it in a `data` folder "
            "next to `app.py`."
        )
        st.stop()

    with st.sidebar:
        st.header("⚡ EV Charging")
        choice = st.radio("Go to", list(PAGES.keys()))
        st.caption(f"{len(df):,} charging sessions loaded")

    PAGES[choice](df)


main()
