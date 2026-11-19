# EV-Charging-Demand-Prediction-and-Usage-Pattern-Analysis-
EV Charging demand prediction

This project studies how electric vehicle (EV) charging sessions behave and builds machine learning models to predict charging demand. It ends with an interactive Streamlit dashboard that presents the analysis, model comparisons, forecasts and single-session predictions.

## Objectives

- Understand charging patterns by time of day, location, charger type and charger company.
- Predict the energy demand of a charging session (regression).
- Predict whether a session will be a **High Demand** session (classification).
- Forecast daily total charging demand (time series).
- Present everything in a single dashboard.

## Project structure and folder

```
EV Charging Demand Session/
├── app.py                          # Streamlit dashboard
├── README.md                       # This file
├── EDA.ipynb                       # Exploratory data analysis and feature engineering
├── Regression.ipynb                # Regression models for demand
├── Classification.ipynb            # Classification models for High Demand
├── Time_Series.ipynb               # Daily demand trends and forecasting
├── data/
│   ├── ChargingRecords_cleaned.csv     # Cleaned raw sessions (input to EDA)
│   └── ev_charging_featured.csv        # Feature-engineered sessions used by all models
├── models/
│   ├── lightgbm_regression.pkl         # Saved regression pipeline
│   ├── lightgbm_classification.pkl     # Saved classification pipeline
└── images/                         # Graphs saved by the notebooks (if not copied to models/)
    ├── *.png                           # Graphs saved by the notebooks for the dashboard
```

The notebooks use relative paths such as `data/` and `images/`, so run them from the project root.

## Dataset

- **Source:** EV charging session records (`ChargingRecords.csv`).
- **Cleaning:** Timestamps are converted to datetime values, Removed the duplicates and Composite primary key uniqueness and duration and demand outliers are flagged but kept. Cleaned dataset saved to (`ChargingRecords_cleaned.csv`).
- **Feature-engineered data:** Created historical session columns and saved the dataset to file named (`ev_charging_featured.csv`). 72,241 sessions with 17 columns.

Main columns:

| Column | Meaning |
|---|---|
| `duration` | Length of the session in minutes |
| `demand` | Energy demand of the session (target for regression) |
| `start_full` | Start date and time |
| `start_hour`, `start_day`, `start_minute`, `week_of_year` | Calendar and time features |
| `day_name`, `month_name`, `day_type`, `time_of_day` | Categorical time features |
| `location` | Apartment, public area, resort, hotel, bus garage and others |
| `chargertype`, `chargercompany` | Charger type (0/1) and company (0/1) |
| `user_previous_sessions`, `charger_previous_sessions`, `location_previous_sessions` | How many earlier sessions the user, charger or location had. Counted in time order to avoid using future data. |

**Target for classification:** a session is **High Demand** (1) if its demand is at or above the median demand (14.25), and **Normal Demand** (0) otherwise. The classes are almost exactly balanced (50.03% and 49.97%).

## Method

All models use a **chronological 80/20 split**: the earliest 80% of sessions train the model and the latest 20% test it. Data is never shuffled, so the models are always tested on later sessions than the ones they learned from.

Preprocessing uses a scikit-learn `Pipeline`. Numeric features are standardized, and categorical features are one-hot encoded. Fitting happens on the training data only, which prevents information from the test set leaking into the model.

## Notebooks

| Notebook | What it does |
|---|---|
| `EDA.ipynb` | Univariate, bivariate, multivariate and correlation analysis. Log transforms for skewed variables. Creates calendar, time-of-day and historical features. Saves the feature-engineered dataset. |
| `Regression.ipynb` | Predicts `demand` with six models: Linear Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost and LightGBM. Includes residual analysis and feature importance. Saves the LightGBM pipeline. |
| `Classification.ipynb` | Predicts High Demand with five models: Logistic Regression, Decision Tree, Random Forest, XGBoost and LightGBM. Includes train/test overfitting checks, confusion matrices, ROC curves and feature importance. Saves the LightGBM pipeline. |
| `Time_Series.ipynb` | Converts sessions into 366 daily totals. Covers trend, weekly seasonality, stationarity (ADF test), first differencing, and forecasting with a 7-day moving average, exponential smoothing, ARIMA(1,1,1) and a naive baseline. |

## Results

**Regression (predicting session demand, test period)**

| Model | MAE ↓ | RMSE ↓ | Test R² ↑ |
|---|---:|---:|---:|
| Linear Regression | 7.1171 | 10.6125 | 0.4158 |
| Decision Tree | 4.9781 | 9.4530 | 0.5365 |
| Random Forest | 4.5852 | 7.7395 | 0.6893 |
| Gradient Boosting | 4.2880 | 7.5216 | 0.7065 |
| XGBoost | 4.2383 | 7.4815 | 0.7097 |
| **LightGBM** | **4.1510** | **7.3873** | **0.7169** |

LightGBM is the best regression model. Duration and previous-session counts are the most important features.

**Classification (High Demand prediction, test period)**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8276 | 0.8265 | 0.8200 | 0.8232 | 0.8881 |
| Decision Tree | 0.8612 | 0.8379 | 0.8885 | 0.8624 | 0.8958 |
| Random Forest | 0.8750 | 0.8286 | 0.9389 | 0.8803 | 0.9339 |
| XGBoost | 0.8860 | 0.8547 | 0.9244 | 0.8881 | **0.9505** |
| **LightGBM** | **0.8880** | **0.8561** | 0.9269 | **0.8901** | 0.9496 |

LightGBM has the best accuracy and F1 score. XGBoost has the best ROC-AUC, so it separates the two classes slightly better overall.

**Time series (daily demand forecasting, last 74 days)**

| Model | MAE | RMSE |
|---|---:|---:|
| 7-Day Moving Average (walk-forward) | 495.04 | 686.51 |
| ARIMA(1,1,1) | 568.27 | 788.70 |
| Naive Baseline | 629.47 | 859.25 |
| Exponential Smoothing (trend) | 784.00 | 1028.75 |
| Exponential Smoothing (trend + weekly) | 790.79 | 1028.69 |

The 7-day moving average is scored one day ahead using real test values, while the other models forecast the whole test period at once. Its lower error is therefore not a direct like-for-like comparison.

## Key findings

- Demand and duration are right-skewed. Most sessions are short or moderate, with a few very long or high-demand sessions.
- Most charging starts between 17:00 and 19:00. Very early morning hours have the fewest sessions.
- Longer sessions generally have higher demand, but duration alone does not explain demand.
- Charger Type 1 sessions usually have higher demand and are shorter than Type 0 sessions.
- Daily demand has a clear upward trend and a repeating weekly pattern.

## Limitations

- **Duration is an input.** It is only known when a session ends. For real-time prediction at the start of a session, duration has to be estimated.
- **Locations with few sessions** (such as bus garage and accommodation) give less reliable statistics.
- **Clipped plots:** the duration histogram may be capped at 600 minutes for readability. Read the caption when interpreting it.
- **Time-series evaluation** uses a single 74-day test period, so the results depend on that period.

## Project Setup

Requires Python 3.10 or newer.

### 1. Clone the repository
```bash
git clone https://github.com/pavi31guvi/EV-Charging-Demand-Prediction-and-Usage-Pattern-Analysis-.git
cd EV-Charging-Demand-Prediction-and-Usage-Pattern-Analysis
```
### 2. Install dependency
```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

`requirements.txt`:

```
streamlit
pandas
numpy
plotly
scikit-learn
lightgbm
xgboost
statsmodels
matplotlib
seaborn
```

The saved `.pkl` models must be loaded with the same scikit-learn and LightGBM versions used to train them. If loading fails with a version warning, reinstall those versions or retrain the models by running the notebooks.

## Running the notebooks

Open the notebooks in Jupyter or VS Code from the project root and run them in order: `Cleaning.ipynb`, `EDA.ipynb`, then `Regression.ipynb`, `Classification.ipynb`, and `Time_Series.ipynb`. The EDA notebook creates `data/ev_charging_featured.csv`, which the other notebooks read. The regression and classification notebooks save the `.pkl` models into `models/`. The graph images are saved into `images/` folders.

## Running the dashboard

```bash
streamlit run app.py
```

The dashboard opens in your browser with four tabs:

1. **Usage patterns:** key metrics for all sessions (sessions, total demand, mean and median demand, mean duration, high-demand share).
2. **Model comparison:** results tables with the best model in bold, a plain-English explanation of the best model, the correlation with demand, residual analysis, confusion matrix, ROC curve and feature importance.
3. **Time Forecast:** daily, monthly and time-of-day demand patterns, the forecast comparison, and the forecast graphs.
4. **Predict session:** enter a planned session (date, time, duration, location, charger, and previous-session counts) to get a predicted demand, a High Demand probability and a predicted class.

The graphs shown in the dashboard are the PNG files saved by the notebooks. The app looks for them in `models/`, `models/images/` and `images/`. Graph sizes can be changed in the `GRAPH_WIDTHS` table at the top of `app.py`.
