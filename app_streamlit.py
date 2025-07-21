# app_streamlit.py

import streamlit as st
import joblib
from datetime import datetime
import pandas as pd
import numpy as np
import cloudpickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit, cross_validate, KFold
from sklearn.ensemble import StackingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Feature transformer: date parsing + fixed ratio features
class FeatureTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        dates = pd.to_datetime(X['Date Sold'], dayfirst=True)
        X['day']            = dates.dt.day
        X['month']          = dates.dt.month
        X['year']           = dates.dt.year
        X['age']            = X['year'] - X['Year Built']
        X['bed_bath_ratio'] = X['Bedrooms'] / X['Bathrooms']
        X['size_bed_ratio'] = X['Size'] / X['Bedrooms']
        start = dates.min()
        X['days_since_start'] = (dates - start).dt.days
        X['time_sq']         = X['days_since_start'] ** 2
        X['month_sin']       = np.sin(2 * np.pi * X['month'] / 12)
        X['month_cos']       = np.cos(2 * np.pi * X['month'] / 12)
        return X.drop(columns=['Date Sold'])


@st.cache(allow_output_mutation=True)
def load_model():
    with open("stacking_model_cpkl.pkl", "rb") as f:
        return cloudpickle.load(f)

@st.cache
def load_data():
    # adjust path if needed
    df = pd.read_excel("Case Study 1 Data.xlsx")
    # drop unused cols
    df = df.drop(columns=["Property ID", "Price"]).dropna(subset=["Location"])
    return df

model = load_model()
df = load_data()

st.title("🏠 House Price Predictor")

st.sidebar.header("Input Features")

# Categorical inputs from training data
location = st.sidebar.selectbox("Location", df["Location"].unique())
condition = st.sidebar.selectbox("Condition", df["Condition"].unique())
ptype = st.sidebar.selectbox("Property Type", df["Type"].unique())

# Numeric inputs
size = st.sidebar.number_input("Size (sqft)", min_value=0.0, value=1000.0, step=50.0)
bedrooms = st.sidebar.number_input("Bedrooms", min_value=0, value=3, step=1)
bathrooms = st.sidebar.number_input("Bathrooms", min_value=0, value=2, step=1)
year_built = st.sidebar.number_input(
    "Year Built",
    min_value=1900,
    max_value=datetime.now().year,
    value=2010,
    step=1
)
date_sold = st.sidebar.date_input(
    "Date Sold",
    value=datetime.now()
)

# Assemble into a single‑row DataFrame
input_df = pd.DataFrame({
    "Location":      [location],
    "Size":          [size],
    "Bedrooms":      [bedrooms],
    "Bathrooms":     [bathrooms],
    "Year Built":    [year_built],
    "Condition":     [condition],
    "Type":          [ptype],
    "Date Sold":     pd.to_datetime([date_sold])
})

st.subheader("Feature Preview")
st.write(input_df)

# Predict button
if st.sidebar.button("Predict Price"):
    pred = model.predict(input_df)[0]
    st.subheader("🔮 Predicted Price")
    st.success(f"₹ {pred:,.0f}")
else:
    st.write("Adjust the inputs and click **Predict Price** in the sidebar.")
