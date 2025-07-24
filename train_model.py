# train_model.py

import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
import numpy as np
import cloudpickle
from scipy.stats import zscore
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit, cross_validate, KFold
from sklearn.metrics import make_scorer, r2_score, mean_absolute_percentage_error
from sklearn.ensemble import StackingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import joblib

# Progress tracking
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib

# GPU detection
def detect_gpu():
    try:
        import subprocess
        subprocess.check_output(['nvidia-smi'], stderr=subprocess.STDOUT)
        return True
    except Exception:
        return False

USE_GPU = detect_gpu()
USE_GPU = False # Force CPU for compatibility and model inference using CPU
print(f"GPU detected: {USE_GPU}")

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

# Main training
def main():
    # Load data
    df = pd.read_excel("Case Study 1 Data.xlsx")
    df = df.drop(columns=['Property ID'])
    df = df.dropna()
    # Remove outliers
    df['price_z'] = zscore(df['Price'])
    df = df[df['price_z'].abs() <= 3].drop(columns=['price_z'])
    # split features and target
    X = df.drop(columns=['Price'])
    y = df['Price']

    # Preprocessor
    numeric_feats = [
        'Size', 'Bedrooms', 'Bathrooms', 'Year Built',
        'day', 'month', 'year', 'age',
        'bed_bath_ratio', 'size_bed_ratio',
        'days_since_start', 'time_sq', 'month_sin', 'month_cos'
    ]
    cat_feats = ['Location', 'Condition', 'Type']
    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median'))])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipe, numeric_feats),
        ('cat', cat_pipe, cat_feats)
    ], remainder='drop')

    # Model parameters
    lgb_params = {
        'random_state': 42,
        'n_jobs': -1,
        'n_estimators': 200,
        'device': 'gpu' if USE_GPU else 'cpu',
        'verbosity': -1,
        'objective': 'mape'
    }
    xgb_params = {
        'random_state': 42,
        'use_label_encoder': False,
        'n_jobs': -1,
        'n_estimators': 200,
        'device': 'cuda' if USE_GPU else 'cpu',
        'tree_method': 'hist',
        'eval_metric': 'mape'
    }
    cat_params = {
        'random_state': 42,
        'iterations': 200,
        'thread_count': -1,
        'task_type': 'GPU' if USE_GPU else 'CPU',
        'devices': '0',
        'verbose': False,
        'loss_function': 'MAPE'
    }
    hgb_params = {'random_state': 42, 'max_iter': 200,'loss': 'absolute_error'}
    ridge_params = {'alpha': 1.0}

    # Instantiate learners
    estimators = [
        ('lgbm', LGBMRegressor(**lgb_params)),
        ('xgb',  XGBRegressor(**xgb_params)),
        ('cat',  CatBoostRegressor(**{k:v for k,v in cat_params.items() if v is not None})),
        ('hgb',  HistGradientBoostingRegressor(**hgb_params))
    ]
    final_est = Ridge(**ridge_params)

    # Stacking ensemble (serialize to avoid parallel conflicts)
    stack = StackingRegressor(
        estimators=estimators,
        final_estimator=final_est,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=1,
        passthrough=True
    )


    # Pipeline (no log‑transform)
    model = Pipeline([
        ('feat', FeatureTransformer()),
        ('prep', preprocessor),
        ('stack', stack)
    ])

    # Cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    scoring = {
        'MAE': 'neg_mean_absolute_error',
        'R2': make_scorer(r2_score),
        'MAPE': make_scorer(mean_absolute_percentage_error)
    }
    cv_res = cross_validate(model, X, y, cv=tscv, scoring=scoring, n_jobs=1)
    print("CV Results:")
    print(f" MAE : {-cv_res['test_MAE'].mean():.2f}")
    print(f" R2  : {cv_res['test_R2'].mean():.3f}")
    print(f" MAPE: {cv_res['test_MAPE'].mean()*100:.2f}%")

    # Final fit & save
    model.fit(X, y)
    with open("stacking_model.pkl", "wb") as f:
        cloudpickle.dump(model, f)
    print("Model serialized to stacking_model.pkl with cloudpickle")
    # joblib.dump(model, "stacking_model_final.pkl", compress=3)
    # print("Model trained and saved to stacking_model.pkl")

if __name__ == '__main__':
    main()
