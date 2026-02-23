"""
train_model.py
Phase 3 - Modeling (CRISP-ML(Q))
Models: ARIMA (baseline), XGBoost, SVM — as committed in mid-review
Group 05 - ISB AMPBA
"""

import numpy as np
import pandas as pd
import pickle, os, sys, warnings, shutil
warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ARIMA_ORDER, XGBOOST_PARAMS, SVM_PARAMS
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor

def calc_metrics(y_true, y_pred, name):
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = np.mean(np.abs(y_true - y_pred))
    print(f"\n  📊 {name}")
    print(f"     MAPE : {mape:.2f}%  (target <= 5%)")
    print(f"     RMSE : {rmse:.2f} Rs/MWh")
    print(f"     MAE  : {mae:.2f} Rs/MWh")
    return {'model': name, 'MAPE': mape, 'RMSE': rmse, 'MAE': mae}

def train_arima(y_train, y_test):
    print("\n=== BASELINE: ARIMA", ARIMA_ORDER, "===")
    try:
        from statsmodels.tsa.arima.model import ARIMA
        fitted = ARIMA(y_train, order=ARIMA_ORDER).fit()
        y_pred = np.maximum(fitted.forecast(steps=len(y_test)), 0)
        with open('models/arima_model.pkl','wb') as f: pickle.dump(fitted, f)
        print("  Saved: models/arima_model.pkl")
    except ImportError:
        print("  statsmodels not installed — using naive mean baseline")
        y_pred = np.full(len(y_test), np.mean(y_train))
        with open('models/arima_model.pkl','wb') as f: pickle.dump({'mean':float(np.mean(y_train))},f)
    return calc_metrics(y_test, y_pred, "ARIMA (Baseline)")

def train_xgboost(X_train, X_test, y_train, y_test):
    print("\n=== ML MODEL 1: XGBoost ===")
    try:
        from xgboost import XGBRegressor
        model = XGBRegressor(**XGBOOST_PARAMS, verbosity=0)
        print("  Using: XGBoost library")
    except ImportError:
        model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, random_state=42)
        print("  Using: GradientBoostingRegressor (XGBoost equivalent)")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    with open('models/xgboost_model.pkl','wb') as f: pickle.dump(model, f)
    print("  Saved: models/xgboost_model.pkl")
    return model, calc_metrics(y_test, y_pred, "XGBoost")

def train_svm(X_train, X_test, y_train, y_test):
    print("\n=== ML MODEL 2: SVM ===")
    max_s = 5000
    if len(X_train) > max_s:
        print(f"  Using {max_s:,} samples (SVM subset for speed)")
        idx = np.random.choice(len(X_train), max_s, replace=False)
        Xtr, ytr = X_train[idx], y_train[idx]
    else:
        Xtr, ytr = X_train, y_train
    model = SVR(**SVM_PARAMS)
    model.fit(Xtr, ytr)
    y_pred = model.predict(X_test)
    with open('models/svm_model.pkl','wb') as f: pickle.dump(model, f)
    print("  Saved: models/svm_model.pkl")
    return model, calc_metrics(y_test, y_pred, "SVM")

def compare_and_select(all_metrics):
    print("\n" + "="*50)
    print("  MODEL COMPARISON — Group 05")
    print("="*50)
    df = pd.DataFrame(all_metrics).sort_values('MAPE')
    print(df.to_string(index=False))
    baseline = df[df['model'].str.contains('ARIMA|Naive')]['MAPE'].values[0]
    best     = df[~df['model'].str.contains('ARIMA|Naive')].iloc[0]
    impr     = (baseline - best['MAPE']) / baseline * 100
    print(f"\n  Best model : {best['model']}")
    print(f"  MAPE       : {best['MAPE']:.2f}%  {'SUCCESS' if best['MAPE']<=5 else 'CHECK'}")
    print(f"  Improvement: {impr:.1f}% over ARIMA  {'SUCCESS' if impr>=25 else 'CHECK'}")
    df.to_csv('models/model_comparison.csv', index=False)
    return best['model']

def save_best(best_name):
    src = 'models/xgboost_model.pkl'
    if 'SVM' in best_name:   src = 'models/svm_model.pkl'
    if 'ARIMA' in best_name: src = 'models/arima_model.pkl'
    shutil.copy(src, 'models/best_model.pkl')
    print(f"\n  Best model saved: models/best_model.pkl")

if __name__ == "__main__":
    print("TRAINING PIPELINE — Group 05 IEX Forecasting")
    print("="*50)
    X_train = np.load('models/X_train.npy')
    X_test  = np.load('models/X_test.npy')
    y_train = np.load('models/y_train.npy')
    y_test  = np.load('models/y_test.npy')
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")

    results = []
    results.append(train_arima(y_train, y_test))
    _, m = train_xgboost(X_train, X_test, y_train, y_test); results.append(m)
    _, m = train_svm(X_train, X_test, y_train, y_test);     results.append(m)
    best = compare_and_select(results)
    save_best(best)
    print("\nModel training complete!")
