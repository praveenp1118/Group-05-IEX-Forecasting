"""
run_pipeline.py
Master pipeline script — runs everything end to end
Group 05 - ISB AMPBA - Final Submission
"""
import os, sys
os.makedirs('models', exist_ok=True)

print("="*60)
print("  IEX ELECTRICITY PRICE FORECASTING — GROUP 05")
print("  ISB AMPBA | CRISP-ML(Q) Complete Pipeline")
print("="*60)

# Phase 2: Data Preparation
print("\n PHASE 2 — DATA PREPARATION")
from src.data_preparation import (load_data, check_data_quality,
    resample_to_15min, clean_data, feature_engineering, prepare_train_test)
import numpy as np

df = load_data()
df = check_data_quality(df)
df = resample_to_15min(df)
df = clean_data(df)
df = feature_engineering(df)
prepare_train_test(df)

# Phase 3: Model Training
print("\n PHASE 3 — MODEL TRAINING")
from src.train_model import (train_arima, train_xgboost, train_svm,
                              compare_and_select, save_best)
X_train = np.load('models/X_train.npy')
X_test  = np.load('models/X_test.npy')
y_train = np.load('models/y_train.npy')
y_test  = np.load('models/y_test.npy')

results = []
results.append(train_arima(y_train, y_test))
_, m = train_xgboost(X_train, X_test, y_train, y_test); results.append(m)
_, m = train_svm(X_train, X_test, y_train, y_test);     results.append(m)
best = compare_and_select(results)
save_best(best)

print("\n" + "="*60)
print("  PIPELINE COMPLETE!")
print("="*60)
print("\nTo start Flask API:")
print("  cd app && python app.py")
print("Then open: http://localhost:5000")
print("Demo prediction: http://localhost:5000/predict/sample")
