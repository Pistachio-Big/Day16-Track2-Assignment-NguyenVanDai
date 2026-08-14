#!/usr/bin/env python3
"""
LAB 16 - Cloud AI Environment Setup (Oracle OCI, luong CPU/LightGBM)
Dataset: Credit Card Fraud Detection (mlg-ulb/creditcardfraud, 284,807 giao dich)
Train + inference LightGBM, do hieu nang, ghi ket qua ra benchmark_result.json.

Toi uu cho instance RAM thap (E2.1.Micro 1GB): doc CSV theo dtype float32.
"""
import json
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

DATA_PATH = "creditcard.csv"  # doi neu file o cho khac
RESULT_PATH = "benchmark_result.json"

results = {}

# 1) Load data (dtype float32 de tiet kiem RAM)
t0 = time.time()
df = pd.read_csv(DATA_PATH)
for col in df.columns:
    if df[col].dtype == "float64":
        df[col] = df[col].astype("float32")
load_time = time.time() - t0
results["load_data_seconds"] = round(load_time, 3)
print(f"[1] Loaded {len(df):,} rows in {load_time:.3f}s")

X = df.drop(columns=["Class"])
y = df["Class"].astype("int8")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2) Train LightGBM
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=neg / pos,  # xu ly mat can bang lop (fraud rat hiem)
    n_jobs=-1,
    random_state=42,
)
t0 = time.time()
model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="auc",
    callbacks=[early_stopping(50), log_evaluation(50)],
)
train_time = time.time() - t0
results["training_seconds"] = round(train_time, 3)
results["best_iteration"] = int(model.best_iteration_)
print(f"[2] Trained in {train_time:.3f}s, best_iteration={model.best_iteration_}")

# 3) Danh gia
proba = model.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)
results["auc_roc"] = round(float(roc_auc_score(y_test, proba)), 4)
results["accuracy"] = round(float(accuracy_score(y_test, pred)), 4)
results["f1_score"] = round(float(f1_score(y_test, pred)), 4)
results["precision"] = round(float(precision_score(y_test, pred)), 4)
results["recall"] = round(float(recall_score(y_test, pred)), 4)
print(f"[3] AUC-ROC={results['auc_roc']}  F1={results['f1_score']}  "
      f"Precision={results['precision']}  Recall={results['recall']}")

# 4) Inference latency (1 row) - trung binh 100 lan
one = X_test.iloc[[0]]
for _ in range(10):
    model.predict(one)  # warm-up
t0 = time.time()
N = 100
for _ in range(N):
    model.predict(one)
latency_ms = (time.time() - t0) / N * 1000
results["inference_latency_ms_1row"] = round(latency_ms, 3)

# 5) Throughput (1000 rows)
batch = X_test.iloc[:1000]
t0 = time.time()
model.predict(batch)
batch_time = time.time() - t0
results["inference_throughput_rows_per_sec"] = round(1000 / batch_time, 1)
print(f"[4] Latency(1 row)={latency_ms:.3f} ms  "
      f"Throughput(1000)={results['inference_throughput_rows_per_sec']:.1f} rows/s")

# 6) Ghi ket qua
with open(RESULT_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"[5] Saved -> {RESULT_PATH}")
print(json.dumps(results, indent=2))
