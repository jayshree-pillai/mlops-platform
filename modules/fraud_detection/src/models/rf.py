import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("fraud_rf")

import sys, os
sys.path.append(os.path.abspath("modules/fraud_detection/src"))

from src.models.base_model import train_model
from utils.s3_loader import load_npy_from_s3
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

X_train = load_npy_from_s3("train_X.npy")
y_train = load_npy_from_s3("train_y.npy")
X_val = load_npy_from_s3("val_X.npy")
y_val = load_npy_from_s3("val_y.npy")
X_train = X_train[:50000]
y_train = y_train[:50000]
print("Starting RF training...")

param_grid = {
    "n_estimators": [100],#, 200],
    "max_depth": [10],#, 20],
    "min_samples_split": [2]#, 10]
}
grid = GridSearchCV(RandomForestClassifier(n_jobs=-1), param_grid, cv=3, scoring='roc_auc',n_jobs=-1,pre_dispatch="2*n_jobs"  )
print("GridSearchCV fitting...")
grid.fit(X_train, y_train)
print("GridSearch done. Best params:", grid.best_params_)

best_model = grid.best_estimator_
best_params = grid.best_params_

train_model(best_model, "random_forest", X_train, y_train, X_val, y_val, best_params)
