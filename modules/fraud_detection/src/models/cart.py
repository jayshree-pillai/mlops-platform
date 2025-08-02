import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("fraud_cart")


import sys, os
sys.path.append(os.path.abspath("modules/fraud_detection/src"))

from models.base_model import train_model
from utils.s3_loader import load_npy_from_s3
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# --- Load data ---
X_train = load_npy_from_s3("train_X.npy")
y_train = load_npy_from_s3("train_y.npy")
X_val = load_npy_from_s3("val_X.npy")
y_val = load_npy_from_s3("val_y.npy")

print("Starting CART training...")  # add at top
# --- Hyperparameter tuning ---
param_grid = {
    "max_depth": [5],#, 10, 20],
    "min_samples_split": [2],#, 10, 50]
}
grid = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=3, scoring='roc_auc')
print("GridSearchCV fitting...")    # before grid.fit
grid.fit(X_train, y_train)
print("GridSearch done. Best params:", grid.best_params_)
# --- Train best model on full train set (re-fit already done) ---
best_model = grid.best_estimator_
best_params = grid.best_params_

train_model(best_model, "decision_tree", X_train, y_train, X_val, y_val, best_params)
