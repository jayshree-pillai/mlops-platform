from src.utils.mlflow_utils import init_mlflow
import mlflow
init_mlflow()
mlflow.set_experiment("fraud_logreg")

import sys, os
sys.path.append(os.path.abspath("modules/fraud_detection/src"))

from sklearn.linear_model import LogisticRegression
from src.models.base_model import train_model
from utils.s3_loader import load_npy_from_s3

# --- Load data ---
X_train = load_npy_from_s3("train_X.npy")
y_train = load_npy_from_s3("train_y.npy")
X_val = load_npy_from_s3("val_X.npy")
y_val = load_npy_from_s3("val_y.npy")

# --- Train ---
params = {"max_iter": 1000}
model = LogisticRegression(**params)
train_model(model, "logistic_regression", X_train, y_train, X_val, y_val, params)
