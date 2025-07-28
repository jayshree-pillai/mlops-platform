from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.utils.s3_loader import load_npy_from_s3
from src.models.base_model import train_model

model_map = {
    "logreg": LogisticRegression,
    "cart": DecisionTreeClassifier,
    "rf": RandomForestClassifier,
    "xgb": XGBClassifier,
}

def run_training(config):
    model_type = config.model_type
    params = config.params
    version = config.version
    run_mode = config.run_mode

    print(f"Training {model_type} | version={version} | mode={run_mode}")

    X_train = load_npy_from_s3("train_X.npy")
    y_train = load_npy_from_s3("train_y.npy")
    X_val = load_npy_from_s3("val_X.npy")
    y_val = load_npy_from_s3("val_y.npy")

    model_cls = model_map.get(model_type)
    if model_cls is None:
        raise ValueError(f"Unknown model_type: {model_type}")

#    model = model_cls(**params)
#    train_model(model, model_type, X_train, y_train, X_val, y_val, params, run_source=run_mode)

    grid = GridSearchCV(model_cls(), config.param_grid, cv=3, scoring='roc_auc')
    print("Running GridSearchCV...")
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    best_params = grid.best_params_

    train_model(best_model, model_type, X_train, y_train, X_val, y_val, best_params, run_source=run_mode)

    print("Training complete.")
