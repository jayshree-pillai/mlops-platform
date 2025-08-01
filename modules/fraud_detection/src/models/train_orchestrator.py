from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from src.utils.feature_store_logger import log_features_to_store
from src.features.feature_loader import load_features_from_fg
from src.utils.s3_loader import load_npy_from_s3

from src.utils.s3_loader import load_npy_from_s3
from src.models.base_model import train_model
from src.features.feature_processor import FeatureProcessor

model_map = {
    "logreg": LogisticRegression,
    "cart": DecisionTreeClassifier,
    "rf": RandomForestClassifier,
    "xgb": XGBClassifier,
}

def run_training(config, model=None):
    model_type = config.model_type
    params = config.params
    version = config.version
    run_mode = config.run_mode

    print(f"Training {model_type} | version={version} | mode={run_mode}")

    if config.get("retrain_mode", False):
        print("🔁 Retraining: loading training features from Feature Store...")
        X_train, y_train = load_features_from_fg(source="athena", split="train")
    else:
        print("🧪 Fresh training: loading training data from S3 .npy...")
        X_train = load_npy_from_s3("train_X.npy")
        y_train = load_npy_from_s3("train_y.npy")
    # Validation always from .npy for now
    X_val = load_npy_from_s3("val_X.npy")
    y_val = load_npy_from_s3("val_y.npy")

    processor = FeatureProcessor()
    processor.fit(X_train)
    processor.save("feature_processor.pkl")

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

    if model is None:
        model_cls = model_map.get(model_type)
        model = model_cls(**params)
        print("🛠️  No staging model found. Using fresh model.")
    else:
        print("✅ Using model loaded from staging.")

    train_model(best_model, model_type, X_train, y_train, X_val, y_val, best_params, run_source=run_mode)

    print("Training complete.")

    # After training is complete and processor saved
    model_id = log_features_to_store(
        X_train, y_train,
        bucket="mlops-fraud-dev",
        s3_prefix="feature_store/best_model_runs",
        processor=processor
    )

    # 🔥 Trigger Glue crawler to update table metadata
    import boto3
    glue = boto3.client("glue")
    glue.start_crawler(Name="fraud_featurestore_crawler")  # 🔁 match your TF name
    print("🧹 Glue crawler triggered to refresh schema.")

