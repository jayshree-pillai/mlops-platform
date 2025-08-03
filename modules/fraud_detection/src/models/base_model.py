from src.models.evaluate import log_and_report


def train_model(model, model_name, X_train, y_train, X_val, y_val, params=None,run_source="manual", processor_path="feature_processor.pkl"):
    model.fit(X_train, y_train)
    log_and_report(model, model_name, X_val, y_val, params,run_source)
